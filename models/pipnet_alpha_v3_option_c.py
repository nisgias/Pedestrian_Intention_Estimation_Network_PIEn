# models/pipnet_alpha_v3_option_c.py
"""
PIPNet-Alpha V3 — Option C: PIP-Net Faithful Fusion

Based on the ACTUAL PIP-Net diagram (Fig. 3):
  - Global branch does NOT have its own attention
  - Global GRU output is CONCATENATED with Local Attention output
  - A JOINT attention operates on the fused sequence
  - No cross-attention at all

Pipeline:
  1. Kinematic → hierarchical GRU → Attention → z_kin [B, 128]
  2. Local → content + motion GRU → fuse GRU → Attention → z_local [B, 128]
  3. Global → Conv2D → gradual reduce → FC → GRU → h_global [B, T, 256]
  4. ★ PIP-Net Fusion ★
     expand z_local → [B, T, 128]
     concat(z_local_expanded, h_global) → [B, T, 384]
     Joint Attention → z_visual [B, 128]
  5. Stack [z_visual, z_kin] → Final Attention → z_final [B, 256]
  6. FC → logit

Key differences from your current model:
  - NO CrossAttention module (removed entirely)
  - NO residual fusion
  - Global branch uses gradual pooling (8→4→2→1) not one-shot AvgPool
  - Fusion is concat + joint attention (exactly like PIP-Net diagram)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# ATTENTION (same Luong-style as before)
# ============================================================

class PIPNetAttention(nn.Module):
    """Luong-style temporal attention with diagonal Wc initialization."""
    def __init__(self, hidden_dim: int, out_dim: int, dropout_p: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.Wp = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wc = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        self._init_diagonal()
    
    def _init_diagonal(self):
        with torch.no_grad():
            out_dim, in_dim = self.Wc.weight.shape
            half_in = in_dim // 2
            
            self.Wc.weight.fill_(0.0)
            scale = 1.0 / math.sqrt(in_dim)
            
            for i in range(min(out_dim, half_in)):
                self.Wc.weight[i, i] = scale
                self.Wc.weight[i, half_in + i] = scale
            
            if out_dim > half_in:
                for i in range(half_in, out_dim):
                    j = i % half_in
                    self.Wc.weight[i, j] = scale * 0.5
                    self.Wc.weight[i, half_in + j] = scale * 0.5
            
            if self.Wc.bias is not None:
                self.Wc.bias.fill_(0.0)
    
    def forward(self, h: torch.Tensor, use_mean_query: bool = False):
        if use_mean_query:
            hm = h.mean(dim=1)
        else:
            hm = h[:, -1, :]
        
        h_proj = self.Wp(h)
        scores = torch.bmm(h_proj, hm.unsqueeze(-1)).squeeze(-1)
        alpha = F.softmax(scores, dim=-1)
        hc = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        concat = torch.cat([hc, hm], dim=-1)
        out = torch.tanh(self.Wc(concat))
        out = self.dropout(out)
        return out, alpha


# ============================================================
# BRANCHES
# ============================================================

class KinematicBranch(nn.Module):
    """SFRNN-style hierarchical: pose → (+bbox) → (+speed) — UNCHANGED"""
    def __init__(self, bbox_dim=4, pose_dim=34, speed_dim=1, hidden_dim=128):
        super().__init__()
        self.pose_gru = nn.GRU(pose_dim, 64, batch_first=True)
        self.pose_bbox_gru = nn.GRU(64 + bbox_dim, 128, batch_first=True)
        self.full_gru = nn.GRU(128 + speed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, bbox, pose, speed):
        h_pose, _ = self.pose_gru(pose)
        h_pb = torch.cat([h_pose, bbox], dim=-1)
        h_pb, _ = self.pose_bbox_gru(h_pb)
        h_full = torch.cat([h_pb, speed], dim=-1)
        h_out, _ = self.full_gru(h_full)
        return self.norm(h_out)


class LocalVisualBranch(nn.Module):
    """Local visual: CNN features + optical flow → GRU — UNCHANGED"""
    def __init__(self, cnn_dim=512, hidden_dim=256):
        super().__init__()
        self.content_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.content_fc = nn.Linear(cnn_dim, 128)
        self.content_norm = nn.LayerNorm(128)
        
        self.motion_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.motion_proj = nn.Linear(64, 128)
        self.motion_norm = nn.LayerNorm(128)
        
        self.content_gru = nn.GRU(128, 128, batch_first=True)
        self.motion_gru = nn.GRU(128, 128, batch_first=True)
        self.fuse_gru = nn.GRU(256, hidden_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, local_cnn, local_motion):
        B, T = local_cnn.shape[:2]
        
        if local_cnn.dim() == 5:
            cnn_flat = local_cnn.view(B * T, *local_cnn.shape[2:])
            cnn_pooled = self.content_pool(cnn_flat).view(B * T, -1)
            cnn_feat = self.content_fc(cnn_pooled).view(B, T, -1)
        else:
            cnn_feat = self.content_fc(local_cnn)
        cnn_feat = self.content_norm(cnn_feat)
        h_content, _ = self.content_gru(cnn_feat)
        
        if local_motion.dim() == 5:
            motion_flat = local_motion.view(B * T, *local_motion.shape[2:])
            motion_feat = self.motion_conv(motion_flat)
            motion_feat = motion_feat.view(B * T, -1)
            motion_feat = self.motion_proj(motion_feat).view(B, T, -1)
        else:
            if local_motion.shape[-1] != 128:
                motion_feat = F.adaptive_avg_pool1d(
                    local_motion.transpose(1, 2), 128
                ).transpose(1, 2)
            else:
                motion_feat = local_motion
        motion_feat = self.motion_norm(motion_feat)
        h_motion, _ = self.motion_gru(motion_feat)
        
        h_cat = torch.cat([h_content, h_motion], dim=-1)
        h_local, _ = self.fuse_gru(h_cat)
        h_local = self.out_norm(h_local)
        
        return h_local


class GlobalContextBranchGradual(nn.Module):
    """
    Global context with GRADUAL spatial reduction (like PIP-Net).
    
    PIP-Net: Conv3D → MaxPool3D → Conv3D → MaxPool3D → ... → Flatten → FC → GRU
    Ours:    Conv2D → stride-2 → stride-2 → pool(1×1) → FC → GRU
    
    Key: 8×8 → 4×4 → 2×2 → 1×1 (gradual, not one-shot)
    Each conv layer learns what spatial info to compress.
    
    Output: [B, T, 256] — temporal sequence for concat fusion
    """
    def __init__(self, sem_num_classes=20, sem_embed_dim=16, hidden_dim=256):
        super().__init__()
        
        self.sem_embed = nn.Embedding(sem_num_classes, sem_embed_dim)
        
        # 64×64 → 8×8 (same as all versions)
        self.sem_conv = nn.Sequential(
            nn.Conv2d(sem_embed_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        
        self.depth_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        
        # ★ GRADUAL reduction: 8×8 → 4×4 → 2×2 → 1×1 ★
        # Each strided conv LEARNS what to keep (unlike AvgPool which blindly averages)
        self.combine_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),  # 8→4
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # 4→2
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),                              # 2→1 (only 4 cells averaged)
        )
        
        # FC to project (like PIP-Net: Flatten → FC)
        self.fc = nn.Linear(256, hidden_dim)
        self.fc_norm = nn.LayerNorm(hidden_dim)
        
        # GRU for temporal (like PIP-Net)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, sem_labels, cat_depth):
        B, T = sem_labels.shape[:2]
        
        # Semantic
        sem_clamped = sem_labels.clamp(0, self.sem_embed.num_embeddings - 1)
        sem_emb = self.sem_embed(sem_clamped)
        sem_emb = sem_emb.view(B * T, *sem_emb.shape[2:])
        sem_emb = sem_emb.permute(0, 3, 1, 2).contiguous()
        sem_feat = self.sem_conv(sem_emb)  # [B*T, 256, 8, 8]
        
        # Depth
        if cat_depth.dtype != torch.float32:
            cat_depth = cat_depth.float()
        depth_flat = cat_depth.view(B * T, *cat_depth.shape[2:])
        depth_feat = self.depth_conv(depth_flat)  # [B*T, 256, 8, 8]
        
        # Combine + gradual reduce: 8→4→2→1
        combined = torch.cat([sem_feat, depth_feat], dim=1)  # [B*T, 512, 8, 8]
        combined = self.combine_conv(combined)  # [B*T, 256, 1, 1]
        combined = combined.view(B * T, -1)     # [B*T, 256]
        
        # FC (like PIP-Net's Flatten → FC)
        combined = self.fc(combined)
        combined = self.fc_norm(combined)
        combined = F.relu(combined)
        combined = combined.view(B, T, -1)  # [B, T, 256]
        
        # GRU temporal processing
        h_global, _ = self.gru(combined)  # [B, T, 256]
        h_global = self.out_norm(h_global)
        
        return h_global  # [B, T, 256]


# ============================================================
# MAIN MODEL — Option C: PIP-Net Faithful
# ============================================================

class PIPNetAlphaV3OptionC(nn.Module):
    """
    PIP-Net-faithful fusion: concat + joint attention.
    
    Pipeline (matches PIP-Net Fig. 3):
    
    1. Kinematic → GRU → Attention → z_kin          [B, 128]
    
    2. Local  → GRU → Attention → z_local            [B, 128]
    
    3. Global → Conv2D → gradual pool → FC → GRU → h_global  [B, T, 256]
    
    4. ★ PIP-NET FUSION ★
       expand z_local → [B, T, 128]
       concat(z_local, h_global) → [B, T, 384]
       Joint Attention(384 → 128) → z_visual          [B, 128]
       
       WHY: z_local tells the joint attention WHICH pedestrian we care about.
       The joint attention then finds the most relevant time steps in the
       fused local+global representation. Global features are weighted by
       their relevance to this specific pedestrian.
    
    5. Stack [z_visual, z_kin] → Final Attention → z_final  [B, 256]
    
    6. FC → logit
    """
    def __init__(
        self,
        bbox_dim: int = 4,
        pose_dim: int = 34,
        speed_dim: int = 1,
        branch_out_dim: int = 128,
        final_dim: int = 256,
        sem_num_classes: int = 20,
        sem_embed_dim: int = 16,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        
        self.branch_out_dim = branch_out_dim
        self.final_dim = final_dim
        
        # ============ KINEMATIC BRANCH (unchanged) ============
        self.kinematic_branch = KinematicBranch(
            bbox_dim=bbox_dim, pose_dim=pose_dim,
            speed_dim=speed_dim, hidden_dim=128
        )
        self.kin_attn = PIPNetAttention(128, branch_out_dim, dropout_p)
        self.norm_kin = nn.LayerNorm(branch_out_dim)
        
        # ============ LOCAL VISUAL BRANCH (unchanged) ============
        self.local_branch = LocalVisualBranch(cnn_dim=512, hidden_dim=256)
        self.local_attn = PIPNetAttention(256, branch_out_dim, dropout_p)
        self.norm_local = nn.LayerNorm(branch_out_dim)
        
        # ============ GLOBAL CONTEXT BRANCH (gradual pooling) ============
        self.global_branch = GlobalContextBranchGradual(
            sem_num_classes=sem_num_classes,
            sem_embed_dim=sem_embed_dim,
            hidden_dim=256,  # GRU output dim
        )
        
        # ============ ★ PIP-NET FUSION: Joint Attention ★ ============
        # Input: concat(z_local_expanded, h_global) = [B, T, 128+256] = [B, T, 384]
        # Output: z_visual [B, 128]
        self.visual_fuse_attn = PIPNetAttention(
            hidden_dim=branch_out_dim + 256,  # 128 + 256 = 384
            out_dim=branch_out_dim,            # 128
            dropout_p=dropout_p,
        )
        self.norm_visual = nn.LayerNorm(branch_out_dim)
        
        # ============ FINAL ATTENTION (unchanged) ============
        self.final_attn = PIPNetAttention(branch_out_dim, final_dim, dropout_p)
        
        # ============ AUXILIARY HEADS ============
        self.aux_kin = nn.Linear(branch_out_dim, 1)
        self.aux_local = nn.Linear(branch_out_dim, 1)
        self.aux_global = nn.Linear(256, 1)  # from GRU hidden, uses last step
        
        # ============ OUTPUT ============
        self.fc_out = nn.Linear(final_dim, 1)

    def forward(self, batch: dict, return_aux: bool = False):
        bbox = batch["bbox"]
        pose = batch["pose"]
        speed = batch["speed"]
        local_cnn = batch["local_cnn"]
        local_motion = batch["local_motion"]
        sem_labels = batch["sem_labels"]
        cat_depth = batch["cat_depth"]
        
        B = bbox.size(0)
        T = bbox.size(1)
        
        # ============ 1. KINEMATIC (unchanged) ============
        h_kin = self.kinematic_branch(bbox, pose, speed)   # [B, T, 128]
        z_kin, alpha_kin = self.kin_attn(h_kin)             # [B, 128]
        z_kin = self.norm_kin(z_kin)
        
        # ============ 2. LOCAL VISUAL (unchanged) ============
        h_local = self.local_branch(local_cnn, local_motion)  # [B, T, 256]
        z_local, alpha_local = self.local_attn(h_local)        # [B, 128]
        z_local = self.norm_local(z_local)
        
        # ============ 3. GLOBAL CONTEXT ============
        h_global = self.global_branch(sem_labels, cat_depth)  # [B, T, 256]
        
        # ============ 4. ★ PIP-NET FUSION ★ ============
        # Expand z_local across time steps:
        # z_local [B, 128] → [B, T, 128]
        # This tells joint attention: "here's what the pedestrian looks like,
        # now find the most relevant global context for THIS pedestrian"
        z_local_expanded = z_local.unsqueeze(1).expand(-1, T, -1)  # [B, T, 128]
        
        # Concat local + global at each time step
        h_fused = torch.cat([z_local_expanded, h_global], dim=-1)  # [B, T, 384]
        
        # Joint attention: picks the best time steps from the fused representation
        z_visual, alpha_visual = self.visual_fuse_attn(h_fused)  # [B, 128]
        z_visual = self.norm_visual(z_visual)
        
        # ============ 5. FINAL ATTENTION ============
        z_visual_kin = torch.stack([z_visual, z_kin], dim=1)  # [B, 2, 128]
        z_final, alpha_final = self.final_attn(z_visual_kin, use_mean_query=True)  # [B, 256]
        
        # ============ 6. OUTPUT ============
        logit = self.fc_out(z_final)
        
        if return_aux:
            # For aux_global, use last hidden state of global GRU
            z_global_last = h_global[:, -1, :]  # [B, 256]
            
            return {
                'logit': logit,
                'aux_kin': self.aux_kin(z_kin),
                'aux_local': self.aux_local(z_local),
                'aux_global': self.aux_global(z_global_last),
                'modality_weights': alpha_final,
                'visual_fuse_weights': alpha_visual,   # replaces cross_attn_weights
                'cross_attn_weights': alpha_visual,    # compatibility with training script
                'z_kin': z_kin,
                'z_local': z_local,
                'z_cross': z_visual,     # compatibility: z_cross → z_visual in this version
                'z_visual': z_visual,
            }
        
        return logit


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    model = PIPNetAlphaV3OptionC()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Option C — PIP-Net Faithful")
    print(f"Parameters: {n_params:,}")
    
    B, T = 2, 10
    batch = {
        "bbox": torch.randn(B, T, 4),
        "pose": torch.randn(B, T, 34),
        "speed": torch.randn(B, T, 1),
        "local_cnn": torch.randn(B, T, 512, 7, 7),
        "local_motion": torch.randn(B, T, 2, 224, 224),
        "sem_labels": torch.randint(0, 20, (B, T, 64, 64)),
        "cat_depth": torch.randn(B, T, 2, 64, 64),
    }
    
    out = model(batch, return_aux=True)
    print(f"\nOutputs:")
    print(f"  logit: {out['logit'].shape}")
    print(f"  modality_weights (visual vs kin): {out['modality_weights']}")
    print(f"  visual_fuse_weights: {out['visual_fuse_weights'].shape}")
    print(f"    → joint attention over {out['visual_fuse_weights'].shape[1]} time steps")
    print(f"    → each step has [local_128 | global_256] = 384-dim fused features")
    
    print(f"\nPipeline summary:")
    print(f"  1. z_kin:    {out['z_kin'].shape}  (kinematic attention)")
    print(f"  2. z_local:  {out['z_local'].shape}  (local visual attention)")
    print(f"  3. z_visual: {out['z_visual'].shape}  (joint local+global attention)")
    print(f"  4. logit:    {out['logit'].shape}  (final prediction)")
    print(f"\n✓ PIP-Net faithful: concat + joint attention, no cross-attention!")