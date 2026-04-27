# models/pipnet_alpha_v3_option_c_spatial.py
"""
PIPNet-Alpha V3 — Option C with Spatial Preservation

Same PIP-Net fusion philosophy (concat + joint attention),
but keeping 4×4 spatial grid instead of collapsing to 1×1.

Pipeline:
  1. Kinematic → hierarchical GRU → Attention → z_kin          [B, 128]
  2. Local  → content + motion GRU → fuse GRU → Attention → z_local  [B, 128]
  3. Global → Conv2D → 8×8 → stride-2 → 4×4 → GRU per slot → h_global [B, T, 16, 256]
  4. ★ SPATIAL PIP-Net Fusion ★
     z_local [B, 128] → expand to [B, T*16, 128]
     h_global [B, T, 16, 256] → reshape to [B, T*16, 256]
     concat → [B, T*16, 384]
     Joint Attention → z_visual [B, 128]
  5. Stack [z_visual, z_kin] → Final Attention → z_final [B, 256]
  6. FC → logit

vs Cross-Attention (Options A/B):
  Cross-attn: z_local is Query, h_global is Key/Value (different projections)
  This:       z_local is CONCATENATED with h_global (same attention space)
  Both give spatial awareness. This stays closer to PIP-Net's design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# ATTENTION
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
    """SFRNN-style hierarchical: pose → (+bbox) → (+speed)"""
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
    """Local visual: CNN features + optical flow → GRU"""
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


class GlobalContextBranchSpatialGRU(nn.Module):
    """
    Global context: 4×4 spatial preservation + per-position GRU.
    
    Pipeline:
    1. sem_conv + depth_conv: 64×64 → 8×8
    2. combine_conv: 8×8 → 4×4 (stride-2, preserves spatial)
    3. GRU per spatial position (shared weights): temporal processing
    4. Output: [B, T, 16, 256] — spatial-temporal tokens
    
    Like PIP-Net: uses GRU for temporal modeling.
    Unlike original: keeps 4×4 instead of pooling to 1×1.
    """
    def __init__(self, sem_num_classes=20, sem_embed_dim=16, hidden_dim=256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.sem_embed = nn.Embedding(sem_num_classes, sem_embed_dim)
        
        # 64×64 → 8×8
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
        
        # 8×8 → 4×4 (keeps spatial!)
        self.combine_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )
        
        # Shared GRU across 16 spatial positions
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, sem_labels, cat_depth):
        B, T = sem_labels.shape[:2]
        
        # Semantic
        sem_clamped = sem_labels.clamp(0, self.sem_embed.num_embeddings - 1)
        sem_emb = self.sem_embed(sem_clamped)
        sem_emb = sem_emb.view(B * T, *sem_emb.shape[2:])
        sem_emb = sem_emb.permute(0, 3, 1, 2).contiguous()
        sem_feat = self.sem_conv(sem_emb)
        
        # Depth
        if cat_depth.dtype != torch.float32:
            cat_depth = cat_depth.float()
        depth_flat = cat_depth.view(B * T, *cat_depth.shape[2:])
        depth_feat = self.depth_conv(depth_flat)
        
        # Combine: 8×8 → 4×4
        combined = torch.cat([sem_feat, depth_feat], dim=1)
        combined = self.combine_conv(combined)  # [B*T, 256, 4, 4]
        
        # Flatten spatial: [B*T, 256, 4, 4] → [B*T, 16, 256]
        S = combined.shape[2] * combined.shape[3]  # 16
        combined = combined.flatten(2).transpose(1, 2)  # [B*T, 16, 256]
        
        # Reshape for GRU: [B, T, 16, 256]
        combined = combined.reshape(B, T, S, self.hidden_dim)
        
        # Shared GRU per spatial position: [B*16, T, 256]
        combined = combined.permute(0, 2, 1, 3).reshape(B * S, T, self.hidden_dim)
        combined, _ = self.temporal_gru(combined)
        
        # Back to [B, T, 16, 256]
        combined = combined.reshape(B, S, T, self.hidden_dim)
        combined = combined.permute(0, 2, 1, 3)  # [B, T, 16, 256]
        combined = self.out_norm(combined)
        
        return combined  # [B, T, 16, 256]


# ============================================================
# MAIN MODEL
# ============================================================

class PIPNetAlphaV3OptionCSpatial(nn.Module):
    """
    PIP-Net faithful fusion WITH spatial preservation.
    
    Pipeline:
    1. Kinematic → GRU → Attention → z_kin                    [B, 128]
    2. Local  → GRU → Attention → z_local                      [B, 128]
    3. Global → Conv2D → 4×4 → GRU per slot → h_global        [B, T, 16, 256]
    
    4. ★ SPATIAL CONCAT FUSION ★
       z_local [B, 128] → expand to every (time, position)    [B, T*16, 128]
       h_global → reshape                                      [B, T*16, 256]
       concat → h_fused                                        [B, T*16, 384]
       Joint Attention (384 → 128) → z_visual                  [B, 128]
       
       The attention sees 160 spatial-temporal tokens.
       Each token is [pedestrian_identity | scene_at_this_location_and_time].
       Attention picks: "crosswalk 2m ahead in frame 7 is most relevant for
       this pedestrian who is standing at the curb."
    
    5. Stack [z_visual, z_kin] → Final Attention → z_final     [B, 256]
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
        
        # ============ KINEMATIC BRANCH ============
        self.kinematic_branch = KinematicBranch(
            bbox_dim=bbox_dim, pose_dim=pose_dim,
            speed_dim=speed_dim, hidden_dim=128
        )
        self.kin_attn = PIPNetAttention(128, branch_out_dim, dropout_p)
        self.norm_kin = nn.LayerNorm(branch_out_dim)
        
        # ============ LOCAL VISUAL BRANCH ============
        self.local_branch = LocalVisualBranch(cnn_dim=512, hidden_dim=256)
        self.local_attn = PIPNetAttention(256, branch_out_dim, dropout_p)
        self.norm_local = nn.LayerNorm(branch_out_dim)
        
        # ============ GLOBAL CONTEXT BRANCH (spatial + GRU) ============
        self.global_branch = GlobalContextBranchSpatialGRU(
            sem_num_classes=sem_num_classes,
            sem_embed_dim=sem_embed_dim,
            hidden_dim=256,
        )
        
        # ============ ★ SPATIAL CONCAT JOINT ATTENTION ★ ============
        # Input: [B, T*16, 128+256] = [B, 160, 384]
        # Attention over 160 spatial-temporal tokens
        self.visual_fuse_attn = PIPNetAttention(
            hidden_dim=branch_out_dim + 256,  # 384
            out_dim=branch_out_dim,            # 128
            dropout_p=dropout_p,
        )
        self.norm_visual = nn.LayerNorm(branch_out_dim)
        
        # ============ FINAL ATTENTION ============
        self.final_attn = PIPNetAttention(branch_out_dim, final_dim, dropout_p)
        
        # ============ AUXILIARY HEADS ============
        self.aux_kin = nn.Linear(branch_out_dim, 1)
        self.aux_local = nn.Linear(branch_out_dim, 1)
        self.aux_global = nn.Linear(256, 1)
        
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
        
        # ============ 1. KINEMATIC ============
        h_kin = self.kinematic_branch(bbox, pose, speed)
        z_kin, alpha_kin = self.kin_attn(h_kin)
        z_kin = self.norm_kin(z_kin)
        
        # ============ 2. LOCAL VISUAL ============
        h_local = self.local_branch(local_cnn, local_motion)
        z_local, alpha_local = self.local_attn(h_local)
        z_local = self.norm_local(z_local)
        
        # ============ 3. GLOBAL CONTEXT (spatial) ============
        h_global = self.global_branch(sem_labels, cat_depth)  # [B, T, 16, 256]
        S = h_global.shape[2]  # 16
        
        # ============ 4. ★ SPATIAL CONCAT FUSION ★ ============
        
        # Reshape global: [B, T, 16, 256] → [B, T*16, 256]
        h_global_flat = h_global.reshape(B, T * S, 256)
        
        # Expand z_local to every spatial-temporal position:
        # [B, 128] → [B, T*16, 128]
        z_local_expanded = z_local.unsqueeze(1).expand(-1, T * S, -1)
        
        # Concat: "who is this pedestrian" + "what's at this location/time"
        # [B, T*16, 128+256] = [B, 160, 384]
        h_fused = torch.cat([z_local_expanded, h_global_flat], dim=-1)
        
        # Joint Attention: picks best spatial-temporal tokens
        z_visual, alpha_visual = self.visual_fuse_attn(h_fused)  # [B, 128]
        z_visual = self.norm_visual(z_visual)
        
        # ============ 5. FINAL ATTENTION ============
        z_visual_kin = torch.stack([z_visual, z_kin], dim=1)
        z_final, alpha_final = self.final_attn(z_visual_kin, use_mean_query=True)
        
        # ============ 6. OUTPUT ============
        logit = self.fc_out(z_final)
        
        if return_aux:
            # Aux global: mean-pool over spatial, take last time step
            z_global_last = h_global[:, -1, :, :].mean(dim=1)  # [B, 256]
            
            return {
                'logit': logit,
                'aux_kin': self.aux_kin(z_kin),
                'aux_local': self.aux_local(z_local),
                'aux_global': self.aux_global(z_global_last),
                'modality_weights': alpha_final,
                'visual_fuse_weights': alpha_visual,
                'cross_attn_weights': alpha_visual,  # compatibility
                'z_kin': z_kin,
                'z_local': z_local,
                'z_cross': z_visual,    # compatibility
                'z_visual': z_visual,
            }
        
        return logit


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    model = PIPNetAlphaV3OptionCSpatial()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Option C Spatial — PIP-Net Faithful + Location Aware")
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
    print(f"  modality_weights: {out['modality_weights']}")
    print(f"  visual_fuse_weights: {out['visual_fuse_weights'].shape}")
    print(f"    → joint attention over {out['visual_fuse_weights'].shape[1]} spatial-temporal tokens")
    
    print(f"\nPipeline:")
    print(f"  z_kin:    {out['z_kin'].shape}")
    print(f"  z_local:  {out['z_local'].shape}")
    print(f"  z_visual: {out['z_visual'].shape}")
    print(f"  logit:    {out['logit'].shape}")
    
    print(f"\n  Global branch: [B, T, 16, 256] = {T}×16 = {T*16} spatial-temporal tokens")
    print(f"  Joint attention sees: [B, {T*16}, 384] = [pedestrian_128 | scene_256]")
    print(f"\n✓ PIP-Net faithful + spatial: concat fusion with location awareness!")