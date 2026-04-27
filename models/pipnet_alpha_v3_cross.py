# models/pipnet_alpha_v3_cross.py
"""
PIPNet-Alpha V3 Cross-Attention - Following Paper Figure 3 More Closely

Architecture:
1. Local  → GRU → Attention → z_local
2. Global → GRU → h_global (NO separate attention!)
3. Cross-Attention: z_local queries h_global → z_cross
4. Stack [z_local, z_cross] → Visual Attention → z_visual
5. Stack [z_visual, z_kin] → Final Attention → z_final
6. FC → logit

Key insight: Global doesn't have its own attention block.
Instead, Local's attention output guides what to extract from Global.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# ATTENTION MODULES
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
        """
        Args:
            h: [B, T, D] - sequence
        Returns:
            out: [B, out_dim]
            alpha: [B, T]
        """
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


class CrossAttention(nn.Module):
    """
    Cross-Attention: Query from one modality attends over another.
    
    Used for: z_local (query) attending over h_global (keys/values)
    "Given what's important in local visual, what scene context matters?"
    """
    def __init__(self, query_dim: int, key_dim: int, out_dim: int, dropout_p: float = 0.5):
        super().__init__()
        self.scale = query_dim ** -0.5
        
        # Project query, key, value
        self.W_q = nn.Linear(query_dim, query_dim)
        self.W_k = nn.Linear(key_dim, query_dim)  # Project key to query dim for dot product
        self.W_v = nn.Linear(key_dim, query_dim)
        
        # Output projection
        self.W_o = nn.Linear(query_dim, out_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, query: torch.Tensor, context: torch.Tensor):
        """
        Args:
            query: [B, D_q] - e.g., z_local (single vector)
            context: [B, T, D_k] - e.g., h_global (sequence)
        Returns:
            out: [B, out_dim]
            alpha: [B, T] - attention weights over context
        """
        B, T, _ = context.shape
        
        # Project
        Q = self.W_q(query)  # [B, D_q]
        K = self.W_k(context)  # [B, T, D_q]
        V = self.W_v(context)  # [B, T, D_q]
        
        # Attention scores
        # Q: [B, D_q] -> [B, 1, D_q]
        # K: [B, T, D_q] -> [B, D_q, T]
        scores = torch.bmm(Q.unsqueeze(1), K.transpose(-1, -2)).squeeze(1)  # [B, T]
        scores = scores * self.scale
        
        alpha = F.softmax(scores, dim=-1)  # [B, T]
        alpha = self.dropout(alpha)
        
        # Weighted sum of values
        # alpha: [B, 1, T] @ V: [B, T, D_q] -> [B, 1, D_q] -> [B, D_q]
        attended = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)  # [B, D_q]
        
        # Output projection
        out = self.W_o(attended)  # [B, out_dim]
        out = self.norm(out)
        
        return out, alpha


# ============================================================
# BRANCH MODULES
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
        
        # Content
        if local_cnn.dim() == 5:
            cnn_flat = local_cnn.view(B * T, *local_cnn.shape[2:])
            cnn_pooled = self.content_pool(cnn_flat).view(B * T, -1)
            cnn_feat = self.content_fc(cnn_pooled).view(B, T, -1)
        else:
            cnn_feat = self.content_fc(local_cnn)
        cnn_feat = self.content_norm(cnn_feat)
        h_content, _ = self.content_gru(cnn_feat)
        
        # Motion
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
        
        # Fuse
        h_cat = torch.cat([h_content, h_motion], dim=-1)
        h_local, _ = self.fuse_gru(h_cat)
        h_local = self.out_norm(h_local)
        
        return h_local


class GlobalContextBranch(nn.Module):
    """
    Global context: semantic + depth → GRU
    NOTE: NO attention here! Cross-attention will be applied later.
    """
    def __init__(self, sem_num_classes=20, sem_embed_dim=16, hidden_dim=256):
        super().__init__()
        
        # Semantic
        self.sem_embed = nn.Embedding(sem_num_classes, sem_embed_dim)
        self.sem_conv = nn.Sequential(
            nn.Conv2d(sem_embed_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.sem_fc = nn.Linear(128, 256)
        self.sem_norm = nn.LayerNorm(256)
        
        # Depth
        self.depth_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.depth_fc = nn.Linear(128, 256)
        self.depth_norm = nn.LayerNorm(256)
        
        # Temporal
        self.fuse_gru = nn.GRU(512, hidden_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sem_labels, cat_depth):
        """
        Returns: h_global [B, T, hidden_dim] - FULL SEQUENCE (no attention yet!)
        """
        B, T = sem_labels.shape[:2]
        
        # Semantic
        sem_clamped = sem_labels.clamp(0, self.sem_embed.num_embeddings - 1)
        sem_emb = self.sem_embed(sem_clamped)
        sem_emb = sem_emb.view(B * T, *sem_emb.shape[2:])
        sem_emb = sem_emb.permute(0, 3, 1, 2).contiguous()
        sem_feat = self.sem_conv(sem_emb)
        sem_feat = sem_feat.view(B * T, -1)
        sem_feat = self.sem_fc(sem_feat).view(B, T, -1)
        sem_feat = self.sem_norm(sem_feat)
        
        # Depth
        if cat_depth.dtype != torch.float32:
            cat_depth = cat_depth.float()
        depth_flat = cat_depth.view(B * T, *cat_depth.shape[2:])
        depth_feat = self.depth_conv(depth_flat)
        depth_feat = depth_feat.view(B * T, -1)
        depth_feat = self.depth_fc(depth_feat).view(B, T, -1)
        depth_feat = self.depth_norm(depth_feat)
        
        # Fuse
        h_cat = torch.cat([sem_feat, depth_feat], dim=-1)
        h_global, _ = self.fuse_gru(h_cat)
        h_global = self.out_norm(h_global)
        
        return h_global  # [B, T, hidden_dim] - Keep full sequence!


# ============================================================
# MAIN MODEL
# ============================================================

class PIPNetAlphaV3Cross(nn.Module):
    """
    PIPNet-Alpha V3 with Cross-Attention (Following Paper Figure 3)
    
    Architecture:
    1. Kinematic → GRU → Attention → z_kin [B, 128]
    
    2. Local  → GRU → h_local [B, T, 256]
              → Attention → z_local [B, 128]
              
    3. Global → GRU → h_global [B, T, 256]  (NO attention!)
    
    4. Cross-Attention: z_local queries h_global → z_cross [B, 128]
       "Given what's important in local, what scene context matters?"
    
    5. Stack [z_local, z_cross] → Visual Attention → z_visual [B, 128]
    
    6. Stack [z_visual, z_kin] → Final Attention → z_final [B, 256]
    
    7. FC → logit
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
        
        # ============ GLOBAL CONTEXT BRANCH ============
        # NOTE: No separate attention! Cross-attention will extract relevant info
        self.global_branch = GlobalContextBranch(
            sem_num_classes=sem_num_classes, 
            sem_embed_dim=sem_embed_dim, 
            hidden_dim=256
        )
        
        # ============ CROSS-ATTENTION: Local queries Global ============
        self.cross_attn = CrossAttention(
            query_dim=branch_out_dim,  # z_local dim
            key_dim=256,               # h_global dim
            out_dim=branch_out_dim,    # output same as branch
            dropout_p=dropout_p
        )
        
        # ============ VISUAL ATTENTION: Combines z_local + z_cross ============
        # Stack [z_local, z_cross] as [B, 2, 128] → Attention → z_visual
        self.visual_attn = PIPNetAttention(branch_out_dim, branch_out_dim, dropout_p)
        self.norm_visual = nn.LayerNorm(branch_out_dim)
        
        # ============ FINAL ATTENTION: Combines z_visual + z_kin ============
        # Stack [z_visual, z_kin] as [B, 2, 128] → Attention → z_final
        self.final_attn = PIPNetAttention(branch_out_dim, final_dim, dropout_p)
        
        # ============ AUXILIARY HEADS ============
        self.aux_kin = nn.Linear(branch_out_dim, 1)
        self.aux_local = nn.Linear(branch_out_dim, 1)
        self.aux_global = nn.Linear(branch_out_dim, 1)  # Uses z_cross
        
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
        
        # ============ 1. KINEMATIC ============
        h_kin = self.kinematic_branch(bbox, pose, speed)  # [B, T, 128]
        z_kin, alpha_kin = self.kin_attn(h_kin)  # [B, 128]
        z_kin = self.norm_kin(z_kin)
        
        # ============ 2. LOCAL VISUAL ============
        h_local = self.local_branch(local_cnn, local_motion)  # [B, T, 256]
        z_local, alpha_local = self.local_attn(h_local)  # [B, 128]
        z_local = self.norm_local(z_local)
        
        # ============ 3. GLOBAL CONTEXT (no attention!) ============
        h_global = self.global_branch(sem_labels, cat_depth)  # [B, T, 256]
        # h_global keeps full temporal sequence!
        
        # ============ 4. CROSS-ATTENTION ============
        # z_local asks: "Given what's important locally, what scene context matters?"
        z_cross, alpha_cross = self.cross_attn(z_local, h_global)  # [B, 128], [B, T]
        
        # ============ 5. VISUAL ATTENTION ============
        # Stack [z_local, z_cross] and apply attention
        z_local_cross = torch.stack([z_local, z_cross], dim=1)  # [B, 2, 128]
        z_visual, alpha_visual = self.visual_attn(z_local_cross, use_mean_query=True)  # [B, 128]
        z_visual = self.norm_visual(z_visual)
        
        # ============ 6. FINAL ATTENTION ============
        # Stack [z_visual, z_kin] and apply attention
        z_visual_kin = torch.stack([z_visual, z_kin], dim=1)  # [B, 2, 128]
        z_final, alpha_final = self.final_attn(z_visual_kin, use_mean_query=True)  # [B, 256]
        
        # ============ 7. OUTPUT ============
        logit = self.fc_out(z_final)
        
        if return_aux:
            return {
                'logit': logit,
                # Auxiliary predictions
                'aux_kin': self.aux_kin(z_kin),
                'aux_local': self.aux_local(z_local),
                'aux_global': self.aux_global(z_cross),  # Use cross-attention output
                # Attention weights for visualization
                'modality_weights': alpha_final,  # [B, 2] - visual vs kin
                'visual_weights': alpha_visual,   # [B, 2] - local vs cross
                'cross_attn_weights': alpha_cross,  # [B, T] - which timesteps in global
                # Branch features
                'z_kin': z_kin,
                'z_local': z_local,
                'z_cross': z_cross,
                'z_visual': z_visual,
            }
        
        return logit


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    model = PIPNetAlphaV3Cross()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Test forward
    B, T = 2, 15
    batch = {
        "bbox": torch.randn(B, T, 4),
        "pose": torch.randn(B, T, 34),
        "speed": torch.randn(B, T, 1),
        "local_cnn": torch.randn(B, T, 512),
        "local_motion": torch.randn(B, T, 512),
        "sem_labels": torch.randint(0, 20, (B, T, 64, 64)),
        "cat_depth": torch.randn(B, T, 2, 64, 64),
    }
    
    out = model(batch, return_aux=True)
    print(f"\nOutputs:")
    print(f"  logit: {out['logit'].shape}")
    print(f"  modality_weights (visual vs kin): {out['modality_weights']}")
    print(f"  visual_weights (local vs cross): {out['visual_weights']}")
    print(f"  cross_attn_weights: {out['cross_attn_weights'].shape} (attention over T={T} global timesteps)")
    
    print(f"\nThis shows:")
    print(f"  1. How much model weighs visual vs kinematic")
    print(f"  2. Within visual: how much local vs scene context")
    print(f"  3. Which global timesteps are attended to")
