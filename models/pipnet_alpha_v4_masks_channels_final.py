# models/pipnet_alpha_v4_masks_channels_final.py
"""
PIPNet-Alpha V4 — Mask-channel semantic variant

Base: Step-2 dropout model (0.853 native test AUC, 0.882 post-hoc fusion).

Fixes applied in V3:
  Fix 1 — Removed branch_logit_fusion (Step-3B). Back to clean attention fusion.
  Fix 2 — aux_global now uses a proper PIPNetAttention head (same as kin/local)
           instead of last-frame spatial mean. Gives global branch a fair
           temporal summary for gradient flow.
  Fix 3 — forward() returns both modality_weights AND visual_fuse_weights
           so MultiTaskLoss can apply entropy to both.
           See train script: entropy loss now = entropy(modality) + entropy(visual_fuse).

Global Branch Refactor (replaces combine_conv3d):
  OLD: combine_conv3d(512→256→256, stride=(1,2,2)) merged sem+depth spatially,
       reducing 8×8 → 4×4 (S=16 spatial tokens per frame), then a per-token GRU.
       Cost: ~3.5M params in combine_conv3d alone.

  NEW: Separate Flatten + FC per tower (paper-aligned).
       - Both Conv3D towers output (B, 256, T, 8×8). Spatial grid preserved.
       - Each tower is permuted to (B, T, 256, 8, 8) then flattened to (B, T, 16384).
         Flattening is lossless — every spatial coordinate maps to a unique index.
       - sem_fc and depth_fc each project 16384 → 256 with LayerNorm + ReLU.
         These are "judge panels": a dedicated weight for every (channel, H, W) position,
         so the network learns geographic importance (e.g. bottom-right = pedestrian zone).
       - Concat both 256-dim vectors → (B, T, 512).
       - Single temporal GRU(input=512, hidden=256) runs over T frames → (B, T, 256).
       No S dimension. One 256-dim vector per frame, not per spatial token.

  Downstream impact:
       - global_branch output is now (B, T, 256) instead of (B, T, S, 256).
       - visual_fuse_attn input: z_local(B,T,128) cat h_global(B,T,256) → (B,T,384).
         Same hidden_dim=384 as before — no change to visual_fuse_attn or global_attn.
       - visual_fuse_weights shape: (B, T=10) instead of (B, T*S=160).
         The entropy term in MultiTaskLoss still works — just over 10 tokens, not 160.

Unchanged from Step-2:
  - LocalVisualBranch: Dropout(0.3) between GRUs
  - GlobalContextBranchConv3D: Dropout3d(0.2) between Conv3D stages
  - Dropout(0.5) before fc_out
  - Conv3D tower channels: 64→128→256 with MaxPool3d(1,2,2) ×3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Attention module (unchanged)
# ---------------------------------------------------------------------------

class PIPNetAttention(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, dropout_p: float = 0.5):
        super().__init__()
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
        hm = h.mean(dim=1) if use_mean_query else h[:, -1, :]
        h_proj = self.Wp(h)
        scores = torch.bmm(h_proj, hm.unsqueeze(-1)).squeeze(-1)
        alpha = F.softmax(scores, dim=-1)
        hc = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        concat = torch.cat([hc, hm], dim=-1)
        out = torch.tanh(self.Wc(concat))
        out = self.dropout(out)
        return out, alpha


# ---------------------------------------------------------------------------
# Kinematic branch (unchanged)
# ---------------------------------------------------------------------------

class KinematicBranch(nn.Module):
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


# ---------------------------------------------------------------------------
# Local visual branch (unchanged)
# ---------------------------------------------------------------------------

class LocalVisualBranch(nn.Module):
    def __init__(self, cnn_dim=512, hidden_dim=256, dropout_p=0.3):
        super().__init__()
        
        # ── 1. Local Content Pipeline (Symmetric V3 State) ─────────────
        self.content_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Locked at 128 to perfectly mirror the motion branch
        self.content_fc = nn.Linear(cnn_dim, 128)
        self.content_norm = nn.LayerNorm(128)

        # ── 2. Local Motion Pipeline (Lightweight 3D Upgraded) ─────────
        self.motion_conv3d = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # 224 -> 112

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # 112 -> 56
            )
        
        self.motion_proj = nn.Linear(64, 128)
        self.motion_norm = nn.LayerNorm(128)

        # ── 3. Temporal GRUs (Perfectly Symmetrical) ───────────────────
        self.content_gru = nn.GRU(128, 128, batch_first=True)
        self.motion_gru = nn.GRU(128, 128, batch_first=True)
        self.fuse_gru = nn.GRU(256, hidden_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.content_drop = nn.Dropout(dropout_p)
        self.motion_drop = nn.Dropout(dropout_p)
        self.fuse_drop = nn.Dropout(dropout_p)

    def forward(self, local_cnn, local_motion):
        B, T = local_cnn.shape[:2]

        # ── Forward: Content ───────────────────────────────────────────
        if local_cnn.dim() == 5:
            # Safely using reshape to avoid contiguous memory crashes
            cnn_flat = local_cnn.reshape(B * T, *local_cnn.shape[2:])
            cnn_pooled = self.content_pool(cnn_flat).reshape(B * T, -1)
            cnn_feat = self.content_fc(cnn_pooled).reshape(B, T, -1)
        else:
            cnn_feat = self.content_fc(local_cnn)
            
        cnn_feat = self.content_norm(cnn_feat)
        h_content, _ = self.content_gru(cnn_feat)
        h_content = self.content_drop(h_content)

        # ── Forward: Motion ────────────────────────────────────────────
        if local_motion.dim() == 5:
            motion_5d = local_motion.permute(0, 2, 1, 3, 4).contiguous()
            m_feat = self.motion_conv3d(motion_5d)  # Output: [B, 64, T, H, W]
            
            # Average pooling gives a smoother spatial summary than max pooling for noisy flow features
            m_feat = m_feat.mean(dim=(-1, -2))      # Output: [B, 64, T]
            
            # Transpose to [B, T, 64] for Linear/GRU. contiguous() kept for CuDNN RNN safety.
            m_feat = m_feat.transpose(1, 2).contiguous()
            motion_feat = self.motion_proj(m_feat)
        else:
            # Fallback path 
            if local_motion.shape[-1] != 128:
                motion_feat = F.adaptive_avg_pool1d(
                    local_motion.transpose(1, 2), 128
                ).transpose(1, 2)
            else:
                motion_feat = local_motion
                
        motion_feat = self.motion_norm(motion_feat)
        h_motion, _ = self.motion_gru(motion_feat)
        h_motion = self.motion_drop(h_motion)

        # ── Forward: Fusion ────────────────────────────────────────────
        h_cat = torch.cat([h_content, h_motion], dim=-1)
        h_local, _ = self.fuse_gru(h_cat)
        h_local = self.fuse_drop(h_local)
        
        return self.out_norm(h_local)


# ---------------------------------------------------------------------------
# Downsampling helpers (UPDATED)
# ---------------------------------------------------------------------------

class SemanticDownsample(nn.Module):
    """
    V4 semantic input replacement: traffic-aware mask channels instead of
    20-class learned embeddings.

    Input:
      sem_labels: (B,T,H,W), integer PIP labels in [0,19]

    Converts semantic labels into 6 binary traffic-relevant channels:
      0 pedestrians: person + rider
      1 vehicles: car/truck/bus/train/motorcycle/bicycle
      2 traffic controls: traffic light + traffic sign
      3 road
      4 sidewalk
      5 pole

    Output is kept identical to the old SemanticDownsample interface:
      (B, out_channels, T, target_size, target_size)

    Notes:
      - num_classes and embed_dim are accepted only for compatibility with
        GlobalContextBranchConv3D, but no nn.Embedding is used.
      - out_channels defaults to the previous semantic embedding width, so the
        rest of V4 remains unchanged.
    """
    def __init__(self, num_classes=20, embed_dim=16, out_channels=16, target_size=64):
        super().__init__()
        self.target_size = int(target_size)
        self.out_channels = int(out_channels)

        pip_to_reduced = torch.tensor([
            0,  # 0 bg          -> bg/noise
            1,  # 1 person      -> person
            2,  # 2 rider       -> rider
            3,  # 3 car         -> vehicle
            3,  # 4 truck       -> vehicle
            3,  # 5 bus         -> vehicle
            3,  # 6 train       -> vehicle
            3,  # 7 motorcycle  -> vehicle
            3,  # 8 bicycle     -> vehicle
            4,  # 9 traffic_lt  -> traffic control
            4,  # 10 traffic_sn -> traffic control
            5,  # 11 road       -> road
            6,  # 12 sidewalk   -> sidewalk
            0,  # 13 building   -> bg/noise
            0,  # 14 wall       -> bg/noise
            0,  # 15 fence      -> bg/noise
            7,  # 16 pole       -> pole
            0,  # 17 vegetation -> bg/noise
            0,  # 18 terrain    -> bg/noise
            0,  # 19 sky        -> bg/noise
        ], dtype=torch.long)
        self.register_buffer("pip_to_reduced", pip_to_reduced, persistent=False)

        # 6 binary mask channels -> previous semantic channel width.
        # This replaces nn.Embedding(num_classes, embed_dim).
        self.mask_proj = nn.Sequential(
            nn.Conv3d(6, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, sem_labels):
        B, T, H, W = sem_labels.shape

        sem = sem_labels.long().clamp(0, 19)
        reduced = self.pip_to_reduced[sem]  # (B,T,H,W)

        masks = torch.stack([
            ((reduced == 1) | (reduced == 2)).float(),  # pedestrians
            (reduced == 3).float(),                     # vehicles
            (reduced == 4).float(),                     # traffic controls
            (reduced == 5).float(),                     # road
            (reduced == 6).float(),                     # sidewalk
            (reduced == 7).float(),                     # pole
        ], dim=2)  # (B,T,6,H,W)

        if H != self.target_size or W != self.target_size:
            x = masks.reshape(B * T, 6, H, W)
            # avg pooling keeps fractional occupancy per traffic mask channel.
            x = F.adaptive_avg_pool2d(x, (self.target_size, self.target_size))
            masks = x.reshape(B, T, 6, self.target_size, self.target_size)

        x = masks.permute(0, 2, 1, 3, 4).contiguous()  # (B,6,T,S,S)
        return self.mask_proj(x)


class DepthDownsample(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, target_size=64):
        super().__init__()
        self.target_size = target_size
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((target_size, target_size)), # CHANGED to MaxPool
        )

    def forward(self, cat_depth):
        B, T, C, H, W = cat_depth.shape
        if cat_depth.dtype != torch.float32:
            cat_depth = cat_depth.float()
        if H == self.target_size and W == self.target_size:
            return cat_depth.permute(0, 2, 1, 3, 4).contiguous()
        x = cat_depth.reshape(B * T, C, H, W)
        out = self.conv(x)
        out = out.reshape(B, T, -1, self.target_size, self.target_size)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        return out


# ---------------------------------------------------------------------------
# Global context branch — REFACTORED
#
# Dimension trace (target_size=64, T=10):
#   sem_down output  : (B, 16, T, 64, 64)   [channels=embed_dim]
#   depth_down output: (B,  2, T, 64, 64)
#   after sem_conv3d : (B, 256, T, 8, 8)    [3× MaxPool3d(1,2,2)]
#   after depth_conv3d:(B, 256, T, 8, 8)
#   permute+flatten  : (B, T, 256*8*8) = (B, T, 16384)  each
#   sem_fc output    : (B, T, 256)            [16384 → 256]
#   depth_fc output  : (B, T, 256)            [16384 → 256]
#   concat           : (B, T, 512)
#   temporal_gru out : (B, T, 256)            [GRU input=512, hidden=256]
#   out_norm output  : (B, T, 256)            ← returned to main model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global context branch — THIN TOWERS REFACTOR
#
# Dimension trace (target_size=64, T=10):
#   sem_down output  : (B, 16, T, 64, 64)   [channels=embed_dim]
#   depth_down output: (B,  2, T, 64, 64)
#   after sem_conv3d : (B, 64, T, 8, 8)     [3× MaxPool3d(1,2,2)]
#   after depth_conv3d:(B, 64, T, 8, 8)
#   permute+flatten  : (B, T, 64*8*8=4096)  each
#   sem_fc output    : (B, T, 256)          [4096 → 256]
#   depth_fc output  : (B, T, 256)          [4096 → 256]
#   concat           : (B, T, 512)
#   temporal_gru out : (B, T, 256)          [GRU input=512, hidden=256]
# ---------------------------------------------------------------------------

class GlobalContextBranchConv3D(nn.Module):
    """
    Conv3D towers with a lightweight channel progression (16→32→64) + MaxPool3d ×3.
    Reduces flat_dim to 4,096 naturally without requiring a bottleneck.
    Separate Flatten+FC per tower preserves spatial geography deterministically.
    """

    # Spatial grid side after 3× MaxPool3d(1,2,2) on a target_size=64 input.
    # 64 → 32 → 16 → 8.  This is a constant given the architecture.
    _SPATIAL_SIDE = 8

    def __init__(
        self,
        sem_num_classes: int = 20,
        sem_embed_dim: int = 16,
        hidden_dim: int = 256,
        target_size: int = 64,
        dropout_p: float = 0.2,
        judge_dropout_p: float = 0.4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_size = target_size

        # ── input downsampling ─────────────────────────────────────────────
        self.sem_down = SemanticDownsample(
            num_classes=sem_num_classes,
            embed_dim=sem_embed_dim,
            out_channels=sem_embed_dim,
            target_size=target_size,
        )
        self.depth_down = DepthDownsample(
            in_channels=2,
            out_channels=2,
            target_size=target_size,
        )

        # ── Conv3D tower: semantic  (16 → 16 → 32 → 64) ─────────────────
        self.sem_conv3d = nn.Sequential(
            nn.Conv3d(sem_embed_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),
        )

        # ── Conv3D tower: depth  (2 → 16 → 32 → 64) ─────────────────────
        self.depth_conv3d = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),
        )

        # ── Separate FC projections (replaces combine_conv3d) ──────────────
        # After the towers: (B, 64, T, 8, 8).
        # Permute to (B, T, 64, 8, 8) then flatten last 3 dims → (B, T, 4096).
        flat_dim = 64 * self._SPATIAL_SIDE * self._SPATIAL_SIDE  # 4096

        self.sem_fc = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(judge_dropout_p),
        )
        self.depth_fc = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(judge_dropout_p),
        )

        # ── Temporal GRU + output norm ─────────────────────────────────────
        # Input: concat(sem_fc, depth_fc) = (B, T, 512)
        # Output: (B, T, hidden_dim=256)
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim * 2,   # 512
            hidden_size=hidden_dim,       # 256
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sem_labels: torch.Tensor, cat_depth: torch.Tensor) -> torch.Tensor:
        B, T = sem_labels.shape[:2]

        # ── 1. Downsample inputs to target_size × target_size ──────────────
        sem_emb  = self.sem_down(sem_labels)
        depth_5d = self.depth_down(cat_depth)

        # ── 2. Conv3D feature extraction ────────────────────────────────────
        sem_feat   = self.sem_conv3d(sem_emb)    # (B, 64, T, 8, 8)
        depth_feat = self.depth_conv3d(depth_5d) # (B, 64, T, 8, 8)

        # ── 3. Permute + flatten spatial dims ───────────────────────────────
        sem_flat   = sem_feat.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
        depth_flat = depth_feat.permute(0, 2, 1, 3, 4).reshape(B, T, -1)

        # ── 4. Separate FC projections → (B, T, 256) each ──────────────────
        z_sem   = self.sem_fc(sem_flat)
        z_depth = self.depth_fc(depth_flat)

        # ── 5. Concat → (B, T, 512) ─────────────────────────────────────────
        combined = torch.cat([z_sem, z_depth], dim=-1)

        # ── 6. Temporal GRU → (B, T, 256) ───────────────────────────────────
        h_global, _ = self.temporal_gru(combined)

        return self.out_norm(h_global)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PIPNetAlphaV4Final(nn.Module):
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
        global_target_size: int = 64,
        local_dropout_p: float = 0.3,
        global_dropout_p: float = 0.2,
    ):
        super().__init__()
        self.branch_out_dim = branch_out_dim
        self.final_dim = final_dim

        # ── Kinematic branch ────────────────────────────────────────────────
        self.kinematic_branch = KinematicBranch(
            bbox_dim=bbox_dim, pose_dim=pose_dim,
            speed_dim=speed_dim, hidden_dim=128,
        )
        self.kin_attn  = PIPNetAttention(128, branch_out_dim, dropout_p)
        self.norm_kin  = nn.LayerNorm(branch_out_dim)

        # ── Local visual branch ─────────────────────────────────────────────
        self.local_branch = LocalVisualBranch(
            cnn_dim=512, hidden_dim=256, dropout_p=local_dropout_p,
        )
        self.local_attn  = PIPNetAttention(256, branch_out_dim, dropout_p)
        self.norm_local  = nn.LayerNorm(branch_out_dim)

        # ── Global context branch ───────────────────────────────────────────
        # Returns (B, T, 256).  One 256-dim vector per frame (no spatial tokens).
        self.global_branch = GlobalContextBranchConv3D(
            sem_num_classes=sem_num_classes,
            sem_embed_dim=sem_embed_dim,
            hidden_dim=256,
            target_size=global_target_size,
            dropout_p=global_dropout_p,
        )

        # ── Visual fusion: local (B,T,128) cat global (B,T,256) → (B,T,384) ─
        # PIPNetAttention(hidden_dim=384) attends over T=10 frames.
        # visual_fuse_weights shape: (B, T=10).
        self.visual_fuse_attn = PIPNetAttention(
            hidden_dim=branch_out_dim + 256,  # 128 + 256 = 384
            out_dim=branch_out_dim,            # → 128
            dropout_p=dropout_p,
        )
        self.norm_visual = nn.LayerNorm(branch_out_dim)

        # ── Final attention: stack [z_visual, z_kin] → (B, 2, 128) ─────────
        self.final_attn = PIPNetAttention(branch_out_dim, final_dim, dropout_p)

        # ── Auxiliary heads ─────────────────────────────────────────────────
        self.aux_kin   = nn.Linear(branch_out_dim, 1)
        self.aux_local = nn.Linear(branch_out_dim, 1)

        # global aux: attend over T global frames → compress to branch_out_dim
        self.global_attn  = PIPNetAttention(256, branch_out_dim, dropout_p)
        self.norm_global  = nn.LayerNorm(branch_out_dim)
        self.aux_global   = nn.Linear(branch_out_dim, 1)

        # ── Output head ─────────────────────────────────────────────────────
        self.fc_drop = nn.Dropout(dropout_p)
        self.fc_out  = nn.Linear(final_dim, 1)

    def forward(self, batch: dict, return_aux: bool = False):
        bbox         = batch["bbox"]
        pose         = batch["pose"]
        speed        = batch["speed"]
        local_cnn    = batch["local_cnn"]
        local_motion = batch["local_motion"]
        sem_labels   = batch["sem_labels"]
        cat_depth    = batch["cat_depth"]

        B = bbox.size(0)
        T = bbox.size(1)

        # ── Kinematic branch → z_kin (B, 128) ───────────────────────────────
        h_kin = self.kinematic_branch(bbox, pose, speed)
        z_kin, _ = self.kin_attn(h_kin)
        z_kin = self.norm_kin(z_kin)

        # ── Local visual branch → z_local (B, 128) ──────────────────────────
        h_local = self.local_branch(local_cnn, local_motion)
        z_local, _ = self.local_attn(h_local)
        z_local = self.norm_local(z_local)

        # ── Global context branch → h_global (B, T, 256) ────────────────────
        h_global = self.global_branch(sem_labels, cat_depth)  # (B, T, 256)

        # ── Visual fusion ────────────────────────────────────────────────────
        # Expand z_local to match T frames: (B, 128) → (B, T, 128)
        z_local_expanded = z_local.unsqueeze(1).expand(-1, T, -1)  # (B, T, 128)

        if getattr(self, '_ablate_global', False):
            # Ablation: zero global tokens while keeping architecture identical.
            h_global_for_fusion = torch.zeros_like(h_global)
        else:
            h_global_for_fusion = h_global

        # concat along feature dim: (B, T, 128+256) = (B, T, 384)
        h_fused = torch.cat([z_local_expanded, h_global_for_fusion], dim=-1)

        # PIPNetAttention over T=10 sequence → z_visual (B, 128)
        # visual_fuse_weights: (B, T=10)
        z_visual, alpha_visual = self.visual_fuse_attn(h_fused)
        z_visual = self.norm_visual(z_visual)

        # ── Final fusion: [z_visual, z_kin] → logit ─────────────────────────
        z_visual_kin = torch.stack([z_visual, z_kin], dim=1)  # (B, 2, 128)
        z_final, alpha_final = self.final_attn(z_visual_kin, use_mean_query=True)

        z_final = self.fc_drop(z_final)
        logit   = self.fc_out(z_final)  # (B, 1)

        if return_aux:
            # global aux: attend over T global frames
            z_global_att, _ = self.global_attn(h_global, use_mean_query=True)
            z_global_att = self.norm_global(z_global_att)

            return {
                'logit':               logit,
                'aux_kin':             self.aux_kin(z_kin),
                'aux_local':           self.aux_local(z_local),
                'aux_global':          self.aux_global(z_global_att),
                'modality_weights':    alpha_final,    # (B, 2)
                'visual_fuse_weights': alpha_visual,   # (B, T=10)
                'cross_attn_weights':  alpha_visual,
                'z_kin':    z_kin,
                'z_local':  z_local,
                'z_cross':  z_visual,
                'z_visual': z_visual,
            }

        return logit


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = PIPNetAlphaV4Final()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n:,}")

    print()

    B, T = 2, 10
    batch = {
        "bbox":         torch.randn(B, T, 4),
        "pose":         torch.randn(B, T, 34),
        "speed":        torch.randn(B, T, 1),
        "local_cnn":    torch.randn(B, T, 512, 7, 7),
        "local_motion": torch.randn(B, T, 2, 224, 224),
        "sem_labels":   torch.randint(0, 20, (B, T, 384, 672)),
        "cat_depth":    torch.randn(B, T, 2, 384, 672),
    }

    out = model(batch, return_aux=True)

    print(f"logit shape:               {out['logit'].shape}         expect (B=2, 1)")
    print(f"aux_kin shape:             {out['aux_kin'].shape}         expect (B=2, 1)")
    print(f"aux_global shape:          {out['aux_global'].shape}         expect (B=2, 1)")
    print(f"modality_weights shape:    {out['modality_weights'].shape}   expect (B=2, 2)")
    print(f"visual_fuse_weights shape: {out['visual_fuse_weights'].shape} expect (B=2, T=10)")

    assert out['logit'].shape              == (B, 1),    "logit shape mismatch"
    assert out['aux_kin'].shape            == (B, 1),    "aux_kin shape mismatch"
    assert out['aux_global'].shape         == (B, 1),    "aux_global shape mismatch"
    assert out['modality_weights'].shape   == (B, 2),    "modality_weights shape mismatch"
    assert out['visual_fuse_weights'].shape == (B, T),   "visual_fuse_weights shape mismatch"
    print("\nAll shape assertions passed.")