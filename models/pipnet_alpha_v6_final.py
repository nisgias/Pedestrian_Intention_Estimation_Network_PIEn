# models/pipnet_alpha_v6_final.py
"""
PIPNet-Alpha V6.0 — Factorized Space-Time Global Branch with Enriched
Pedestrian Token, Funnel Pooling, and Trajectory Auxiliary Supervision.

Based on V5.2, with the GlobalContextBranchTransformer replaced by a
factorized space-time design that combines insights from three recent
state-of-the-art papers:

  - PedGT (Riaz et al. 2025): Factorized GCN-per-frame + Transformer-temporal
    pattern. We adopt the analogous Spatial-Transformer-per-frame +
    GRU-temporal pattern. The GRU provides a strong sequential inductive
    bias (addressing the supervisor's suggestion to keep a temporal
    recurrent encoder).

  - Achaji et al. 2022 (TEP architecture): Funnel-style spatial pooling
    INSIDE the Transformer stack. Patch tokens are pooled spatially
    between encoder blocks (64 → 16), while the pedestrian token is
    preserved. Addresses the supervisor's suggestion of "CNN-style dim
    reduction inside the Transformer".

  - Achaji et al. 2022 (TED architecture): Joint action + trajectory
    learning improved F1 by +6% in their paper. We add a trajectory
    decoder that predicts bbox displacements from frame 0 as auxiliary
    supervision.

─────────────────────────────────────────────────────────────────────────────
KEY ARCHITECTURAL CHANGES vs V5.2
─────────────────────────────────────────────────────────────────────────────

1. Factorized Space-Time
   V5.2: One big Transformer over T*(NP+1) = 10*65 = 650 tokens with
         block-causal masking.
   V6:   Spatial Transformer runs per-frame (B*T batched). A GRU then
         handles temporal modelling over the per-frame pedestrian-token
         outputs. Far fewer attention operations, stronger temporal
         inductive bias.

2. Enriched Pedestrian Token (answers supervisor's bbox question)
   V5.2: ped_token = Linear(pose_center_2d).
   V6:   ped_token = concat(Linear(pose_center), Linear(bbox_4d)).
         The pedestrian token now carries both pose and bbox geometry
         from the start, before any spatial attention is computed.

3. Funnel Pooling
   V5.2: Constant 64 patch tokens through both Transformer layers.
   V6:   64 patches after Block 1, then spatially pooled to 16 patches
         (the ped token survives the pool untouched). Block 2 attends
         over 17 tokens instead of 65 — cheaper and forces the model to
         summarise spatial info.

4. Trajectory Auxiliary Decoder
   V5.2: Three aux classification heads (kin, local, global).
   V6:   Adds an aux regression head: predict the bbox displacement
         from frame 0, i.e. delta_bbox[t] = bbox[t] - bbox[0].
         The training loop should add an L2 term on `aux_trajectory`.

─────────────────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared attention module (unchanged from V5)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Kinematic branch (unchanged from V5)
# ─────────────────────────────────────────────────────────────────────────────

class KinematicBranch(nn.Module):
    def __init__(self, bbox_dim=4, pose_dim=34, speed_dim=1, hidden_dim=128):
        super().__init__()
        self.pose_gru      = nn.GRU(pose_dim, 64, batch_first=True)
        self.pose_bbox_gru = nn.GRU(64 + bbox_dim, 128, batch_first=True)
        self.full_gru      = nn.GRU(128 + speed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, bbox, pose, speed):
        h_pose, _ = self.pose_gru(pose)
        h_pb = torch.cat([h_pose, bbox], dim=-1)
        h_pb, _ = self.pose_bbox_gru(h_pb)
        h_full = torch.cat([h_pb, speed], dim=-1)
        h_out, _ = self.full_gru(h_full)
        return self.norm(h_out)


# ─────────────────────────────────────────────────────────────────────────────
# Local visual branch (unchanged from V5)
# ─────────────────────────────────────────────────────────────────────────────

class LocalVisualBranch(nn.Module):
    def __init__(self, cnn_dim=512, hidden_dim=256, dropout_p=0.3):
        super().__init__()
        self.content_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.content_fc   = nn.Linear(cnn_dim, 128)
        self.content_norm = nn.LayerNorm(128)

        self.motion_conv3d = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        self.motion_proj = nn.Linear(64, 128)
        self.motion_norm = nn.LayerNorm(128)

        self.content_gru = nn.GRU(128, 128, batch_first=True)
        self.motion_gru  = nn.GRU(128, 128, batch_first=True)
        self.fuse_gru    = nn.GRU(256, hidden_dim, batch_first=True)
        self.out_norm    = nn.LayerNorm(hidden_dim)

        self.content_drop = nn.Dropout(dropout_p)
        self.motion_drop  = nn.Dropout(dropout_p)
        self.fuse_drop    = nn.Dropout(dropout_p)

    def forward(self, local_cnn, local_motion):
        B, T = local_cnn.shape[:2]

        # content
        if local_cnn.dim() == 5:
            cnn_flat  = local_cnn.reshape(B * T, *local_cnn.shape[2:])
            cnn_pooled = self.content_pool(cnn_flat).reshape(B * T, -1)
            cnn_feat  = self.content_fc(cnn_pooled).reshape(B, T, -1)
        else:
            cnn_feat = self.content_fc(local_cnn)
        cnn_feat = self.content_norm(cnn_feat)
        h_content, _ = self.content_gru(cnn_feat)
        h_content = self.content_drop(h_content)

        # motion
        if local_motion.dim() == 5:
            m5d = local_motion.permute(0, 2, 1, 3, 4).contiguous()
            m   = self.motion_conv3d(m5d)
            m   = m.mean(dim=(-1, -2))
            m   = m.transpose(1, 2).contiguous()
            motion_feat = self.motion_proj(m)
        else:
            if local_motion.shape[-1] != 128:
                motion_feat = F.adaptive_avg_pool1d(
                    local_motion.transpose(1, 2), 128).transpose(1, 2)
            else:
                motion_feat = local_motion
        motion_feat = self.motion_norm(motion_feat)
        h_motion, _ = self.motion_gru(motion_feat)
        h_motion = self.motion_drop(h_motion)

        h_cat = torch.cat([h_content, h_motion], dim=-1)
        h_local, _ = self.fuse_gru(h_cat)
        h_local = self.fuse_drop(h_local)
        return self.out_norm(h_local)


# ─────────────────────────────────────────────────────────────────────────────
# Input downsampling helpers (unchanged from V5)
# ─────────────────────────────────────────────────────────────────────────────

class SemanticDownsample(nn.Module):
    def __init__(self, num_classes=20, embed_dim=16, out_channels=16, target_size=64):
        super().__init__()
        self.target_size = target_size
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((target_size, target_size)),
        )

    def forward(self, sem_labels):
        B, T, H, W = sem_labels.shape
        sem_clamped = sem_labels.clamp(0, self.embed.num_embeddings - 1)
        if H == self.target_size and W == self.target_size:
            emb = self.embed(sem_clamped)
            return emb.permute(0, 4, 1, 2, 3).contiguous()
        emb = self.embed(sem_clamped)
        emb = emb.reshape(B * T, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.conv(emb)
        out = out.reshape(B, T, -1, self.target_size, self.target_size)
        return out.permute(0, 2, 1, 3, 4).contiguous()


class DepthDownsample(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, target_size=64):
        super().__init__()
        self.target_size = target_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((target_size, target_size)),
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
        return out.permute(0, 2, 1, 3, 4).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# NEW V6: Global Context Branch — Factorized Space-Time
# ─────────────────────────────────────────────────────────────────────────────

class GlobalContextBranchV6(nn.Module):
    """
    Factorized space-time global branch.

    Per frame:
      1. Stem → 64 patch tokens (8×8 grid, d_model channels).
      2. Build enriched pedestrian token: concat(pose_center_emb, bbox_emb).
      3. Spatial Transformer Block 1 over [ped_token, 64 patches].
      4. Funnel pool patches 64 → 16 (8×8 → 4×4), ped token untouched.
      5. Spatial Transformer Block 2 over [ped_token, 16 patches].
      6. Extract ped token (ViT-CLS style) → (d_model,) per frame.

    Temporal:
      7. GRU over (B, T, d_model) → (B, T, hidden_dim). Causal by design.

    Auxiliary:
      8. Trajectory decoder predicts bbox displacement from frame 0.
    """

    def __init__(
        self,
        sem_num_classes: int = 20,
        sem_embed_dim: int = 16,
        hidden_dim: int = 256,
        target_size: int = 64,
        patch_grid: int = 8,
        funnel_grid: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        ff_dim: int = 256,
        tf_dropout: float = 0.1,
        stem_dropout: float = 0.1,
        pose_dim: int = 34,
        bbox_dim: int = 4,
        gru_dropout: float = 0.0,
    ):
        super().__init__()

        assert target_size % patch_grid == 0
        assert patch_grid % funnel_grid == 0
        assert d_model % n_heads == 0
        assert d_model % 2 == 0, "d_model must be even (ped token = pose_half + bbox_half)"

        self.hidden_dim   = hidden_dim
        self.target_size  = target_size
        self.patch_grid   = patch_grid
        self.funnel_grid  = funnel_grid
        self.d_model      = d_model
        self.n_patches_l1 = patch_grid * patch_grid    # 64
        self.n_patches_l2 = funnel_grid * funnel_grid  # 16
        self.n_keypoints  = pose_dim // 2

        # ── Input downsampling ─────────────────────────────────────────────
        self.sem_down = SemanticDownsample(
            num_classes=sem_num_classes, embed_dim=sem_embed_dim,
            out_channels=sem_embed_dim, target_size=target_size,
        )
        self.depth_down = DepthDownsample(
            in_channels=2, out_channels=2, target_size=target_size,
        )

        # ── Stem: fuse sem + depth channels → d_model ──────────────────────
        stem_in = sem_embed_dim + 2  # 18
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
        )
        self.stem_drop = nn.Dropout(stem_dropout)

        # Spatial patch pooling (downsample feature map → patch_grid)
        self.patch_pool = nn.AdaptiveAvgPool2d((patch_grid, patch_grid))

        # ── Enriched pedestrian token projections ──────────────────────────
        # ped_token = concat(pose_emb, bbox_emb), each of size d_model // 2
        half = d_model // 2
        self.pose_proj = nn.Linear(2, half)        # pose center (x, y) → d/2
        self.bbox_proj = nn.Linear(bbox_dim, half) # bbox (4d) → d/2
        self.ped_pos   = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.ped_pos, std=0.02)

        # ── Spatial positional encodings (level 1: 64 patches, level 2: 16)
        self.spatial_pos_l1 = nn.Parameter(torch.zeros(1, self.n_patches_l1, d_model))
        self.spatial_pos_l2 = nn.Parameter(torch.zeros(1, self.n_patches_l2, d_model))
        nn.init.trunc_normal_(self.spatial_pos_l1, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_l2, std=0.02)

        # ── Spatial Transformer blocks (factorized: per-frame attention) ───
        def make_block():
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=tf_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
        self.spatial_block_1 = make_block()
        self.spatial_block_2 = make_block()

        # ── Temporal GRU (supervisor's suggestion: keep recurrent temporal)
        self.temporal_gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.temporal_drop = nn.Dropout(gru_dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

        # ── Trajectory auxiliary decoder (Achaji TED-style joint learning) ─
        # Predicts bbox displacement from frame 0: delta_bbox[t] = bbox[t] - bbox[0]
        self.traj_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, bbox_dim),
        )

    # -------------------------------------------------------------------------

    def _funnel_pool(self, x: torch.Tensor, current_P: int, new_P: int) -> torch.Tensor:
        """
        Spatially pool patch tokens, keep the pedestrian token (index 0) unchanged.

        Args:
            x: (B, 1 + current_P*current_P, d_model)
            current_P: current spatial grid side
            new_P: new spatial grid side (must divide current_P)
        Returns:
            x_pooled: (B, 1 + new_P*new_P, d_model)
        """
        B, N, d = x.shape
        ped     = x[:, 0:1, :]                                    # (B, 1, d)
        patches = x[:, 1:, :]                                     # (B, P*P, d)
        patches_2d = patches.reshape(B, current_P, current_P, d).permute(0, 3, 1, 2)
        patches_pooled = F.adaptive_avg_pool2d(patches_2d, (new_P, new_P))
        patches_pooled = patches_pooled.permute(0, 2, 3, 1).reshape(B, new_P * new_P, d)
        return torch.cat([ped, patches_pooled], dim=1)            # (B, 1 + new_P², d)

    # -------------------------------------------------------------------------

    def forward(
        self,
        sem_labels: torch.Tensor,
        cat_depth:  torch.Tensor,
        pose:       torch.Tensor,
        bbox:       torch.Tensor,
    ):
        """
        Returns:
            h_temporal: (B, T, hidden_dim) — features for downstream fusion.
            traj_pred:  (B, T, 4)          — predicted bbox displacements (aux).
        """
        B, T = sem_labels.shape[:2]
        P  = self.patch_grid
        d  = self.d_model

        # ── Step 1: Downsample + Stem (per-frame conv operations) ──────────
        sem_emb  = self.sem_down(sem_labels)
        depth_5d = self.depth_down(cat_depth)

        sem_bt   = sem_emb.permute(0, 2, 1, 3, 4).reshape(
            B * T, sem_emb.shape[1], self.target_size, self.target_size)
        depth_bt = depth_5d.permute(0, 2, 1, 3, 4).reshape(
            B * T, 2, self.target_size, self.target_size)

        x_bt = torch.cat([sem_bt, depth_bt], dim=1)               # (B*T, 18, 64, 64)
        x_bt = self.stem(x_bt)                                    # (B*T, d, 64, 64)
        x_bt = self.stem_drop(x_bt)

        # ── Step 2: Spatial patch pooling: 64×64 → 8×8 ─────────────────────
        x_bt = self.patch_pool(x_bt)                              # (B*T, d, 8, 8)
        patches = x_bt.permute(0, 2, 3, 1).reshape(B * T, self.n_patches_l1, d)
        patches = patches + self.spatial_pos_l1

        # ── Step 3: Build enriched pedestrian token (pose + bbox) ──────────
        pose_2d = pose.view(B, T, self.n_keypoints, 2)
        ped_center = pose_2d.mean(dim=2)                          # (B, T, 2)

        pose_emb = self.pose_proj(ped_center.reshape(B * T, 2))   # (B*T, d/2)
        bbox_emb = self.bbox_proj(bbox.reshape(B * T, -1))        # (B*T, d/2)
        ped_token = torch.cat([pose_emb, bbox_emb], dim=-1)       # (B*T, d)
        ped_token = ped_token.unsqueeze(1) + self.ped_pos         # (B*T, 1, d)

        # ── Step 4: Spatial Transformer Block 1 (65 tokens, per-frame) ─────
        x = torch.cat([ped_token, patches], dim=1)                # (B*T, 65, d)
        x = self.spatial_block_1(x)

        # ── Step 5: Funnel pool: 64 patches → 16, ped token untouched ──────
        x = self._funnel_pool(x, current_P=P, new_P=self.funnel_grid)  # (B*T, 17, d)
        # Refresh patch positional encoding at the new resolution
        ped_l2     = x[:, 0:1, :]
        patches_l2 = x[:, 1:, :] + self.spatial_pos_l2
        x = torch.cat([ped_l2, patches_l2], dim=1)

        # ── Step 6: Spatial Transformer Block 2 (17 tokens, per-frame) ─────
        x = self.spatial_block_2(x)                               # (B*T, 17, d)

        # ── Step 7: ViT-style CLS extraction: take pedestrian token only ───
        h_spatial = x[:, 0, :].reshape(B, T, d)                   # (B, T, d_model)

        # ── Step 8: Temporal GRU (naturally causal) ────────────────────────
        h_temporal, _ = self.temporal_gru(h_spatial)              # (B, T, hidden_dim)
        h_temporal = self.temporal_drop(h_temporal)
        h_out = self.out_norm(h_temporal)

        # ── Step 9: Trajectory auxiliary prediction ────────────────────────
        traj_pred = self.traj_decoder(h_out)                      # (B, T, 4)

        return h_out, traj_pred


# ─────────────────────────────────────────────────────────────────────────────
# Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class PIPNetAlphaV6Final(nn.Module):
    """
    PIPNet-Alpha V6 with factorized space-time global branch.

    Identical to V5.2 except:
      - GlobalContextBranchTransformer → GlobalContextBranchV6
      - The global branch consumes bbox in addition to pose/sem/depth.
      - The aux dict now contains `aux_trajectory` (B, T, 4) which the
        training loop should regress against (bbox[t] - bbox[0]).

    visual_fuse_weights shape: (B, T=10) — unchanged.
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
        global_target_size: int = 64,
        local_dropout_p: float = 0.3,
        # V6 global branch hyper-parameters
        global_patch_grid: int = 8,
        global_funnel_grid: int = 4,
        global_d_model: int = 128,
        global_n_heads: int = 4,
        global_ff_dim: int = 256,
        global_tf_dropout: float = 0.1,
        global_stem_dropout: float = 0.1,
        global_gru_dropout: float = 0.0,
    ):
        super().__init__()
        self.branch_out_dim = branch_out_dim
        self.final_dim = final_dim

        # ── Kinematic branch (unchanged) ────────────────────────────────────
        self.kinematic_branch = KinematicBranch(
            bbox_dim=bbox_dim, pose_dim=pose_dim,
            speed_dim=speed_dim, hidden_dim=128,
        )
        self.kin_attn = PIPNetAttention(128, branch_out_dim, dropout_p)
        self.norm_kin = nn.LayerNorm(branch_out_dim)

        # ── Local visual branch (unchanged) ─────────────────────────────────
        self.local_branch = LocalVisualBranch(
            cnn_dim=512, hidden_dim=256, dropout_p=local_dropout_p,
        )
        self.local_attn = PIPNetAttention(256, branch_out_dim, dropout_p)
        self.norm_local = nn.LayerNorm(branch_out_dim)

        # ── Global context branch (V6: factorized space-time) ──────────────
        self.global_branch = GlobalContextBranchV6(
            sem_num_classes=sem_num_classes,
            sem_embed_dim=sem_embed_dim,
            hidden_dim=256,
            target_size=global_target_size,
            patch_grid=global_patch_grid,
            funnel_grid=global_funnel_grid,
            d_model=global_d_model,
            n_heads=global_n_heads,
            ff_dim=global_ff_dim,
            tf_dropout=global_tf_dropout,
            stem_dropout=global_stem_dropout,
            pose_dim=pose_dim,
            bbox_dim=bbox_dim,
            gru_dropout=global_gru_dropout,
        )

        # ── Visual fusion (unchanged: z_local + h_global → z_visual) ───────
        self.visual_fuse_attn = PIPNetAttention(
            hidden_dim=branch_out_dim + 256,
            out_dim=branch_out_dim,
            dropout_p=dropout_p,
        )
        self.norm_visual = nn.LayerNorm(branch_out_dim)

        # ── Final attention: [z_visual, z_kin] ─────────────────────────────
        self.final_attn = PIPNetAttention(branch_out_dim, final_dim, dropout_p)

        # ── Auxiliary classification heads (unchanged) ─────────────────────
        self.aux_kin   = nn.Linear(branch_out_dim, 1)
        self.aux_local = nn.Linear(branch_out_dim, 1)

        self.global_attn = PIPNetAttention(256, branch_out_dim, dropout_p)
        self.norm_global = nn.LayerNorm(branch_out_dim)
        self.aux_global  = nn.Linear(branch_out_dim, 1)

        # ── Output ──────────────────────────────────────────────────────────
        self.fc_drop = nn.Dropout(dropout_p)
        self.fc_out  = nn.Linear(final_dim, 1)

    # -------------------------------------------------------------------------

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

        # ── Kinematic ────────────────────────────────────────────────────────
        h_kin = self.kinematic_branch(bbox, pose, speed)
        z_kin, _ = self.kin_attn(h_kin)
        z_kin = self.norm_kin(z_kin)

        # ── Local visual ─────────────────────────────────────────────────────
        h_local = self.local_branch(local_cnn, local_motion)
        z_local, _ = self.local_attn(h_local)
        z_local = self.norm_local(z_local)

        # ── Global context (V6: factorized) → (B, T, 256) + traj_pred ───────
        h_global, traj_pred = self.global_branch(sem_labels, cat_depth, pose, bbox)

        # ── Visual fusion ─────────────────────────────────────────────────────
        z_local_exp = z_local.unsqueeze(1).expand(-1, T, -1)

        if getattr(self, '_ablate_global', False):
            h_global_for_fusion = torch.zeros_like(h_global)
        else:
            h_global_for_fusion = h_global

        h_fused = torch.cat([z_local_exp, h_global_for_fusion], dim=-1)

        z_visual, alpha_visual = self.visual_fuse_attn(h_fused)
        z_visual = self.norm_visual(z_visual)

        # ── Final fusion ──────────────────────────────────────────────────────
        z_vk = torch.stack([z_visual, z_kin], dim=1)
        z_final, alpha_final = self.final_attn(z_vk, use_mean_query=True)

        z_final = self.fc_drop(z_final)
        logit   = self.fc_out(z_final)

        if return_aux:
            z_global_att, _ = self.global_attn(h_global, use_mean_query=True)
            z_global_att = self.norm_global(z_global_att)

            return {
                'logit':               logit,
                'aux_kin':             self.aux_kin(z_kin),
                'aux_local':           self.aux_local(z_local),
                'aux_global':          self.aux_global(z_global_att),
                'aux_trajectory':      traj_pred,           # NEW: (B, T, 4)
                'modality_weights':    alpha_final,
                'visual_fuse_weights': alpha_visual,
                'cross_attn_weights':  alpha_visual,
                'z_kin':    z_kin,
                'z_local':  z_local,
                'z_cross':  z_visual,
                'z_visual': z_visual,
            }

        return logit


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 72)
    print("PIPNet-Alpha V6.0 — Factorized Space-Time Global Branch")
    print("                  + Enriched Ped Token (pose+bbox)")
    print("                  + Funnel Pooling (TEP-style)")
    print("                  + GRU Temporal Encoder")
    print("                  + Trajectory Auxiliary Head (TED-style)")
    print("=" * 72)

    model = PIPNetAlphaV6Final()
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}\n")

    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<25s}: {n:>10,}")

    # Break down the global branch
    print("\nGlobal-branch sub-modules:")
    for sub_name, sub_mod in model.global_branch.named_children():
        n = sum(p.numel() for p in sub_mod.parameters())
        print(f"    {sub_name:<22s}: {n:>10,}")

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

    with torch.no_grad():
        out = model(batch, return_aux=True)

    checks = [
        ("logit",               (B, 1)),
        ("aux_kin",             (B, 1)),
        ("aux_local",           (B, 1)),
        ("aux_global",          (B, 1)),
        ("aux_trajectory",      (B, T, 4)),    # NEW
        ("modality_weights",    (B, 2)),
        ("visual_fuse_weights", (B, T)),
    ]

    print("Output shapes:")
    all_ok = True
    for key, expected in checks:
        got = tuple(out[key].shape)
        ok  = got == expected
        all_ok = all_ok and ok
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {key:<22s}: {str(got):<20s}  (expected {expected})")

    print()
    if all_ok:
        print("All shape assertions passed.")
    else:
        print("SHAPE MISMATCH — see above.")
        sys.exit(1)

    # Ablation sanity check
    model._ablate_global = True
    with torch.no_grad():
        out_abl = model(batch, return_aux=False)
    model._ablate_global = False
    print(f"Ablation path (global zeroed): logit shape {tuple(out_abl.shape)}  OK")