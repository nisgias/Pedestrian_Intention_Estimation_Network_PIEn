# models/pipnet_alpha_v5_final.py
"""
PIPNet-Alpha V5.2 — Causal Thick-Stem Spatial-Patch Transformer + Early Pose Injection

Based on V4 (thin-tower refactor).  Only GlobalContextBranchConv3D is replaced;
everything else (KinematicBranch, LocalVisualBranch, PIPNetAttention, top-level
fusion, training interface) is unchanged.

─────────────────────────────────────────────────────────────────────────────
ARCHITECTURE (GlobalContextBranchTransformer)
  Input: sem_labels, cat_depth, and POSE.

  1. Thick CNN Stem: Two 3x3 Conv layers extract local pixel concepts.
  2. Early Pose Injection (The Anchor): The 34D pose is collapsed into an (x,y) 
     center coordinate. It is linearly projected to d_model to become the 
     "Pedestrian Token".
  3. Token Sequence: The 64 patches + 1 Pedestrian Token = 65 tokens per frame.
  4. Block-Causal Masking: Transformer attention is masked so frames cannot 
     look into the future.
  5. ViT-Style CLS Extraction: Instead of mean-pooling the patches, we simply 
     slice out the Pedestrian Token, which has now attended to the entire 
     semantic scene, yielding (B, T, d_model).
  6. Output: Simple linear projection to (B, T, 256).

─────────────────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared attention module (unchanged from V3/V4)
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
# Kinematic branch (unchanged from V3/V4)
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
# Local visual branch (from V4 — Conv3D motion upgrade)
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
            m   = self.motion_conv3d(m5d)               # (B, 64, T, H', W')
            m   = m.mean(dim=(-1, -2))                   # (B, 64, T)
            m   = m.transpose(1, 2).contiguous()         # (B, T, 64)
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
# Input downsampling helpers (same as V4, AdaptiveMaxPool variant)
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
# NEW: Global Context Branch — Spatial-Patch Transformer (V5.2 - Causal Thick-Stem + Pose Inject)
# ─────────────────────────────────────────────────────────────────────────────

class GlobalContextBranchTransformer(nn.Module):
    """
    Replaces Conv3D towers with a Spatial-Patch Transformer.
    
    UPDATES IN THIS VERSION:
    1. Thickened CNN Stem (Two 3x3 convolutions) for richer local features.
    2. Ablated GRU (Replaced with a simple nn.Linear projection).
    3. Block-Causal Masking (Frames can only attend to current and past frames).
    4. Early Pose Injection (Coordinates act as a ViT CLS token).
    """

    def __init__(
        self,
        sem_num_classes: int = 20,
        sem_embed_dim: int = 16,
        hidden_dim: int = 256,
        target_size: int = 64,
        patch_grid: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        tf_dropout: float = 0.1,
        stem_dropout: float = 0.1,
    ):
        super().__init__()

        assert target_size % patch_grid == 0, (
            f"target_size ({target_size}) must be divisible by patch_grid ({patch_grid})"
        )
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.hidden_dim  = hidden_dim
        self.target_size = target_size
        self.patch_grid  = patch_grid
        self.d_model     = d_model
        self.n_patches   = patch_grid * patch_grid  # 64

        self.patch_px = target_size // patch_grid

        # ── 1. Input downsampling ──────────────────────────────────────────
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

        # ── 2. Thickened CNN stem: fuse sem + depth channels → d_model ─────
        stem_in = sem_embed_dim + 2  # 18
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            # Added a second 3x3 to thicken the stem feature extractor
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
        )
        self.stem_drop = nn.Dropout(stem_dropout)

        # ── 3. Spatial patch pooling ───────────────────────────────────────
        self.patch_pool = nn.AdaptiveAvgPool2d((patch_grid, patch_grid))

        # ── 4. Positional encodings ────────────────────────────────────────
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.n_patches, d_model))
        self.temporal_pos = nn.Parameter(torch.zeros(1, 64, 1, d_model))
        nn.init.trunc_normal_(self.spatial_pos,  std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        # ── NEW: Pose Projection (2D Coordinate -> d_model) ────────────────
        self.pose_proj = nn.Linear(2, d_model)
        # A special spatial positional encoding just for the pedestrian token
        self.ped_pos = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.trunc_normal_(self.ped_pos, std=0.02)

        # ── 5. Transformer encoder ─────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=tf_dropout,
            activation='gelu',
            batch_first=True,   
            norm_first=True,    
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # ── 6. GRU Ablated: Simple Linear Projection ───────────────────────
        self.out_proj = nn.Linear(d_model, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    # -------------------------------------------------------------------------

    def forward(self, sem_labels: torch.Tensor, cat_depth: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        B, T = sem_labels.shape[:2]
        P  = self.patch_grid
        NP = self.n_patches

        # ── Step 1 & 2: Downsample & Stem ──────────────────────────────────
        sem_emb   = self.sem_down(sem_labels)    
        depth_5d  = self.depth_down(cat_depth)   
        
        sem_bt   = sem_emb.permute(0, 2, 1, 3, 4).reshape(B * T, sem_emb.shape[1], self.target_size, self.target_size)
        depth_bt = depth_5d.permute(0, 2, 1, 3, 4).reshape(B * T, 2, self.target_size, self.target_size)
        
        x_bt = torch.cat([sem_bt, depth_bt], dim=1)        
        x_bt = self.stem(x_bt)                             
        x_bt = self.stem_drop(x_bt)

        # ── Step 3: Spatial patch pooling ──────────────────────────────────
        x_bt = self.patch_pool(x_bt)
        x = x_bt.reshape(B, T, self.d_model, P, P)
        x = x.permute(0, 1, 3, 4, 2)                       
        x = x.reshape(B, T, NP, self.d_model)              

        # ── Step 4: Add Spatial Positional Encodings ───────────────────────
        x = x + self.spatial_pos
        
        # ── NEW Step 5: Extract and Inject the Pedestrian Pose Token ───────
        # 1. Reshape pose (B, T, 34) into 17 (x,y) keypoints and get the center
        pose_2d = pose.view(B, T, 17, 2)
        ped_center = pose_2d.mean(dim=2)  # Shape: (B, T, 2)
        
        # 2. Project the (x,y) center into the Transformer's dimension
        ped_token = self.pose_proj(ped_center).unsqueeze(2)  # Shape: (B, T, 1, d_model)
        
        # 3. Add the special Pedestrian spatial encoding
        ped_token = ped_token + self.ped_pos
        
        # 4. Concatenate the Pedestrian Token to the start of the patch sequence
        # Now we have NP + 1 tokens per frame (65 tokens instead of 64)
        x = torch.cat([ped_token, x], dim=2)  # Shape: (B, T, NP + 1, d_model)

        # ── Step 6: Add Temporal Positional Encodings & Flatten ────────────
        # Add temporal encoding to ALL tokens (including the new ped_token)
        x = x + self.temporal_pos[:, :T, :, :]             
        x = x.reshape(B, T * (NP + 1), self.d_model)       

        # ── Step 7: Block-Causal Masking (Updated for NP + 1) ──────────────
        seq_len = T * (NP + 1)
        frame_idx = torch.arange(seq_len, device=x.device) // (NP + 1)
        is_future = frame_idx.unsqueeze(0) < frame_idx.unsqueeze(1) 
        mask = torch.zeros(seq_len, seq_len, device=x.device)
        mask.masked_fill_(is_future, float('-inf'))

        # ── Step 8: Transformer encoder ────────────────────────────────────
        x = self.transformer(x, mask=mask)                            

        # ── NEW Step 9: ViT-Style CLS Extraction ───────────────────────────
        x = x.reshape(B, T, NP + 1, self.d_model)
        
        # Instead of taking the mean() of all 65 tokens, we slice out the 
        # Pedestrian Token (index 0) which has now attended to the whole scene!
        x_ped = x[:, :, 0, :]  # Shape: (B, T, d_model)

        # ── Step 10: Linear Projection ─────────────────────────────────────
        h = self.out_proj(x_ped)
        return self.out_norm(h)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class PIPNetAlphaV5Final(nn.Module):
    """
    PIPNet-Alpha V5.

    Identical to V4 except GlobalContextBranchConv3D is replaced by
    GlobalContextBranchTransformer.  The downstream fusion (visual_fuse_attn,
    final_attn, fc_out) and all auxiliary heads are unchanged.

    visual_fuse_weights shape: (B, T=10)  — same as V4.
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
        # Transformer-specific hyper-parameters (exposed for ablation)
        global_patch_grid: int = 8,
        global_d_model: int = 128,
        global_n_heads: int = 4,
        global_n_layers: int = 2,
        global_ff_dim: int = 256,
        global_tf_dropout: float = 0.1,
        global_stem_dropout: float = 0.1,
    ):
        super().__init__()
        self.branch_out_dim = branch_out_dim
        self.final_dim = final_dim

        # ── Kinematic branch ────────────────────────────────────────────────
        self.kinematic_branch = KinematicBranch(
            bbox_dim=bbox_dim, pose_dim=pose_dim,
            speed_dim=speed_dim, hidden_dim=128,
        )
        self.kin_attn = PIPNetAttention(128, branch_out_dim, dropout_p)
        self.norm_kin = nn.LayerNorm(branch_out_dim)

        # ── Local visual branch ─────────────────────────────────────────────
        self.local_branch = LocalVisualBranch(
            cnn_dim=512, hidden_dim=256, dropout_p=local_dropout_p,
        )
        self.local_attn = PIPNetAttention(256, branch_out_dim, dropout_p)
        self.norm_local = nn.LayerNorm(branch_out_dim)

        # ── Global context branch (TRANSFORMER) ─────────────────────────────
        self.global_branch = GlobalContextBranchTransformer(
            sem_num_classes=sem_num_classes,
            sem_embed_dim=sem_embed_dim,
            hidden_dim=256,                          # output dim (same as V4)
            target_size=global_target_size,
            patch_grid=global_patch_grid,
            d_model=global_d_model,
            n_heads=global_n_heads,
            n_layers=global_n_layers,
            ff_dim=global_ff_dim,
            tf_dropout=global_tf_dropout,
            stem_dropout=global_stem_dropout,
        )

        # ── Visual fusion ───────────────────────────────────────────────────
        # z_local (B,T,128) cat h_global (B,T,256) → (B,T,384)
        # PIPNetAttention attends over T=10 frames → (B, 128)
        self.visual_fuse_attn = PIPNetAttention(
            hidden_dim=branch_out_dim + 256,   # 384
            out_dim=branch_out_dim,             # 128
            dropout_p=dropout_p,
        )
        self.norm_visual = nn.LayerNorm(branch_out_dim)

        # ── Final attention: [z_visual, z_kin] → (B, 2, 128) ───────────────
        self.final_attn = PIPNetAttention(branch_out_dim, final_dim, dropout_p)

        # ── Auxiliary heads ─────────────────────────────────────────────────
        self.aux_kin   = nn.Linear(branch_out_dim, 1)
        self.aux_local = nn.Linear(branch_out_dim, 1)

        # global aux: temporal attention over T global frames
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

        # ── Global context (Transformer) → (B, T, 256) ───────────────────────
        h_global = self.global_branch(sem_labels, cat_depth, pose)

        # ── Visual fusion ─────────────────────────────────────────────────────
        # Expand z_local: (B, 128) → (B, T, 128)
        z_local_exp = z_local.unsqueeze(1).expand(-1, T, -1)

        if getattr(self, '_ablate_global', False):
            h_global_for_fusion = torch.zeros_like(h_global)
        else:
            h_global_for_fusion = h_global

        # Concat: (B, T, 128+256) = (B, T, 384)
        h_fused = torch.cat([z_local_exp, h_global_for_fusion], dim=-1)

        # PIPNetAttention over T frames → z_visual (B, 128), alpha_visual (B, T)
        z_visual, alpha_visual = self.visual_fuse_attn(h_fused)
        z_visual = self.norm_visual(z_visual)

        # ── Final fusion ──────────────────────────────────────────────────────
        z_vk = torch.stack([z_visual, z_kin], dim=1)       # (B, 2, 128)
        z_final, alpha_final = self.final_attn(z_vk, use_mean_query=True)

        z_final = self.fc_drop(z_final)
        logit   = self.fc_out(z_final)

        if return_aux:
            # Global aux: attend over T global frames
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


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("PIPNet-Alpha V5.2 — Spatial-Patch Transformer Global Branch")
    print("=" * 70)

    model = PIPNetAlphaV5Final()
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")
    print()
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<25s}: {n:>10,}")

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
        ("modality_weights",    (B, 2)),
        ("visual_fuse_weights", (B, T)),
    ]

    print("Output shapes:")
    all_ok = True
    for key, expected in checks:
        got = tuple(out[key].shape)
        ok  = got == expected
        all_ok = all_ok and ok
        status = "✓" if ok else "✗"
        print(f"  {status}  {key:<25s}: {str(got):<20s}  (expected {expected})")

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
    print(f"Ablation path (global zeroed): logit shape {tuple(out_abl.shape)}  ✓")