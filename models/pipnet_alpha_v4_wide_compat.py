"""
PIPNet-Alpha V4 — WIDE CHECKPOINT-COMPATIBLE VERSION

Use this file only to load/evaluate checkpoints trained with the older/wide V4
architecture, e.g. checkpoints where the state_dict contains:

  - local_branch.motion_conv.*              (2D motion conv, not motion_conv3d)
  - global_branch.sem_conv3d channels 64→128→256
  - global_branch.depth_conv3d channels 64→128→256
  - global_branch.sem_fc.0.weight shape   [256, 16384]
  - global_branch.depth_fc.0.weight shape [256, 16384]

Do NOT overwrite your current thin V4 model if you are still training it.
Place this file as:

    models/pipnet_alpha_v4_wide_compat.py

Then in eval_tte_bins.py use:

    from models.pipnet_alpha_v4_wide_compat import PIPNetAlphaV4Final
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention module
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
# Kinematic branch
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
# Local visual branch — CHECKPOINT-COMPATIBLE OLD MOTION CONV
# ---------------------------------------------------------------------------

class LocalVisualBranch(nn.Module):
    def __init__(self, cnn_dim=512, hidden_dim=256, dropout_p=0.3):
        super().__init__()

        self.content_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.content_fc = nn.Linear(cnn_dim, 128)
        self.content_norm = nn.LayerNorm(128)

        # IMPORTANT: old checkpoint expects local_branch.motion_conv.* keys.
        # Module indices match the checkpoint:
        #   0 Conv2d, 1 BatchNorm2d, 2 ReLU,
        #   3 Conv2d, 4 BatchNorm2d, 5 ReLU, 6 AdaptiveAvgPool2d
        self.motion_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.motion_proj = nn.Linear(64, 128)
        self.motion_norm = nn.LayerNorm(128)

        self.content_gru = nn.GRU(128, 128, batch_first=True)
        self.motion_gru = nn.GRU(128, 128, batch_first=True)
        self.fuse_gru = nn.GRU(256, hidden_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.content_drop = nn.Dropout(dropout_p)
        self.motion_drop = nn.Dropout(dropout_p)
        self.fuse_drop = nn.Dropout(dropout_p)

    def forward(self, local_cnn, local_motion):
        B, T = local_cnn.shape[:2]

        # Content branch
        if local_cnn.dim() == 5:
            cnn_flat = local_cnn.reshape(B * T, *local_cnn.shape[2:])
            cnn_pooled = self.content_pool(cnn_flat).reshape(B * T, -1)
            cnn_feat = self.content_fc(cnn_pooled).reshape(B, T, -1)
        else:
            cnn_feat = self.content_fc(local_cnn)

        cnn_feat = self.content_norm(cnn_feat)
        h_content, _ = self.content_gru(cnn_feat)
        h_content = self.content_drop(h_content)

        # Motion branch — old 2D-per-frame path
        if local_motion.dim() == 5:
            motion_flat = local_motion.reshape(B * T, *local_motion.shape[2:])
            motion_feat = self.motion_conv(motion_flat)
            motion_feat = motion_feat.reshape(B * T, -1)
            motion_feat = self.motion_proj(motion_feat).reshape(B, T, -1)
        else:
            if local_motion.shape[-1] != 128:
                motion_feat = F.adaptive_avg_pool1d(
                    local_motion.transpose(1, 2), 128
                ).transpose(1, 2)
            else:
                motion_feat = local_motion

        motion_feat = self.motion_norm(motion_feat)
        h_motion, _ = self.motion_gru(motion_feat)
        h_motion = self.motion_drop(h_motion)

        h_cat = torch.cat([h_content, h_motion], dim=-1)
        h_local, _ = self.fuse_gru(h_cat)
        h_local = self.fuse_drop(h_local)
        return self.out_norm(h_local)


# ---------------------------------------------------------------------------
# Downsampling helpers
# ---------------------------------------------------------------------------

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
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        return out


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
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        return out


# ---------------------------------------------------------------------------
# Global context branch — WIDE CHECKPOINT-COMPATIBLE VERSION
#
# Dimension trace, target_size=64:
#   sem_down output   : (B, 16, T, 64, 64)
#   depth_down output : (B,  2, T, 64, 64)
#   sem_conv3d        : (B, 256, T, 8, 8)   via 64→128→256
#   depth_conv3d      : (B, 256, T, 8, 8)   via 64→128→256
#   flatten           : (B, T, 256*8*8) = (B, T, 16384)
#   sem_fc/depth_fc   : (B, T, 256)
#   concat            : (B, T, 512)
#   temporal_gru      : (B, T, 256)
# ---------------------------------------------------------------------------

class GlobalContextBranchConv3D(nn.Module):
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

        # Semantic tower: 16 → 64 → 128 → 256
        self.sem_conv3d = nn.Sequential(
            nn.Conv3d(sem_embed_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),
        )

        # Depth tower: 2 → 64 → 128 → 256
        self.depth_conv3d = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(dropout_p),
        )

        flat_dim = 256 * self._SPATIAL_SIDE * self._SPATIAL_SIDE  # 16384

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

        self.temporal_gru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sem_labels: torch.Tensor, cat_depth: torch.Tensor) -> torch.Tensor:
        B, T = sem_labels.shape[:2]

        sem_emb = self.sem_down(sem_labels)
        depth_5d = self.depth_down(cat_depth)

        sem_feat = self.sem_conv3d(sem_emb)       # (B, 256, T, 8, 8)
        depth_feat = self.depth_conv3d(depth_5d)  # (B, 256, T, 8, 8)

        sem_flat = sem_feat.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
        depth_flat = depth_feat.permute(0, 2, 1, 3, 4).reshape(B, T, -1)

        z_sem = self.sem_fc(sem_flat)
        z_depth = self.depth_fc(depth_flat)

        combined = torch.cat([z_sem, z_depth], dim=-1)
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

        self.kinematic_branch = KinematicBranch(
            bbox_dim=bbox_dim,
            pose_dim=pose_dim,
            speed_dim=speed_dim,
            hidden_dim=128,
        )
        self.kin_attn = PIPNetAttention(128, branch_out_dim, dropout_p)
        self.norm_kin = nn.LayerNorm(branch_out_dim)

        self.local_branch = LocalVisualBranch(
            cnn_dim=512,
            hidden_dim=256,
            dropout_p=local_dropout_p,
        )
        self.local_attn = PIPNetAttention(256, branch_out_dim, dropout_p)
        self.norm_local = nn.LayerNorm(branch_out_dim)

        self.global_branch = GlobalContextBranchConv3D(
            sem_num_classes=sem_num_classes,
            sem_embed_dim=sem_embed_dim,
            hidden_dim=256,
            target_size=global_target_size,
            dropout_p=global_dropout_p,
        )

        self.visual_fuse_attn = PIPNetAttention(
            hidden_dim=branch_out_dim + 256,
            out_dim=branch_out_dim,
            dropout_p=dropout_p,
        )
        self.norm_visual = nn.LayerNorm(branch_out_dim)

        self.final_attn = PIPNetAttention(branch_out_dim, final_dim, dropout_p)

        self.aux_kin = nn.Linear(branch_out_dim, 1)
        self.aux_local = nn.Linear(branch_out_dim, 1)

        self.global_attn = PIPNetAttention(256, branch_out_dim, dropout_p)
        self.norm_global = nn.LayerNorm(branch_out_dim)
        self.aux_global = nn.Linear(branch_out_dim, 1)

        self.fc_drop = nn.Dropout(dropout_p)
        self.fc_out = nn.Linear(final_dim, 1)

    def forward(self, batch: dict, return_aux: bool = False):
        bbox = batch["bbox"]
        pose = batch["pose"]
        speed = batch["speed"]
        local_cnn = batch["local_cnn"]
        local_motion = batch["local_motion"]
        sem_labels = batch["sem_labels"]
        cat_depth = batch["cat_depth"]

        T = bbox.size(1)

        h_kin = self.kinematic_branch(bbox, pose, speed)
        z_kin, _ = self.kin_attn(h_kin)
        z_kin = self.norm_kin(z_kin)

        h_local = self.local_branch(local_cnn, local_motion)
        z_local, _ = self.local_attn(h_local)
        z_local = self.norm_local(z_local)

        h_global = self.global_branch(sem_labels, cat_depth)

        z_local_expanded = z_local.unsqueeze(1).expand(-1, T, -1)

        if getattr(self, "_ablate_global", False):
            h_global_for_fusion = torch.zeros_like(h_global)
        else:
            h_global_for_fusion = h_global

        h_fused = torch.cat([z_local_expanded, h_global_for_fusion], dim=-1)
        z_visual, alpha_visual = self.visual_fuse_attn(h_fused)
        z_visual = self.norm_visual(z_visual)

        z_visual_kin = torch.stack([z_visual, z_kin], dim=1)
        z_final, alpha_final = self.final_attn(z_visual_kin, use_mean_query=True)

        z_final = self.fc_drop(z_final)
        logit = self.fc_out(z_final)

        if return_aux:
            z_global_att, _ = self.global_attn(h_global, use_mean_query=True)
            z_global_att = self.norm_global(z_global_att)

            return {
                "logit": logit,
                "aux_kin": self.aux_kin(z_kin),
                "aux_local": self.aux_local(z_local),
                "aux_global": self.aux_global(z_global_att),
                "modality_weights": alpha_final,
                "visual_fuse_weights": alpha_visual,
                "cross_attn_weights": alpha_visual,
                "z_kin": z_kin,
                "z_local": z_local,
                "z_cross": z_visual,
                "z_visual": z_visual,
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

    B, T = 2, 10
    batch = {
        "bbox": torch.randn(B, T, 4),
        "pose": torch.randn(B, T, 34),
        "speed": torch.randn(B, T, 1),
        "local_cnn": torch.randn(B, T, 512, 7, 7),
        "local_motion": torch.randn(B, T, 2, 224, 224),
        "sem_labels": torch.randint(0, 20, (B, T, 384, 672)),
        "cat_depth": torch.randn(B, T, 2, 384, 672),
    }

    out = model(batch, return_aux=True)

    print(f"logit shape:               {out['logit'].shape}         expect (B=2, 1)")
    print(f"aux_kin shape:             {out['aux_kin'].shape}       expect (B=2, 1)")
    print(f"aux_global shape:          {out['aux_global'].shape}    expect (B=2, 1)")
    print(f"modality_weights shape:    {out['modality_weights'].shape}   expect (B=2, 2)")
    print(f"visual_fuse_weights shape: {out['visual_fuse_weights'].shape} expect (B=2, T=10)")

    assert out["logit"].shape == (B, 1)
    assert out["aux_kin"].shape == (B, 1)
    assert out["aux_global"].shape == (B, 1)
    assert out["modality_weights"].shape == (B, 2)
    assert out["visual_fuse_weights"].shape == (B, T)
    print("\nAll shape assertions passed.")
