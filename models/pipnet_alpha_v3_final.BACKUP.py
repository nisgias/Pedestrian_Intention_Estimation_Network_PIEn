# models/pipnet_alpha_v3_final.py
"""
PIPNet-Alpha V3 — FINAL: PIP-Net Faithful with Conv3D + Spatial Preservation

Original 7.66M architecture restored with one targeted addition:
  - Dropout (0.5) before final FC layer

Conv3D channels: 64→128→256 (original width)
No extra GRU dropout, no spatial dropout — those hurt performance.
Learned downsamplers retained for resolution agnosticism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class LocalVisualBranch(nn.Module):
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
        return self.out_norm(h_local)


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
            nn.AdaptiveAvgPool2d((target_size, target_size)),
        )

    def forward(self, sem_labels):
        B, T, H, W = sem_labels.shape
        if H == self.target_size and W == self.target_size:
            sem_clamped = sem_labels.clamp(0, self.embed.num_embeddings - 1)
            emb = self.embed(sem_clamped)
            return emb.permute(0, 4, 1, 2, 3).contiguous()
        sem_clamped = sem_labels.clamp(0, self.embed.num_embeddings - 1)
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
            nn.AdaptiveAvgPool2d((target_size, target_size)),
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


class GlobalContextBranchConv3D(nn.Module):
    """Original Conv3D channels: 64→128→256. Learned downsamplers for resolution agnosticism."""
    def __init__(self, sem_num_classes=20, sem_embed_dim=16, hidden_dim=256, target_size=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_size = target_size

        self.sem_down = SemanticDownsample(
            num_classes=sem_num_classes, embed_dim=sem_embed_dim,
            out_channels=sem_embed_dim, target_size=target_size,
        )
        self.depth_down = DepthDownsample(
            in_channels=2, out_channels=2, target_size=target_size,
        )

        self.sem_conv3d = nn.Sequential(
            nn.Conv3d(sem_embed_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )

        self.depth_conv3d = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )

        self.combine_conv3d = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Conv3d(256, hidden_dim, kernel_size=1),
            nn.BatchNorm3d(hidden_dim), nn.ReLU(inplace=True),
        )

        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sem_labels, cat_depth):
        B, T = sem_labels.shape[:2]

        sem_emb = self.sem_down(sem_labels)
        depth_5d = self.depth_down(cat_depth)

        sem_feat = self.sem_conv3d(sem_emb)
        depth_feat = self.depth_conv3d(depth_5d)

        combined = torch.cat([sem_feat, depth_feat], dim=1)
        combined = self.combine_conv3d(combined)

        C_out = combined.shape[1]
        S = combined.shape[3] * combined.shape[4]
        combined = combined.permute(0, 2, 3, 4, 1).reshape(B, T, S, C_out)

        combined = combined.permute(0, 2, 1, 3).reshape(B * S, T, self.hidden_dim)
        combined, _ = self.temporal_gru(combined)

        combined = combined.reshape(B, S, T, self.hidden_dim)
        combined = combined.permute(0, 2, 1, 3)
        combined = self.out_norm(combined)

        return combined


class PIPNetAlphaV3Final(nn.Module):
    """
    Original 7.66M architecture + FC dropout.
    
    Only change from proven 0.8445 AUC model:
      - Added nn.Dropout(0.5) before fc_out
    """
    def __init__(
        self, bbox_dim=4, pose_dim=34, speed_dim=1,
        branch_out_dim=128, final_dim=256,
        sem_num_classes=20, sem_embed_dim=16,
        dropout_p=0.5, global_target_size=64,
    ):
        super().__init__()
        self.branch_out_dim = branch_out_dim
        self.final_dim = final_dim

        self.kinematic_branch = KinematicBranch(
            bbox_dim=bbox_dim, pose_dim=pose_dim,
            speed_dim=speed_dim, hidden_dim=128,
        )
        self.kin_attn = PIPNetAttention(128, branch_out_dim, dropout_p)
        self.norm_kin = nn.LayerNorm(branch_out_dim)

        self.local_branch = LocalVisualBranch(cnn_dim=512, hidden_dim=256)
        self.local_attn = PIPNetAttention(256, branch_out_dim, dropout_p)
        self.norm_local = nn.LayerNorm(branch_out_dim)

        self.global_branch = GlobalContextBranchConv3D(
            sem_num_classes=sem_num_classes,
            sem_embed_dim=sem_embed_dim,
            hidden_dim=256,
            target_size=global_target_size,
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
        self.aux_global = nn.Linear(256, 1)

        self.fc_drop = nn.Dropout(dropout_p)  # ★ Only new addition
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

        h_kin = self.kinematic_branch(bbox, pose, speed)
        z_kin, alpha_kin = self.kin_attn(h_kin)
        z_kin = self.norm_kin(z_kin)

        h_local = self.local_branch(local_cnn, local_motion)
        z_local, alpha_local = self.local_attn(h_local)
        z_local = self.norm_local(z_local)

        h_global = self.global_branch(sem_labels, cat_depth)
        S = h_global.shape[2]

        h_global_flat = h_global.reshape(B, T * S, 256)
        z_local_expanded = z_local.unsqueeze(1).expand(-1, T * S, -1)
        h_fused = torch.cat([z_local_expanded, h_global_flat], dim=-1)

        z_visual, alpha_visual = self.visual_fuse_attn(h_fused)
        z_visual = self.norm_visual(z_visual)

        z_visual_kin = torch.stack([z_visual, z_kin], dim=1)
        z_final, alpha_final = self.final_attn(z_visual_kin, use_mean_query=True)

        z_final = self.fc_drop(z_final)  # ★ Only new addition
        logit = self.fc_out(z_final)

        if return_aux:
            z_global_last = h_global[:, -1, :, :].mean(dim=1)
            return {
                'logit': logit,
                'aux_kin': self.aux_kin(z_kin),
                'aux_local': self.aux_local(z_local),
                'aux_global': self.aux_global(z_global_last),
                'modality_weights': alpha_final,
                'visual_fuse_weights': alpha_visual,
                'cross_attn_weights': alpha_visual,
                'z_kin': z_kin,
                'z_local': z_local,
                'z_cross': z_visual,
                'z_visual': z_visual,
            }

        return logit


if __name__ == "__main__":
    model = PIPNetAlphaV3Final()
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")

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
    tokens = out['visual_fuse_weights'].shape[1]
    print(f"Tokens: {tokens} {'OK' if tokens == 160 else 'FAIL'}")
    print(f"Logit: {out['logit'].shape}")
