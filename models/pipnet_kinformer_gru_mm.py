"""
PIPNet-Alpha — KinFormer-GRU-MM (Conv3D stems + Smart Segmentation)
====================================================================

Multimodal, kinematic-anchored architecture with V4‑style spatio‑temporal
inductive bias: segmentation, depth, and optical flow are processed with
lightweight Conv3D + MaxPool3d stems before being converted to patch/vector
tokens. The kinematic anchor remains the strong GRU branch. Gated injection
allows the model to start kinematic‑only and progressively open modalities.

Segmentation uses SmartSegmentationPreprocessor with three ablation modes:
- "embed_full":      original behaviour (20-class learned embedding)  -- control
- "embed_reduced":   Solution 1 alone (9 reduced classes, learned embedding)
- "masks":           Solutions 1+2 (6 binary mask channels) -- default

Instance information (Solution 3) has been removed entirely — neither the model
nor the dataset reference dvis_inst anymore.

Progressive protocol
--------------------
1) GRU-anchor sanity check (no --use_* flags)
2) --use_seg --sem_mode embed_full       (control = original behaviour)
3) --use_seg --sem_mode masks            (Solutions 1+2, the fix)
4) --use_depth, --use_local_flow, --use_local_context  (additive)

Learned gates are exposed via model.gate_values().
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.pipnet_alpha_v6_final import (
    KinematicBranch,
    PIPNetAttention,
    DepthDownsample,
)
from models.sem_preprocessor import SmartSegmentationPreprocessor


# ----------------------------------------------------------------------
# Conv3D stems (V4‑style spatio‑temporal)
# ----------------------------------------------------------------------

class _Conv3DPatchStem(nn.Module):
    """Conv3D + MaxPool3d stem for segmentation or depth maps."""
    def __init__(self, in_ch: int, d_model: int, patch_grid: int = 4):
        super().__init__()
        self.patch_grid = patch_grid
        self.d_model = d_model
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, d_model, kernel_size=3, padding=1),
            nn.BatchNorm3d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)
        B, d, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, d, H, W)
        x = nn.functional.adaptive_avg_pool2d(x, (self.patch_grid, self.patch_grid))
        x = x.permute(0, 2, 3, 1).reshape(B * T, self.patch_grid * self.patch_grid, d)
        return x


class _Conv3DFlowStem(nn.Module):
    """Conv3D stem for optical flow (B,2,T,H,W) → (B*T, d_model)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.conv3d = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)
        B, C, T, _, _ = x.shape
        x = x.view(B, T, C)
        x = self.proj(x)
        return x.reshape(B * T, self.d_model)


# ----------------------------------------------------------------------
# Main model
# ----------------------------------------------------------------------

class PIPNetKinFormerGRUMM(nn.Module):
    """
    Kinematic-GRU anchored multimodal Transformer with Conv3D stems for
    segmentation, depth, and optical flow.
    """

    def __init__(
        self,
        bbox_dim: int = 4,
        pose_dim: int = 34,
        speed_dim: int = 1,
        kin_hidden_dim: int = 128,
        branch_out_dim: int = 128,
        dropout_p: float = 0.4,
        # modality switches
        use_local_context: bool = False,
        use_local_flow: bool = False,
        use_seg: bool = False,
        use_depth: bool = False,
        # scene/context settings
        sem_num_classes: int = 20,
        sem_embed_dim: int = 16,
        sem_mode: str = "masks",
        scene_target_size: int = 64,
        scene_patch_grid: int = 4,
        gate_init: float = -2.0,
        gate_l1_weight: float = 0.0,
        # transformer settings
        d_model: int = 128,
        n_heads: int = 2,
        spatial_layers: int = 1,
        ff_dim: int = 256,
        tf_dropout: float = 0.1,
        # behaviour
        run_transformer_without_context: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.use_local_context = bool(use_local_context)
        self.use_local_flow = bool(use_local_flow)
        self.use_seg = bool(use_seg)
        self.use_depth = bool(use_depth)
        self.d_model = int(d_model)
        self.n_scene_patches = int(scene_patch_grid) * int(scene_patch_grid)
        self.gate_l1_weight = float(gate_l1_weight)
        self.run_transformer_without_context = bool(run_transformer_without_context)
        self.sem_mode = sem_mode

        # ------------------------------------------------------------------
        # Strong kinematic anchor: GRU branch.
        # ------------------------------------------------------------------
        self.kinematic_branch = KinematicBranch(
            bbox_dim=bbox_dim,
            pose_dim=pose_dim,
            speed_dim=speed_dim,
            hidden_dim=kin_hidden_dim,
        )
        self.kin_proj = nn.Linear(kin_hidden_dim, d_model)
        self.kin_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.kin_pos, std=0.02)

        # ------------------------------------------------------------------
        # Target-local visual context (VGG features, already temporally pooled)
        # ------------------------------------------------------------------
        if self.use_local_context:
            self.local_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.local_proj = nn.Sequential(
                nn.Linear(512, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.local_pos = nn.Parameter(torch.zeros(1, 1, d_model))
            self.local_gate = nn.Parameter(torch.tensor(float(gate_init)))
            nn.init.trunc_normal_(self.local_pos, std=0.02)

        # ------------------------------------------------------------------
        # Optical flow stem (Conv3D)
        # ------------------------------------------------------------------
        if self.use_local_flow:
            self.flow_stem = _Conv3DFlowStem(d_model=d_model)
            self.flow_pos = nn.Parameter(torch.zeros(1, 1, d_model))
            self.flow_gate = nn.Parameter(torch.tensor(float(gate_init)))
            nn.init.trunc_normal_(self.flow_pos, std=0.02)

        # ------------------------------------------------------------------
        # Segmentation patch tokens via SmartSegmentationPreprocessor.
        # ------------------------------------------------------------------
        if self.use_seg:
            self.sem_preprocessor = SmartSegmentationPreprocessor(
                mode=sem_mode,
                target_size=scene_target_size,
                embed_dim=sem_embed_dim,
            )
            self.seg_stem = _Conv3DPatchStem(
                in_ch=self.sem_preprocessor.out_channels,
                d_model=d_model,
                patch_grid=scene_patch_grid,
            )
            self.seg_pos = nn.Parameter(torch.zeros(1, self.n_scene_patches, d_model))
            self.seg_gate = nn.Parameter(torch.tensor(float(gate_init)))
            nn.init.trunc_normal_(self.seg_pos, std=0.02)

        # ------------------------------------------------------------------
        # Categorical-depth patch tokens (Conv3D stem)
        # ------------------------------------------------------------------
        if self.use_depth:
            self.depth_down = DepthDownsample(
                in_channels=2,
                out_channels=2,
                target_size=scene_target_size,
            )
            self.depth_stem = _Conv3DPatchStem(
                in_ch=2,
                d_model=d_model,
                patch_grid=scene_patch_grid,
            )
            self.depth_pos = nn.Parameter(torch.zeros(1, self.n_scene_patches, d_model))
            self.depth_gate = nn.Parameter(torch.tensor(float(gate_init)))
            nn.init.trunc_normal_(self.depth_pos, std=0.02)

        # ------------------------------------------------------------------
        # Per-frame cross-modal Transformer
        # ------------------------------------------------------------------
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=tf_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=spatial_layers,
            enable_nested_tensor=False,
        )

        # ------------------------------------------------------------------
        # Temporal pooling and classifier
        # ------------------------------------------------------------------
        self.kin_attn = PIPNetAttention(d_model, branch_out_dim, dropout_p)
        self.norm_kin = nn.LayerNorm(branch_out_dim)
        self.fc_drop = nn.Dropout(dropout_p)
        self.fc_out = nn.Linear(branch_out_dim, 1)

    # ----------------------------------------------------------------------
    # Token builders
    # ----------------------------------------------------------------------

    def _build_local_token(self, batch: dict, b: int, t: int) -> torch.Tensor:
        local_cnn = batch["local_cnn"]
        if local_cnn.dim() == 5:
            x = local_cnn.reshape(b * t, *local_cnn.shape[2:])
            x = self.local_pool(x).reshape(b * t, -1)
        elif local_cnn.dim() == 3:
            x = local_cnn.reshape(b * t, -1)
        else:
            raise ValueError(f"Unsupported local_cnn shape: {tuple(local_cnn.shape)}")
        tok = self.local_proj(x).reshape(b * t, 1, self.d_model)
        tok = tok + self.local_pos
        return tok * torch.sigmoid(self.local_gate)

    def _build_flow_token(self, batch: dict, b: int, t: int) -> torch.Tensor:
        flow = batch["local_motion"]
        if flow.dim() == 5:
            x = flow.permute(0, 2, 1, 3, 4).contiguous()
            vec = self.flow_stem(x)
        else:
            raise ValueError(f"local_motion must be (B,T,2,H,W). Got {tuple(flow.shape)}")
        tok = vec.reshape(b * t, 1, self.d_model)
        tok = tok + self.flow_pos
        return tok * torch.sigmoid(self.flow_gate)

    def _build_seg_tokens(self, batch: dict, b: int, t: int) -> torch.Tensor:
        sem = self.sem_preprocessor(batch)        # (B, C, T, 64, 64)
        tok = self.seg_stem(sem)                  # (B*T, P*P, d_model)
        tok = tok + self.seg_pos
        return tok * torch.sigmoid(self.seg_gate)

    def _build_depth_tokens(self, batch: dict, b: int, t: int) -> torch.Tensor:
        dep = self.depth_down(batch["cat_depth"])
        tok = self.depth_stem(dep)
        tok = tok + self.depth_pos
        return tok * torch.sigmoid(self.depth_gate)

    # ----------------------------------------------------------------------

    def forward(self, batch: dict, return_aux: bool = False):
        bbox = batch["bbox"]
        pose = batch["pose"]
        speed = batch["speed"]
        b, t = pose.shape[:2]

        h_kin = self.kinematic_branch(bbox, pose, speed)
        kin_tok = self.kin_proj(h_kin) + self.kin_pos
        kin_tok = kin_tok.reshape(b * t, 1, self.d_model)

        tokens = [kin_tok]
        if self.use_local_context:
            tokens.append(self._build_local_token(batch, b, t))
        if self.use_local_flow:
            tokens.append(self._build_flow_token(batch, b, t))
        if self.use_seg:
            tokens.append(self._build_seg_tokens(batch, b, t))
        if self.use_depth:
            tokens.append(self._build_depth_tokens(batch, b, t))

        if len(tokens) > 1 or self.run_transformer_without_context:
            seq = torch.cat(tokens, dim=1)
            seq = self.spatial_transformer(seq)
            kin_out = seq[:, 0, :].reshape(b, t, self.d_model)
        else:
            kin_out = kin_tok.reshape(b, t, self.d_model)

        z_kin, alpha = self.kin_attn(kin_out)
        z_kin = self.norm_kin(z_kin)
        logit = self.fc_out(self.fc_drop(z_kin))

        if return_aux:
            out = {
                "logit": logit,
                "z_kin": z_kin,
                "kin_attention": alpha,
                "gates": self.gate_values(detach=False),
            }
            if self.gate_l1_weight > 0:
                out["gate_l1"] = self.gate_l1_loss()
            return out
        return logit

    # ----------------------------------------------------------------------

    def _gate_tensor_values(self):
        vals = []
        if self.use_local_context:
            vals.append(torch.sigmoid(self.local_gate))
        if self.use_local_flow:
            vals.append(torch.sigmoid(self.flow_gate))
        if self.use_seg:
            vals.append(torch.sigmoid(self.seg_gate))
        if self.use_depth:
            vals.append(torch.sigmoid(self.depth_gate))
        return vals

    def gate_l1_loss(self) -> torch.Tensor:
        vals = self._gate_tensor_values()
        if not vals:
            return self.fc_out.weight.sum() * 0.0
        return self.gate_l1_weight * torch.stack(vals).sum()

    def gate_values(self, detach: bool = True):
        out = {}
        if self.use_local_context:
            v = torch.sigmoid(self.local_gate)
            out["local_gate"] = float(v.detach().cpu()) if detach else v
        if self.use_local_flow:
            v = torch.sigmoid(self.flow_gate)
            out["flow_gate"] = float(v.detach().cpu()) if detach else v
        if self.use_seg:
            v = torch.sigmoid(self.seg_gate)
            out["seg_gate"] = float(v.detach().cpu()) if detach else v
        if self.use_depth:
            v = torch.sigmoid(self.depth_gate)
            out["depth_gate"] = float(v.detach().cpu()) if detach else v
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    b, t = 2, 10
    batch = {
        "bbox": torch.rand(b, t, 4),
        "pose": torch.rand(b, t, 34),
        "speed": torch.rand(b, t, 1),
        "local_cnn": torch.rand(b, t, 512, 7, 7),
        "local_motion": torch.rand(b, t, 2, 224, 224),
        "sem_labels": torch.randint(0, 20, (b, t, 384, 672)),
        "cat_depth": torch.rand(b, t, 2, 384, 672),
    }

    configs = {
        "anchor only":           dict(),
        "seg embed_full":        dict(use_seg=True, sem_mode="embed_full"),
        "seg embed_reduced":     dict(use_seg=True, sem_mode="embed_reduced"),
        "seg masks":             dict(use_seg=True, sem_mode="masks"),
        "seg+depth masks":       dict(use_seg=True, use_depth=True, sem_mode="masks"),
        "full all modalities":   dict(use_local_context=True, use_local_flow=True,
                                       use_seg=True, use_depth=True, sem_mode="masks"),
    }
    for name, cfg in configs.items():
        model = PIPNetKinFormerGRUMM(**cfg)
        model.eval()
        with torch.no_grad():
            out = model(batch, return_aux=True)
        n = sum(p.numel() for p in model.parameters())
        print(f"{name:24s} params={n:>8,} logit={tuple(out['logit'].shape)} gates={model.gate_values()}")