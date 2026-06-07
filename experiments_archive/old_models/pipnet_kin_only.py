# models/pipnet_kin_only.py
"""
PIPNet-Alpha — Kinematic-Only Diagnostic Model
==============================================

A deliberately minimal model that uses ONLY the kinematic branch
(bbox + pose + speed → GRUs → attention → classifier). No local visual
branch, no global scene branch, no fusion.

PURPOSE
-------
The V5 per-branch test AUCs showed the kinematic auxiliary head alone
(0.871) outranking the full fused model (0.860). This model measures that
directly: if a ~0.25M-parameter kinematic-only network matches the
1.95M / 2.37M multimodal models, then the scene/global context is not
contributing — which is exactly what TrEP, Achaji (2022), and PedGT (2025)
report for pedestrian intention.

DROP-IN USAGE
-------------
This is compatible with train_v6.py with TWO one-line edits:

  1. Replace the import:
        # from models.pipnet_alpha_v6_final import PIPNetAlphaV6Final
        from models.pipnet_kin_only import PIPNetKinOnly

  2. In make_model(), replace the constructor:
        # return PIPNetAlphaV6Final(...).to(device)
        return PIPNetKinOnly(
            dropout_p=args.dropout_p,
            bbox_dim=4, pose_dim=34, speed_dim=1,
        ).to(device)

The model accepts (and ignores) all the V6 global-branch kwargs via **kwargs,
so even leaving the old make_model call mostly intact will work.

The forward returns ONLY {"logit", "z_kin"} when return_aux=True. It returns
no aux_kin / aux_local / aux_global / modality_weights / visual_fuse_weights /
aux_trajectory keys, so MultiTaskLossV6 cleanly reduces to the main focal
(or BCE) classification loss — which is the correct, clean measurement of
"kinematics alone". The training script's branch-AUC logging and entropy /
trajectory terms are all gated on those keys, so they simply switch off.

RECOMMENDED LAUNCH
------------------
Use --traj_weight 0 (no trajectory head exists here) and the same focal /
LR / regularisation settings as your V6 runs so the comparison is fair:

  python train/train_v6.py \
    --jaad_root /Datasets/JAAD_PREP_OUT --pie_root /Datasets/PIE_PREP_OUT \
    --skip_transfer --baseline_epochs 15 --baseline_lr 2e-05 \
    --pie_batch_size 64 --num_workers 2 \
    --loss_type focal --focal_alpha 0.75 --focal_gamma 2.0 \
    --optimizer_mode branch_lr --cnn_gru_lr_mult 1.0 \
    --fusion_lr_mult 1.0 --grad_clip_norm 5.0 \
    --aux_weight 0.0 --entropy_weight 0.0 --traj_weight 0.0 \
    --last_fc_l2 0.01 --dropout_p 0.4 \
    --select_metric pr_auc --amp \
    --save_dir checkpoints_kin_only_seed42

(transformer_lr_mult / global_* flags are accepted and ignored.)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared attention module (identical to V5/V6)
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
# Kinematic branch (identical to V5/V6)
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
# Top-level kinematic-only model
# ─────────────────────────────────────────────────────────────────────────────

class PIPNetKinOnly(nn.Module):
    """
    Kinematic-only classifier.

      bbox, pose, speed
          → KinematicBranch (3 stacked GRUs)            (B, T, 128)
          → PIPNetAttention (temporal pooling)          (B, 128)
          → LayerNorm → Dropout → Linear                (B, 1)

    Accepts and ignores all V6 global/local kwargs via **kwargs so it is a
    one-line swap in make_model().
    """

    def __init__(
        self,
        bbox_dim: int = 4,
        pose_dim: int = 34,
        speed_dim: int = 1,
        branch_out_dim: int = 128,
        dropout_p: float = 0.5,
        **kwargs,   # swallow local_dropout_p, global_*, etc. (ignored)
    ):
        super().__init__()
        self.branch_out_dim = branch_out_dim

        self.kinematic_branch = KinematicBranch(
            bbox_dim=bbox_dim, pose_dim=pose_dim,
            speed_dim=speed_dim, hidden_dim=128,
        )
        self.kin_attn = PIPNetAttention(128, branch_out_dim, dropout_p)
        self.norm_kin = nn.LayerNorm(branch_out_dim)

        self.fc_drop = nn.Dropout(dropout_p)
        self.fc_out  = nn.Linear(branch_out_dim, 1)

    def forward(self, batch: dict, return_aux: bool = False):
        bbox  = batch["bbox"]
        pose  = batch["pose"]
        speed = batch["speed"]

        h_kin = self.kinematic_branch(bbox, pose, speed)   # (B, T, 128)
        z_kin, _ = self.kin_attn(h_kin)                    # (B, 128)
        z_kin = self.norm_kin(z_kin)

        logit = self.fc_out(self.fc_drop(z_kin))           # (B, 1)

        if return_aux:
            # Only the keys we genuinely have. Everything branch/fusion/traj
            # related is intentionally absent so the training loop + loss
            # gracefully reduce to a single-head classifier.
            return {"logit": logit, "z_kin": z_kin}

        return logit


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PIPNet-Alpha — Kinematic-Only Diagnostic Model")
    print("=" * 60)

    model = PIPNetKinOnly(dropout_p=0.4)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}\n")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<20s}: {n:>10,}")

    B, T = 4, 10
    batch = {
        "bbox":  torch.randn(B, T, 4),
        "pose":  torch.randn(B, T, 34),
        "speed": torch.randn(B, T, 1),
        # extra keys are present in real batches but unused here:
        "local_cnn":    torch.randn(B, T, 512, 7, 7),
        "sem_labels":   torch.randint(0, 20, (B, T, 64, 64)),
    }

    with torch.no_grad():
        out = model(batch, return_aux=True)
        logit = model(batch, return_aux=False)

    print("\nOutput keys (return_aux=True):", list(out.keys()))
    print(f"logit shape: {tuple(out['logit'].shape)}  (expected ({B}, 1))")
    assert tuple(out["logit"].shape) == (B, 1)
    assert tuple(logit.shape) == (B, 1)
    print("\nAll checks passed.")