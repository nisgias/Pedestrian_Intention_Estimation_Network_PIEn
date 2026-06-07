# models/sem_preprocessor.py
"""
Smart Semantic Preprocessor — Solutions 1+2 with precomputed-mask support
===========================================================================

Replaces SemanticDownsample for the segmentation modality.

  Solution 1 — Class reduction
    Collapses 20 PIP classes into 9 reduced ones; noise classes (sky, building,
    vegetation, wall, fence, terrain, bg) are merged into a single "background".

  Solution 2 — Binary mask channels (instead of learned embedding)
    Outputs per-class binary masks: "channel 3 = road", not a 16-dim embedding.

PRECOMPUTED FAST PATH
---------------------
When the batch contains "sem_masks" (uint8 tensor of shape (B,T,6,target,target))
the preprocessor SHORT-CIRCUITS and uses the precomputed masks directly. No
class-remap, no full-resolution masking, no avg-pool. Just normalize uint8 to
float and reshape for the Conv3D stem.

This is the production path for trained models. Use precompute_masks.py to
populate sem_masks in the npz files. The fallback path (sem_labels in batch)
is kept only for backward compatibility / sanity-checking equivalence.

ABLATION MODES
--------------
  mode = "masks"           Solutions 1+2 (default) — 6 channels, uses precomputed
       = "embed_reduced"   Solution 1 only — embed_dim channels (slow runtime path)
       = "embed_full"      Original 20-class learned embedding — embed_dim channels
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# PIP class space (must match prep_scene.py and precompute_masks.py)
# ─────────────────────────────────────────────────────────────────────────────
_PIP_TO_REDUCED = torch.tensor([
    0, 1, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 6, 0, 0, 0, 7, 0, 0, 0
], dtype=torch.long)
_N_REDUCED = 9

_MASK_GROUPS_REDUCED = [
    (1, 2),    # ch 0: pedestrians (person + rider)
    (3,),      # ch 1: vehicles
    (4,),      # ch 2: traffic signals
    (5,),      # ch 3: road
    (6,),      # ch 4: sidewalk
    (7,),      # ch 5: pole
]
_N_MASK_CHANNELS = len(_MASK_GROUPS_REDUCED)   # 6


class SmartSegmentationPreprocessor(nn.Module):
    """
    Input:  batch dict containing one of:
              "sem_masks"  (B, T, 6, target, target) uint8  -- fast path
              "sem_labels" (B, T, H, W) long                -- slow fallback path
    Output: (B, C, T, H_tgt, W_tgt) float — ready for Conv3D stem.
    """

    VALID_MODES = ("masks", "embed_reduced", "embed_full")

    def __init__(
        self,
        mode: str = "masks",
        target_size: int = 64,
        embed_dim: int = 16,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode={mode!r} not in {self.VALID_MODES}")
        self.mode = mode
        self.target_size = int(target_size)

        self.register_buffer("pip_to_reduced", _PIP_TO_REDUCED, persistent=False)

        if mode == "embed_reduced":
            self.embed = nn.Embedding(_N_REDUCED, embed_dim)
            self.out_channels = embed_dim
        elif mode == "embed_full":
            self.embed = nn.Embedding(20, embed_dim)
            self.out_channels = embed_dim
        elif mode == "masks":
            self.out_channels = _N_MASK_CHANNELS    # 6

    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _build_masks_runtime(self, sem_reduced: torch.Tensor) -> torch.Tensor:
        """Fallback: compute masks at full resolution. Slow path."""
        chans = []
        for group in _MASK_GROUPS_REDUCED:
            if len(group) == 1:
                m = (sem_reduced == group[0])
            else:
                m = torch.zeros_like(sem_reduced, dtype=torch.bool)
                for g in group:
                    m = m | (sem_reduced == g)
            chans.append(m.float())
        return torch.stack(chans, dim=2)

    # -------------------------------------------------------------------------

    def forward(self, batch: dict) -> torch.Tensor:
        # ── FAST PATH: precomputed masks already in batch ──
        if self.mode == "masks" and "sem_masks" in batch:
            # sem_masks: (B, T, 6, target, target) uint8 with values in [0, 255]
            masks = batch["sem_masks"]
            if masks.dtype != torch.float32:
                masks = masks.float() / 255.0
            # Reshape (B, T, C, H, W) -> (B, C, T, H, W) for Conv3D stem
            return masks.permute(0, 2, 1, 3, 4).contiguous()

        # ── SLOW PATH: compute on-the-fly from sem_labels ──
        sem = batch["sem_labels"]
        if sem.dtype != torch.long:
            sem = sem.long()
        sem_clamped = sem.clamp(0, 19)

        if self.mode == "embed_full":
            emb = self.embed(sem_clamped)
            x = emb.permute(0, 1, 4, 2, 3).contiguous()
            x = self._spatial_resize_5d(x)
            return x.permute(0, 2, 1, 3, 4).contiguous()

        if self.mode == "embed_reduced":
            sem_red = self.pip_to_reduced[sem_clamped]
            emb = self.embed(sem_red)
            x = emb.permute(0, 1, 4, 2, 3).contiguous()
            x = self._spatial_resize_5d(x)
            return x.permute(0, 2, 1, 3, 4).contiguous()

        # mode == "masks" but no precomputed -> runtime build (legacy)
        sem_red = self.pip_to_reduced[sem_clamped]
        x = self._build_masks_runtime(sem_red)
        x = self._spatial_resize_5d(x)
        return x.permute(0, 2, 1, 3, 4).contiguous()

    # -------------------------------------------------------------------------

    def _spatial_resize_5d(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        if H == self.target_size and W == self.target_size:
            return x
        x = x.reshape(B * T, C, H, W)
        x = F.adaptive_avg_pool2d(x, (self.target_size, self.target_size))
        return x.reshape(B, T, C, self.target_size, self.target_size)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SmartSegmentationPreprocessor — fast path (precomputed) + fallback")
    print("=" * 70)

    B, T, H, W = 2, 10, 384, 672

    # Test FAST path: precomputed masks
    print("\n[FAST PATH] sem_masks in batch (precomputed):")
    sem_masks = (torch.rand(B, T, 6, 64, 64) * 255).byte()
    batch_fast = {"sem_masks": sem_masks}
    pre = SmartSegmentationPreprocessor(mode="masks")
    with torch.no_grad():
        out = pre(batch_fast)
    print(f"  shape={tuple(out.shape)}  values in [0, 1]: {(out.min().item(), out.max().item())}")
    assert out.shape == (B, 6, T, 64, 64)

    # Test SLOW path: only sem_labels in batch
    print("\n[SLOW PATH] sem_labels in batch (runtime compute):")
    sem_labels = torch.randint(0, 20, (B, T, H, W), dtype=torch.long)
    batch_slow = {"sem_labels": sem_labels}
    pre = SmartSegmentationPreprocessor(mode="masks")
    with torch.no_grad():
        out = pre(batch_slow)
    print(f"  shape={tuple(out.shape)}  values in [0, 1]: {(out.min().item(), out.max().item())}")
    assert out.shape == (B, 6, T, 64, 64)

    # Test embedding modes still work (only need sem_labels)
    print("\n[EMBED MODES] sem_labels:")
    for mode in ["embed_full", "embed_reduced"]:
        pre = SmartSegmentationPreprocessor(mode=mode)
        with torch.no_grad():
            out = pre(batch_slow)
        print(f"  mode={mode:15s} out_channels={pre.out_channels} shape={tuple(out.shape)}")

    print("\nAll modes pass.")