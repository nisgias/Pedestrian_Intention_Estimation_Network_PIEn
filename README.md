<div align="center">

# PIPNet — Pedestrian Crossing-Intention Prediction

**A multimodal, multi-branch deep-learning model that predicts whether a pedestrian is about to cross the road — and how soon — from an autonomous vehicle's camera.**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: PIE](https://img.shields.io/badge/Dataset-PIE%20%2F%20JAAD-blue)](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)

</div>

---

## What is this?

Knowing **where** a pedestrian is isn't enough for a self-driving car — it needs to anticipate **what they intend to do**. A person standing at the kerb might step into the road or stay put, and that decision is invisible to the sensor until it's almost too late.

**PIPN** tackles this as a binary forecasting problem — *will this pedestrian cross in the next 1–2 seconds?* — by fusing three complementary streams of information:

- **How the pedestrian moves** (kinematics)
- **What they look like up close** (local appearance & motion)
- **Where they are in the scene** (global context & geometry)

This repository contains the full, reproducible pipeline — preprocessing, models, training strategies, and evaluation — developed for the diploma thesis *"Evaluation of deep-learning techniques for pedestrian intention and motion prediction from autonomous vehicles."*

---

## Highlights

- **Three-branch architecture** with hierarchical attention-based fusion and per-branch auxiliary heads for interpretability.
- **Modern perception stack** — VGG19 (local appearance), RAFT-large (optical flow), Mask2Former (panoptic segmentation), ManyDepth (categorical depth), YOLOv8-pose (keypoints), plus an **ego-vehicle suppression** step that cleans the scene maps.
- **Honest, multi-horizon evaluation** across five time-to-event horizons (0.5–4 s), not a single cherry-picked number.
- **KinFormer-GRU** — a lightweight variant that matches the full model's mean accuracy with **6.8× fewer parameters**.
- **Reproducible by design** — fixed seeds, leakage-controlled splits, and one evaluation script per reported figure.

---

## Results at a glance

Primary benchmark: **PIE** test set. Headline metric: **ROC-AUC**.

| Setting | ROC-AUC | Notes |
|---|:---:|---|
| **PIPN** — standard test (TTE ≈ 1.5 s), 5 seeds | **0.887 ± 0.011** | mean across seeds 42–46 |
| **PIPN** — peak horizon (TTE ≈ 2.0 s) | **0.908** | within the 1–2 s design range |
| **PIPN** — stable band (TTE 1–4 s) | **≥ 0.87** | robust temporal generalization |
| **KinFormer-GRU** (652K params) | **0.887** | 6.8× fewer params than PIPN (4.42M) |

> **Read with care.** At TTE ≈ 0.5 s — *outside* the model's training range — PIPN reaches 0.883, which is competitive but below some recent high-capacity architectures optimized for that horizon. The model's strength is **stability across longer horizons**, where kinematic and global-context cues complement each other. Cross-dataset transfer to JAAD is intentionally reported as **modest** (≈ 0.635 with combined training), reflecting a real domain shift rather than strong generalization.

PIPN uses **4.42M parameters — about 20% fewer than PIP-Net-α (≈ 5.5M)** — while keeping a full three-branch multimodal design.

---

## Architecture

```
  bbox · pose · ego-speed   ─►  Kinematic Branch   (stacked GRUs)         ─►  z_kin   ─┐
                                                                                       │
  VGG19 crop · RAFT flow    ─►  Local Visual Branch (content + motion GRU) ─► z_local ─┤
                                                                                       ├─►  z_visual ─┐
  Mask2Former · ManyDepth   ─►  Global Context Branch (Conv3D + GRU)       ─► h_global ┘             │
                                                                                                     │
        stage 1 — Visual fusion:  (z_local expanded ⊕ h_global) → attention over T frames → z_visual │
        stage 2 — Modality fusion: attention over [z_visual, z_kin] → dropout → FC ─────────────────►├─► P(cross)
```

- **Kinematic branch** — bounding box (4D) + body pose (34D) + ego-speed (1D) through three stacked GRUs and temporal attention pooling → `z_kin` (128D).
- **Local visual branch** — per-frame **VGG19** crop features (ImageNet-1K) + RAFT optical flow through content and motion GRUs → `z_local` (128D).
- **Global context branch** — Mask2Former semantic segmentation + ManyDepth instance-aware categorical depth through lightweight Conv3D towers and a temporal GRU → `h_global` (256D).
- **Fusion** — a two-stage attention scheme (visual fusion, then modality fusion). Three auxiliary heads supervise the branches and enable per-branch analysis.

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt          # PyTorch ≥ 2.0, torchvision, transformers, ultralytics, scikit-learn, OpenCV ...
# or
docker compose up --build
```

> **ManyDepth** is an external dependency (depth estimation). Clone it separately or add it as a submodule — it is **not** vendored in this repo. YOLOv8-pose weights are downloaded on first run.

### 2. Preprocess (per dataset → `.npz` sequences)

```bash
# PIE
python -m preprocess.pie.prep_kinematics --pie_root /data/PIE --out_root /data/PIE_PREP_OUT
python -m preprocess.pie.prep_local      --pie_root /data/PIE --out_root /data/PIE_PREP_OUT
python -m preprocess.pie.prep_scene      --pie_root /data/PIE --cache_root /data/m2f_cache \
                                         --npz_root /data/PIE_PREP_OUT --out_h 384 --out_w 672

# JAAD  (same three steps under preprocess/jaad/)

# Normalisation statistics (after both datasets are prepared)
python -m preprocess.compute_stats --pie_root /data/PIE_PREP_OUT --jaad_root /data/JAAD_PREP_OUT --out_dir data/
```

**Hard negatives.** Negative windows are not sampled at random. They are drawn from pedestrians with **high PIE human-intention reference scores** that nonetheless do *not* cross — visually crossing-like cases that force the model to rely on subtle cues rather than mere proximity to the road.

### 3. Train

The reported configurations follow **Table 4.1** of the thesis (AdamW, AMP, `ReduceLROnPlateau`, observation window 10 frames / stride 2, TTE ∈ [1.0, 2.0] s). Seeds 42–46 were used; the representative PIE-only run is seed 46.

```bash
# Transfer (JAAD → PIE) + PIE-only baseline
PYTHONPATH=. python -m train.train_v4_transfer \
  --jaad_root /data/JAAD_PREP_OUT --pie_root /data/PIE_PREP_OUT --amp \
  --jaad_epochs 40 --jaad_lr 1.47e-5 --jaad_batch_size 10 \
  --pie_epochs  30 --pie_lr  2.08e-5 --pie_batch_size 64 \
  --aux_weight 0.20 --entropy_weight 0.08 --last_fc_l2 3.24e-3 \
  --early_stopping_patience 6 --seed 42 --save_dir checkpoints/transfer_seed42

# PIE-only baseline (representative run)
PYTHONPATH=. python -m train.train_v4_transfer \
  --pie_root /data/PIE_PREP_OUT --skip_transfer --amp \
  --baseline_epochs 15 --baseline_lr 1.99e-5 --pie_batch_size 64 \
  --aux_weight 0.25 --entropy_weight 0.03 --last_fc_l2 7.11e-3 \
  --seed 46 --save_dir checkpoints/baseline_seed46

# Combine (JAAD + PIE joint training; validated on PIE, tested on PIE and JAAD)
PYTHONPATH=. python -m train.train_v4_combine \
  --jaad_root /data/JAAD_PREP_OUT --pie_root /data/PIE_PREP_OUT --amp \
  --pie_epochs 30 --pie_lr 2.08e-5 --pie_batch_size 64 \
  --aux_weight 0.20 --entropy_weight 0.08 --seed 42 --save_dir checkpoints/combine_seed42
```

### 4. Evaluate

```bash
python -m eval.eval_all_horizons          # full metric sweep, TTE 0.5–4 s
python -m eval.eval_per_branch_horizons   # per-branch AUC / F1 per horizon
python -m eval.eval_multiseed_v4          # multi-seed reliability, ECE, jitter, error correlation
python -m eval.eval_compare_strategies    # transfer vs combine vs baseline (PIE test)
python -m eval.eval_roc_horizons          # ROC curves per horizon
```

---

## Training strategies

| Strategy | Description | Script |
|---|---|---|
| **PIE-only** | Train from scratch on PIE only (main benchmark). | `train/train_v4_transfer.py --skip_transfer` |
| **Transfer** | Pretrain on JAAD, fine-tune on PIE. Best generalization. | `train/train_v4_transfer.py` |
| **Combine** | Joint JAAD + PIE training; tested on PIE *and* JAAD. JAAD ego-speed is zero-filled (unavailable). | `train/train_v4_combine.py` |

---

## Project structure

```
pie_intention/
├── models/
│   ├── pipnet_alpha_v4_final.py       # PIPN — main model, Conv3D + GRU global branch        (thesis §3.3)
│   ├── pipnet_alpha_v5_final.py       # Spatial-Patch Transformer variant                    (thesis §3.4.1)
│   ├── pipnet_alpha_v6_final.py       # Factorized Space-Time variant                        (thesis §3.4.2)
│   ├── pipnet_kinformer_gru_mm.py     # KinFormer-GRU — kinematic anchor + gated tokens      (thesis §3.4.3)
│   └── sem_preprocessor.py            # 20→9 class reduction / 6 binary mask channels
├── train/
│   ├── train_v4_transfer.py           # Transfer (JAAD→PIE) + PIE-only baseline
│   ├── train_v4_combine.py            # Combine (JAAD + PIE)
│   ├── train_v5_transfer.py           # Spatial-Patch Transformer training
│   ├── train_v6.py                    # Factorized variant (focal loss, trajectory aux head)
│   └── train_kinformer_gru_mm.py      # KinFormer progressive gated training
├── eval/                              # per-horizon, per-branch, multi-seed, strategy & ROC evaluation
├── tools/                             # live demo, modality & qualitative visualisation, mask precompute
├── preprocess/
│   ├── pie/   (prep_kinematics, prep_local, prep_scene)
│   ├── jaad/  (prep_kinematics, prep_local, prep_scene)
│   └── compute_stats.py
├── data/                              # dataset loader (pie.py) + per-split normalisation stats
├── requirements.txt · Dockerfile · docker-compose.yml
```

---

## Reproducing the thesis figures

| Figure / Table | Script |
|---|---|
| Multi-horizon metrics (ROC-AUC, F1, Acc) | `eval/eval_all_horizons.py` |
| Per-branch contribution across TTE | `eval/eval_per_branch_horizons.py`, `tools/plot_branch_horizons.py` |
| Branch-zeroing ablations | `eval/eval_ablation_horizons.py` |
| Seed stability, error correlation, jitter | `eval/eval_multiseed_v4.py` |
| Post-hoc logit fusion | `eval/eval_posthoc_fusion_horizons.py` |
| ROC curves per horizon | `eval/eval_roc_horizons.py` |
| Cross-dataset (PIE / JAAD) | `eval/eval_compare_strategies.py`, `eval/eval_compare_strategies_jaad.py` |
| KinFormer horizon sweep | `tools/eval_kinformer_horizons.py` |
| Qualitative TP/TN/FP/FN grid | `tools/viz_qualitative_grid.py` |

---

## Citation

```bibtex
@thesis{tsitsirigkos_pipn,
  author      = {Tsitsirigkos, Ioannis},
  title       = {Evaluation of Deep-Learning Techniques for Pedestrian Intention
                 and Motion Prediction from Autonomous Vehicles},
  school      = {Democritus University of Thrace, Dept. of Electrical and Computer Engineering},
  type        = {Diploma Thesis},
  year        = {2026},
  note        = {Supervisor: Ilias Theodorakopoulos}
}
```

---

## Acknowledgements

Built on the shoulders of excellent open work: the **PIE** and **JAAD** datasets (York University), **PIP-Net** as the architectural reference, and the perception stack — **Mask2Former**, **ManyDepth**, **RAFT**, and **Ultralytics YOLOv8-pose**.

## License

Released under the [MIT License](LICENSE). Third-party components (datasets, ManyDepth, perception models) remain under their respective licenses.