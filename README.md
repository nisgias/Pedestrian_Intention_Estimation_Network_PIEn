# Pedestrian Intention Estimation Network - PIPNet V3

This repository contains a PIPNet-based pedestrian crossing intention model.

## Main files

- `models/pipnet_alpha_v3_final.py` — final model architecture
- `train/train_v3_transfer.py` — main training script
- `eval/eval_branch_fusion.py` — branch AUC and fusion evaluation
- `preprocess/` — preprocessing scripts
- `data/pie.py` — dataset loader
- `data/*stats*.json` — normalization statistics

## Training

```bash
python -m train.train_v3_transfer \
  --jaad_root /data/JAAD_PREP_OUT \
  --pie_root /data/PIE_PREP_OUT \
  --amp \
  --skip_transfer \
  --baseline_epochs 40 \
  --baseline_lr 5e-5 \
  --pie_batch_size 10 \
  --aux_weight 0.1 \
  --entropy_weight 0.05 \
  --last_fc_l2 1e-4 \
  --early_stopping_patience 10 \
  --seed 42 \
  --save_dir checkpoints_final_seed42