#!/bin/bash

# Define the 5 seeds to test (42 is your original, plus 4 new ones)
SEEDS=(42 43 44 45 46)

# Loop through each seed
for SEED in "${SEEDS[@]}"; do
  echo "============================================================"
  echo "🚀 Starting V4 Training for Seed: $SEED"
  echo "============================================================"

  python train/train_v4_transfer.py \
    --jaad_root /Datasets/JAAD_PREP_OUT \
    --pie_root /Datasets/PIE_PREP_OUT \
    --skip_transfer \
    --baseline_epochs 15 \
    --baseline_lr 1.9906996673933362e-05 \
    --pie_batch_size 64 \
    --num_workers 2 \
    --aux_weight 0.25 \
    --entropy_weight 0.03 \
    --last_fc_l2 0.0071144760093434225 \
    --visual_l2 0.004 \
    --dropout_p 0.2 \
    --local_dropout_p 0.1 \
    --global_dropout_p 0.4 \
    --amp \
    --seed "$SEED" \
    --save_dir "checkpoints_v4_best_seed${SEED}"

  echo "✅ Finished training for Seed: $SEED"
  echo ""
done

echo "🎉 All 5 seeds completed!"