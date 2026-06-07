#!/usr/bin/env bash
#
# setup_repo.sh — prepare the PIPNet repo for GitHub.
# Run from the repository root:  bash setup_repo.sh
#
# It rescues thesis-relevant code out of experiments_archive/ into models/ and
# train/, then removes stray bytecode. It does NOT delete the archive — review
# and remove that yourself once you're happy.

set -u
moved=0

move() {  # move <src> <dest-dir>
  local src="$1" dst="$2"
  if [ -f "$src" ]; then
    mkdir -p "$dst"
    mv "$src" "$dst/"
    echo "  moved  $src  ->  $dst/"
    moved=$((moved+1))
  else
    echo "  skip   $src  (not found — maybe already moved)"
  fi
}

echo "Rescuing V5 (Spatial-Patch Transformer) — a reported thesis result..."
move experiments_archive/old_models/pipnet_alpha_v5_final.py        models
move experiments_archive/old_train_scripts/train_v5_transfer.py     train

echo
echo "Rescuing PIPN mask-channel ablation (Table 4.19 / slide 51)..."
move experiments_archive/old_models/pipnet_alpha_v4_masks_channels_final.py  models
move experiments_archive/old_train_scripts/train_v4_masks_channels.py        train

# Optional: kinematic-only diagnostic. Uncomment only if a reported number used it
# (per-branch AUCs come from the auxiliary heads, so this is usually not needed).
# echo
# echo "Rescuing kinematic-only diagnostic..."
# move experiments_archive/old_models/pipnet_kin_only.py models

echo
echo "Cleaning stray bytecode..."
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null
echo "  done"

echo
echo "Files rescued: $moved"
echo
echo "Next:"
echo "  1) Quick import smoke-test:"
echo "       python -c 'import models.pipnet_alpha_v5_final'"
echo "       python -c 'import models.pipnet_alpha_v4_masks_channels_final'"
echo "     (V5 trainer already imports 'from models.pipnet_alpha_v5_final import ...',"
echo "      so this should pass. Fix any import path that still points at experiments_archive.)"
echo "  2) Decide what to do with the rest of experiments_archive/ and manydepth/"
echo "     (both are git-ignored by default)."
echo "  3) Commit and push (see README upload steps)."