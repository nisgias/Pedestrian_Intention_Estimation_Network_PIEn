import os
import argparse
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend (prevents crashes on headless servers)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(csv_path, out_dir):
    print(f"Loading data from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find {csv_path}")
        return

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Filter out 'test' splits, we only want to plot 'train' vs 'val'
    df_curves = df[df['split'].isin(['train', 'val'])]
    
    # Get the unique stages (e.g., S1_JAAD, S2_PIE_transfer, S3_PIE_baseline)
    stages = df_curves['stage'].unique()
    
    # Set a clean visual style
    sns.set_theme(style="whitegrid")

    for stage in stages:
        stage_df = df_curves[df_curves['stage'] == stage]
        
        # Create a figure with 2 side-by-side plots (AUC and Loss)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Learning Curves: {stage}', fontsize=18, fontweight='bold')

        # --- PLOT 1: AUC ---
        sns.lineplot(data=stage_df, x='epoch', y='auc', hue='split', marker='o', ax=axes[0], palette=['#1f77b4', '#ff7f0e'])
        axes[0].set_title('Main AUC over Epochs', fontsize=14)
        axes[0].set_ylabel('AUC', fontsize=12)
        axes[0].set_xlabel('Epoch', fontsize=12)
        
        # --- PLOT 2: LOSS ---
        sns.lineplot(data=stage_df, x='epoch', y='loss', hue='split', marker='o', ax=axes[1], palette=['#1f77b4', '#ff7f0e'])
        axes[1].set_title('Total Loss over Epochs', fontsize=14)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_xlabel('Epoch', fontsize=12)

        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(out_dir, f'{stage}_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved plot: {save_path}")

    print("\nAll plots generated successfully! Check your output directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate learning curve plots from training CSV log.")
    parser.add_argument('--csv', type=str, default='checkpoints_sanity/training_log.csv',
                        help='Path to the CSV log file.')
    parser.add_argument('--out_dir', type=str, default='checkpoints_transfer',
                        help='Directory to save the PNG images.')
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    plot_learning_curves(args.csv, args.out_dir)
