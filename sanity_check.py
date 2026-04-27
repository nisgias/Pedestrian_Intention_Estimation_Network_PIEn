import os
import torch
import torch.nn as nn
from data.pie import PIESeqDataset
from torch.utils.data import DataLoader
from models.pipnet_alpha_v3_final import PIPNetAlphaV3Final

def run_sanity_check():
    print("========================================")
    print(" PIP-Net V3 Final - ULTIMATE Sanity Check")
    print("========================================")

    # 1. Check Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/6] Hardware Check: Using {device.type.upper()}")
    if device.type == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    # 2. Check Data Paths & JSON Stats
    print("\n[2/6] Data Loading Check (JAAD)...")
    dataset_prefix = "jaad"
    data_root = "/data/JAAD_PREP_OUT"
    speed_stats = f"/workspace/project/data/{dataset_prefix}_speed_stats_splits.json"
    motion_stats = f"/workspace/project/data/{dataset_prefix}_motion_stats_splits.json"
    
    try:
        ds = PIESeqDataset(
            data_root, split="train", mode="train", seq_len=10,
            speed_norm="minmax", speed_stats_path=speed_stats, speed_scope="global",
            motion_norm="p99abs", motion_stats_path=motion_stats, motion_scope="global"
        )
        print(f"      SUCCESS: Loaded dataset. Found {len(ds)} training sequences.")
    except Exception as e:
        print(f"      FAIL: Could not load dataset or JSON stats. Error:\n{e}")
        return

    # 3. Check DataLoader
    print("\n[3/6] Batch Generation Check...")
    try:
        loader = DataLoader(ds, batch_size=2, shuffle=True)
        batch = next(iter(loader))
        print(f"      SUCCESS: Generated batch of size {batch['bbox'].shape[0]}.")
    except Exception as e:
        print(f"      FAIL: DataLoader crashed. Error:\n{e}")
        return

    # 4. Check Data Normalization
    print("\n[4/6] Data Normalization Check...")
    speed_max = batch['speed'].max().item()
    motion_mean = batch['local_motion'].abs().mean().item()
    print(f"      Speed Max (Should be <= 1.0): {speed_max:.4f}")
    print(f"      Motion Abs Mean: {motion_mean:.4f}")
    if speed_max <= 1.1:
        print("      SUCCESS: Normalization stats applied correctly.")
    else:
        print("      WARNING: Speed values seem unnormalized! Check JSON stats.")

    # 5. Check Model Forward Pass
    print("\n[5/6] Model Forward Pass Check...")
    try:
        model = PIPNetAlphaV3Final(dropout_p=0.5).to(device)
        model.train() # Keep in training mode to test gradients
        
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
                
        out = model(batch, return_aux=True)
        tokens = out['visual_fuse_weights'].shape[1]
        
        print(f"      SUCCESS: Forward pass complete!")
        print(f"      Joint Attention Tokens: {tokens} " + ("[(: PERFECT!]" if tokens == 160 else "[✗ ERROR]"))
    except Exception as e:
        print(f"      FAIL: Model forward pass crashed. Error:\n{e}")
        return

    # 6. Check Loss & Backward Pass (Gradient Flow)
    print("\n[6/6] Backward Pass (Gradient Flow) Check...")
    try:
        labels = batch["label"].float().to(device)
        criterion = nn.BCEWithLogitsLoss()
        
        # Calculate main loss
        loss = criterion(out['logit'].squeeze(-1), labels)
        print(f"      Loss computed: {loss.item():.4f}")
        
        # Perform backward pass
        loss.backward()
        
        # Verify gradients populated the Conv3D branch
        has_grad = False
        for name, param in model.global_branch.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
                
        if has_grad:
            print("      SUCCESS: Gradients successfully flowed through Conv3D branch!")
        else:
            print("      FAIL: No gradients found. The computation graph is broken.")
            return
            
    except Exception as e:
        print(f"      FAIL: Backward pass crashed. Error:\n{e}")
        return

    print("\n========================================")
    print(" ALL 6 CHECKS PASSED! YOU ARE READY TO TRAIN. 🚀")
    print("========================================")

if __name__ == "__main__":
    run_sanity_check()
