"""
Patch an LDM Stage1 checkpoint to reset EarlyStopping state.
Fixes the case where a checkpoint was saved with a smaller patience value,
which would otherwise be restored and cause premature stopping on resume.

Usage:
    python scripts/patch_early_stopping.py <checkpoint_path> [--patience 30] [--wait_count 0]

Example:
    python scripts/patch_early_stopping.py \
        /scratch-shared/scur0164/checkpoints/Stage1/cub200/.../epoch=000020.ckpt
"""
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--wait_count", type=int, default=0)
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu")

patched = False
for key in list(ckpt.get("callbacks", {}).keys()):
    if "EarlyStopping" in key:
        old_patience = ckpt["callbacks"][key].get("patience")
        old_wait = ckpt["callbacks"][key].get("wait_count")
        ckpt["callbacks"][key]["patience"] = args.patience
        ckpt["callbacks"][key]["wait_count"] = args.wait_count
        print(f"Patched '{key}': patience {old_patience} -> {args.patience}, wait_count {old_wait} -> {args.wait_count}")
        patched = True

if not patched:
    print("No EarlyStopping callback found in checkpoint.")

out_path = args.checkpoint.replace(".ckpt", "_fixed.ckpt")
torch.save(ckpt, out_path)
print(f"Saved patched checkpoint to: {out_path}")
