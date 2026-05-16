#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch_fidelity import calculate_metrics
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_paths(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


class PairedImageDataset(Dataset):
    def __init__(self, generated_root, reference_root, transform):
        self.generated_root = Path(generated_root)
        self.reference_root = Path(reference_root)
        gen = {str(p.relative_to(self.generated_root)): p for p in image_paths(self.generated_root)}
        ref = {str(p.relative_to(self.reference_root)): p for p in image_paths(self.reference_root)}
        self.keys = sorted(set(gen).intersection(ref))
        self.gen = gen
        self.ref = ref
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        gen = Image.open(self.gen[key]).convert("RGB")
        ref = Image.open(self.ref[key]).convert("RGB")
        return self.transform(gen), self.transform(ref), key


def lpips_score(generated_root, reference_root, batch_size, workers, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = PairedImageDataset(generated_root, reference_root, transform)
    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model = lpips.LPIPS(net="alex").to(device).eval()
    scores = []
    with torch.no_grad():
        for gen, ref, _ in loader:
            values = model(gen.to(device), ref.to(device))
            scores.extend(values.flatten().detach().cpu().numpy().tolist())
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "num_pairs": len(dataset),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, help="Generated image folder")
    parser.add_argument("--reference", required=True, help="Reference/real image folder")
    parser.add_argument("--out", required=True, help="Output metrics JSON path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--kid_subsets", type=int, default=100)
    parser.add_argument("--pr_k", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache_root", default=None, help="torch-fidelity cache directory")
    parser.add_argument("--skip_lpips", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    generated = str(Path(args.generated))
    reference = str(Path(args.reference))
    num_generated = len(image_paths(generated))
    num_reference = len(image_paths(reference))
    if num_generated == 0:
        raise ValueError(f"No generated images found under {generated}")
    if num_reference == 0:
        raise ValueError(f"No reference images found under {reference}")

    fidelity = calculate_metrics(
        input1=generated,
        input2=reference,
        cuda=device.type == "cuda",
        batch_size=args.batch_size,
        fid=True,
        kid=True,
        kid_subset_size=args.kid_subset_size,
        kid_subsets=args.kid_subsets,
        prc=True,
        prc_neighborhood=args.pr_k,
        cache_root=args.cache_root,
        verbose=True,
    )

    metrics = {
        "generated": generated,
        "reference": reference,
        "num_generated": num_generated,
        "num_reference": num_reference,
        "torch_fidelity": {key: float(value) for key, value in fidelity.items()},
    }
    if not args.skip_lpips:
        try:
            metrics["lpips"] = lpips_score(generated, reference, args.batch_size, args.workers, device)
        except Exception as exc:
            metrics["lpips_error"] = repr(exc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
