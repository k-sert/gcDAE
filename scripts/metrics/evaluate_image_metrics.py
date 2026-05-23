#!/usr/bin/env python
import argparse
import itertools
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch_fidelity import calculate_metrics
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HYPDIFFUSION_PREFIX_RE = re.compile(r"^\d+_(.+)$")


def image_paths(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def prepare_metric_images(root, prepared_root, size):
    root = Path(root)
    prepared_root = Path(prepared_root)
    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)
    for index, src in enumerate(image_paths(root)):
        rel = src.relative_to(root)
        safe_name = "__".join(rel.parts)
        dst = prepared_root / f"{index:06d}_{Path(safe_name).stem}.png"
        with Image.open(src) as image:
            image = image.convert("RGB").resize((size, size), Image.BICUBIC)
            image.save(dst)
    return str(prepared_root)


def strip_hypdiffusion_prefix(relative_path):
    path = Path(relative_path)
    match = HYPDIFFUSION_PREFIX_RE.match(path.name)
    if not match:
        return str(path)
    return str(path.with_name(match.group(1)))


class PairedImageDataset(Dataset):
    def __init__(self, generated_root, reference_root, transform):
        self.generated_root = Path(generated_root)
        self.reference_root = Path(reference_root)
        ref = {str(p.relative_to(self.reference_root)): p for p in image_paths(self.reference_root)}
        self.pairs = []
        for gen_path in image_paths(self.generated_root):
            gen_key = str(gen_path.relative_to(self.generated_root))
            ref_key = gen_key if gen_key in ref else strip_hypdiffusion_prefix(gen_key)
            if ref_key in ref:
                self.pairs.append((gen_key, gen_path, ref[ref_key]))
        self.ref = ref
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        key, gen_path, ref_path = self.pairs[index]
        gen = Image.open(gen_path).convert("RGB")
        ref = Image.open(ref_path).convert("RGB")
        return self.transform(gen), self.transform(ref), key


class ImagePairDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        left_path, right_path, key = self.pairs[index]
        left = Image.open(left_path).convert("RGB")
        right = Image.open(right_path).convert("RGB")
        return self.transform(left), self.transform(right), key


def lpips_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def run_lpips_dataset(dataset, batch_size, workers, device):
    import lpips

    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model = lpips.LPIPS(net="alex").to(device).eval()
    scores = []
    with torch.no_grad():
        for left, right, _ in loader:
            values = model(left.to(device), right.to(device))
            scores.extend(values.flatten().detach().cpu().numpy().tolist())
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "num_pairs": len(dataset),
    }


def paired_lpips_score(generated_root, reference_root, batch_size, workers, device):
    dataset = PairedImageDataset(generated_root, reference_root, lpips_transform())
    result = run_lpips_dataset(dataset, batch_size, workers, device)
    if result is not None:
        result["mode"] = "paired"
    return result


def diversity_lpips_score(generated_root, batch_size, workers, device, max_pairs_per_class, seed):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    generated_root = Path(generated_root)
    groups = {}
    for path in image_paths(generated_root):
        relative = path.relative_to(generated_root)
        class_name = relative.parts[0] if len(relative.parts) > 1 else "__root__"
        groups.setdefault(class_name, []).append(path)

    rng = random.Random(seed)
    pairs = []
    per_class = {}
    for class_name, paths in sorted(groups.items()):
        paths = sorted(paths)
        all_pairs = list(itertools.combinations(paths, 2))
        if max_pairs_per_class is not None and len(all_pairs) > max_pairs_per_class:
            all_pairs = rng.sample(all_pairs, max_pairs_per_class)
        for left, right in all_pairs:
            key = f"{class_name}/{left.name}::{right.name}"
            pairs.append((left, right, key))
        per_class[class_name] = {
            "num_images": len(paths),
            "num_pairs": len(all_pairs),
        }

    dataset = ImagePairDataset(pairs, transform)
    result = run_lpips_dataset(dataset, batch_size, workers, device)
    if result is not None:
        result["mode"] = "diversity"
        result["num_classes"] = len(groups)
        result["max_pairs_per_class"] = max_pairs_per_class
        result["seed"] = seed
        result["per_class"] = per_class
    return result


def lpips_score(generated_root, reference_root, mode, batch_size, workers, device, max_pairs_per_class, seed):
    if mode == "paired":
        return paired_lpips_score(generated_root, reference_root, batch_size, workers, device)
    if mode == "diversity":
        return diversity_lpips_score(generated_root, batch_size, workers, device, max_pairs_per_class, seed)
    if mode == "both":
        return {
            "paired": paired_lpips_score(generated_root, reference_root, batch_size, workers, device),
            "diversity": diversity_lpips_score(generated_root, batch_size, workers, device, max_pairs_per_class, seed),
        }
    raise ValueError(f"Unsupported LPIPS mode: {mode}")


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
    parser.add_argument("--lpips_mode", choices=["paired", "diversity", "both"], default="paired")
    parser.add_argument("--lpips_max_pairs_per_class", type=int, default=1000)
    parser.add_argument("--lpips_seed", type=int, default=42)
    parser.add_argument("--metric_image_size", type=int, default=299)
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
    kid_subset_size = min(args.kid_subset_size, num_generated, num_reference)
    prepared_base = Path(args.out).parent / "_prepared" / Path(args.out).stem
    prepared_generated = prepare_metric_images(generated, prepared_base / "generated", args.metric_image_size)
    prepared_reference = prepare_metric_images(reference, prepared_base / "reference", args.metric_image_size)

    fidelity = calculate_metrics(
        input1=prepared_generated,
        input2=prepared_reference,
        cuda=device.type == "cuda",
        batch_size=args.batch_size,
        fid=True,
        kid=True,
        kid_subset_size=kid_subset_size,
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
            metrics["lpips"] = lpips_score(
                generated,
                reference,
                args.lpips_mode,
                args.batch_size,
                args.workers,
                device,
                args.lpips_max_pairs_per_class,
                args.lpips_seed,
            )
        except Exception as exc:
            metrics["lpips_error"] = repr(exc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
