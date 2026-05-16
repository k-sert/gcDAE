#!/usr/bin/env python3
"""Split an ImageFolder-style dataset into train/val/test directories.

Uses a stratified per-class split controlled by --train-size, --val-size,
--test-size, and --seed.
Output is written as symlinks by default; pass --copy to duplicate the files.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tiff"}


def collect_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )


def place(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def split(
    src: Path,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for p in collect_images(src):
        by_class[p.parent.name].append(p)

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    test_paths: list[Path] = []

    rng = random.Random(seed)
    for _, paths in sorted(by_class.items()):
        paths = list(paths)
        rng.shuffle(paths)

        if len(paths) < 3:
            train_paths.extend(paths)
            continue

        n_train = round(len(paths) * train_size)
        n_val = max(1, round(len(paths) * val_size))
        n_test = max(1, round(len(paths) * test_size))

        # Keep at least one training example for every class when possible.
        while n_train + n_val + n_test > len(paths):
            if n_train > 1 and n_train >= n_val and n_train >= n_test:
                n_train -= 1
            elif n_test > 1 and n_test >= n_val:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        while n_train + n_val + n_test < len(paths):
            n_train += 1

        train_end = n_train
        val_end = train_end + n_val
        train_paths.extend(paths[:train_end])
        val_paths.extend(paths[train_end:val_end])
        test_paths.extend(paths[val_end:val_end + n_test])

    return train_paths, val_paths, test_paths


def validate_sizes(train_size: float, val_size: float, test_size: float) -> None:
    sizes = {
        "train-size": train_size,
        "val-size": val_size,
        "test-size": test_size,
    }
    for name, value in sizes.items():
        if value <= 0 or value >= 1:
            raise ValueError(f"--{name} must be between 0 and 1, got {value}")

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "--train-size, --val-size, and --test-size must sum to 1.0; "
            f"got {total:.6f}"
        )


def prepare_dst(dst: Path, overwrite: bool) -> None:
    for split_name in ("train", "val", "test"):
        split_dir = dst / split_name
        if split_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{split_dir} already exists. Pass --overwrite to recreate splits."
                )
            shutil.rmtree(split_dir)


def write_split(paths: list[Path], src_root: Path, dst_root: Path, copy: bool) -> int:
    written = 0
    for src_path in paths:
        try:
            rel = src_path.relative_to(src_root)
        except ValueError:
            rel = Path(src_path.parent.name) / src_path.name
        place(src_path, dst_root / rel, copy)
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Dataset root containing class subfolders")
    parser.add_argument("--dst", required=True, help="Output root; train/, val/, and test/ will be created here")
    parser.add_argument("--train-size", type=float, default=0.7,
                        help="Fraction reserved for train (default: 0.7)")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Fraction reserved for validation (default: 0.1)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction reserved for test (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove existing train/val/test directories before writing")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    validate_sizes(args.train_size, args.val_size, args.test_size)
    prepare_dst(dst, args.overwrite)

    print(
        f"Splitting {src} "
        f"(train_size={args.train_size}, val_size={args.val_size}, "
        f"test_size={args.test_size}, seed={args.seed})"
    )
    train_paths, val_paths, test_paths = split(
        src,
        args.train_size,
        args.val_size,
        args.test_size,
        args.seed,
    )

    n_train = write_split(train_paths, src, dst / "train", args.copy)
    n_val = write_split(val_paths, src, dst / "val", args.copy)
    n_test = write_split(test_paths, src, dst / "test", args.copy)

    total = n_train + n_val + n_test
    print(f"Train: {n_train} images -> {dst / 'train'}")
    print(f"Val:   {n_val} images -> {dst / 'val'}")
    print(f"Test:  {n_test} images -> {dst / 'test'}")
    print(
        f"Total: {total}  "
        f"({100*n_train/total:.1f}% / {100*n_val/total:.1f}% / {100*n_test/total:.1f}%)"
    )


if __name__ == "__main__":
    main()
