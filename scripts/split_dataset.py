#!/usr/bin/env python3
"""Split an ImageFolder-style dataset into train/ and test/ directories.

Uses a stratified per-class sklearn split controlled by --test-size and --seed.
Output is written as symlinks by default; pass --copy to duplicate the files.
"""

from __future__ import annotations

import argparse
import os
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


def split(src: Path, test_size: float, seed: int) -> tuple[list[Path], list[Path]]:
    from sklearn.model_selection import train_test_split

    by_class: dict[str, list[Path]] = defaultdict(list)
    for p in collect_images(src):
        by_class[p.parent.name].append(p)

    train_paths: list[Path] = []
    test_paths: list[Path] = []
    for _, paths in sorted(by_class.items()):
        if len(paths) < 2:
            train_paths.extend(paths)
            continue
        n_test = max(1, round(len(paths) * test_size))
        if n_test >= len(paths):
            n_test = len(paths) - 1
        tr, te = train_test_split(paths, test_size=n_test, random_state=seed, shuffle=True)
        train_paths.extend(tr)
        test_paths.extend(te)

    return train_paths, test_paths


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
    parser.add_argument("--dst", required=True, help="Output root; train/ and test/ will be created here")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction reserved for test (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    print(f"Splitting {src} (test_size={args.test_size}, seed={args.seed})")
    train_paths, test_paths = split(src, args.test_size, args.seed)

    n_train = write_split(train_paths, src, dst / "train", args.copy)
    n_test  = write_split(test_paths,  src, dst / "test",  args.copy)

    total = n_train + n_test
    print(f"Train: {n_train} images -> {dst / 'train'}")
    print(f"Test:  {n_test} images  -> {dst / 'test'}")
    print(f"Total: {total}  ({100*n_train/total:.1f}% / {100*n_test/total:.1f}%)")


if __name__ == "__main__":
    main()
