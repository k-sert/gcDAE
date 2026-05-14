#!/usr/bin/env python3
"""Create small train/test smoke subsets for image-folder datasets."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".tiff",
}


PRESETS = {
    "cub200": {
        "train": Path("/scratch-shared/scur0164/datasets/cub200_split/train"),
        "test": Path("/scratch-shared/scur0164/datasets/cub200_split/test"),
        "dst": Path("/scratch-shared/scur0164/datasets/cub200_smoke"),
    },
    "caltech256": {
        "train": Path("/scratch-shared/scur0164/datasets/caltech256_split/train"),
        "test": Path("/scratch-shared/scur0164/datasets/caltech256_split/test"),
        "dst": Path("/scratch-shared/scur0164/datasets/caltech256_smoke"),
    },
}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def class_dirs(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a valid directory")
    return {path.name: path for path in sorted(root.iterdir()) if path.is_dir()}


def image_paths(class_root: Path) -> list[Path]:
    return sorted(path for path in class_root.rglob("*") if is_image(path))


def choose_classes(
    train_root: Path,
    test_root: Path,
    num_classes: int,
    seed: int,
) -> list[str]:
    train_classes = class_dirs(train_root)
    test_classes = class_dirs(test_root)
    shared_classes = sorted(set(train_classes) & set(test_classes))
    valid_classes = [
        class_name
        for class_name in shared_classes
        if image_paths(train_classes[class_name]) and image_paths(test_classes[class_name])
    ]

    if len(valid_classes) < num_classes:
        raise ValueError(
            f"Requested {num_classes} classes, but only found "
            f"{len(valid_classes)} classes with images in both train and test."
        )

    rng = random.Random(seed)
    return sorted(rng.sample(valid_classes, num_classes))


def copy_or_link(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def populate_split(
    src_root: Path,
    dst_root: Path,
    selected_classes: list[str],
    images_per_class: int,
    seed: int,
    symlink: bool,
) -> int:
    rng = random.Random(seed)
    total = 0

    for class_name in selected_classes:
        src_class_root = src_root / class_name
        dst_class_root = dst_root / class_name
        candidates = image_paths(src_class_root)
        if len(candidates) < images_per_class:
            raise ValueError(
                f"{src_class_root} has {len(candidates)} images, "
                f"but {images_per_class} were requested."
            )

        for src_path in sorted(rng.sample(candidates, images_per_class)):
            relative_path = src_path.relative_to(src_class_root)
            dst_path = dst_class_root / relative_path
            copy_or_link(src_path, dst_path, symlink)
            total += 1

    return total


def create_smoke_subset(
    train_root: Path,
    test_root: Path,
    dst_root: Path,
    num_classes: int,
    train_images_per_class: int,
    test_images_per_class: int,
    seed: int,
    symlink: bool,
    overwrite: bool,
) -> None:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dst_root} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(dst_root)

    selected_classes = choose_classes(train_root, test_root, num_classes, seed)
    train_count = populate_split(
        train_root,
        dst_root / "train",
        selected_classes,
        train_images_per_class,
        seed,
        symlink,
    )
    test_count = populate_split(
        test_root,
        dst_root / "test",
        selected_classes,
        test_images_per_class,
        seed + 1,
        symlink,
    )

    print(f"Created {dst_root}")
    print(f"Classes: {len(selected_classes)}")
    print(f"Train images: {train_count}")
    print(f"Test images: {test_count}")


def resolve_jobs(args: argparse.Namespace) -> list[tuple[Path, Path, Path]]:
    dataset = args.dataset.lower()
    if dataset == "calltech256":
        dataset = "caltech256"

    if dataset == "custom":
        if args.src_train is None or args.src_test is None or args.dst is None:
            raise ValueError("--src-train, --src-test, and --dst are required for custom")
        return [(args.src_train, args.src_test, args.dst)]

    preset_names = PRESETS.keys() if dataset == "all" else [dataset]
    jobs = []
    for name in preset_names:
        if name not in PRESETS:
            raise ValueError(f"{args.dataset} is not a valid dataset preset")
        preset = PRESETS[name]
        jobs.append((preset["train"], preset["test"], preset["dst"]))
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create split-aware smoke subsets for Stage1 datasets."
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Preset to create: all, cub200, caltech256, calltech256, or custom.",
    )
    parser.add_argument("--src-train", type=Path, default=None)
    parser.add_argument("--src-test", type=Path, default=None)
    parser.add_argument("--dst", type=Path, default=None)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--train-images-per-class", type=int, default=5)
    parser.add_argument("--test-images-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying image files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination smoke dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for train_root, test_root, dst_root in resolve_jobs(args):
        create_smoke_subset(
            train_root=train_root,
            test_root=test_root,
            dst_root=dst_root,
            num_classes=args.classes,
            train_images_per_class=args.train_images_per_class,
            test_images_per_class=args.test_images_per_class,
            seed=args.seed,
            symlink=args.symlink,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
