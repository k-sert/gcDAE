#!/usr/bin/env python3
"""Create a tiny ImageFolder-style NABirds subset for smoke tests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".ppm",
    ".tiff",
}


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="NABirds images root with class subfolders")
    parser.add_argument("--dst", required=True, help="Output subset directory")
    parser.add_argument("--classes", type=int, default=2, help="Number of class folders to include")
    parser.add_argument("--images-per-class", type=int, default=5, help="Images per selected class")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"{src} is not a directory")

    class_dirs = sorted(path for path in src.iterdir() if path.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found under {src}")

    dst.mkdir(parents=True, exist_ok=True)
    written = 0
    for class_dir in class_dirs[: args.classes]:
        out_class = dst / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)
        for image_path in image_files(class_dir)[: args.images_per_class]:
            target = out_class / image_path.name
            if target.exists():
                continue
            if args.copy:
                target.write_bytes(image_path.read_bytes())
            else:
                os.symlink(image_path, target)
            written += 1

    print(f"Created {written} files in {dst}")


if __name__ == "__main__":
    main()
