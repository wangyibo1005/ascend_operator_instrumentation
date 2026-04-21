#!/usr/bin/env python3
"""
Bootstrap trace toolchain: copy the real trace_*.py scripts into the target build directory.
Idempotent: existing files are kept unless --force is used.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

TOOLCHAIN_FILES = [
    "trace_preprocessor.py",
    "trace_utils.py",
    "trace_save.py",
    "trace_collector.py",
]


def deploy_file(src: Path, dst: Path, force: bool) -> str:
    if dst.exists() and not force:
        return f"skip  {dst}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return f"write {dst}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy trace toolchain scripts to build directory.")
    parser.add_argument("--build-dir", required=True, help="target directory, e.g. build/cam/comm_operator")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    base = Path(args.build_dir)
    for fname in TOOLCHAIN_FILES:
        src = SCRIPT_DIR / fname
        dst = base / fname
        if not src.exists():
            print(f"WARN  source not found: {src}")
            continue
        print(deploy_file(src, dst, args.force))


if __name__ == "__main__":
    main()
