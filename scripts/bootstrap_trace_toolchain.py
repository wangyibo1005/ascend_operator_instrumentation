#!/usr/bin/env python3
"""
Bootstrap minimal trace toolchain files for operator instrumentation projects.
This script is idempotent: existing files are kept unless --force is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TRACE_PREPROCESSOR = """#!/usr/bin/env python3
import argparse
import json
import os
import re

def process_file(path, point_start):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = r'TRACE_POINT\\s*\\(\\s*"([^"]+)"\\s*,\\s*"([^"]+)"\\s*\\)'
    matches = list(re.finditer(pattern, content))
    point_map = {}
    offset = 0
    for i, m in enumerate(matches, start=point_start):
        label, event_type = m.group(1), m.group(2)
        point_map[str(i)] = {"label": label, "event_type": event_type, "file": path}
        s, e = m.start() + offset, m.end() + offset
        content = content[:s] + str(i) + content[e:]
        offset += len(str(i)) - (e - s)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return point_map, point_start + len(matches)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="source directory")
    parser.add_argument("out", help="output directory")
    parser.add_argument("--modify", action="store_true", default=True)
    args = parser.parse_args()
    all_map = {}
    point_id = 1
    for root, _, files in os.walk(args.src):
        for name in files:
            if not name.endswith((".h", ".hpp", ".c", ".cc", ".cpp")):
                continue
            fp = os.path.join(root, name)
            pm, point_id = process_file(fp, point_id)
            all_map.update(pm)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "point_map.json"), "w", encoding="utf-8") as f:
        json.dump(all_map, f, indent=2, ensure_ascii=False)
    print("generated point_map.json")

if __name__ == "__main__":
    main()
"""

TRACE_UTILS = """#!/usr/bin/env python3
import os
import torch

def save_profiling_data(profiling_raw: torch.Tensor, rank_id: int, output_dir: str = "profiling_data"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"rank{rank_id:03d}.pt")
    torch.save(profiling_raw, out_path)
    print(f"Saved: {out_path}")
"""

TRACE_SAVE = """#!/usr/bin/env python3
import argparse
import torch
from trace_utils import save_profiling_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output", default="profiling_data")
    args = parser.parse_args()
    profiling_raw = torch.load(args.input, map_location="cpu")
    save_profiling_data(profiling_raw, args.rank, args.output)

if __name__ == "__main__":
    main()
"""

TRACE_COLLECTOR = """#!/usr/bin/env python3
import argparse
import glob
import json
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("mapping_file")
    parser.add_argument("-o", "--output", default="trace.json")
    args = parser.parse_args()

    with open(args.mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    events = []
    for pt in sorted(glob.glob(os.path.join(args.data_dir, "rank*.pt"))):
        rank_id = int(os.path.basename(pt).split(".")[0][4:])
        tensor = torch.load(pt, map_location="cpu")
        ts = 0.0
        for i, _ in enumerate(tensor.flatten()[:100]):
            point = mapping.get(str((i % max(len(mapping), 1)) + 1), {})
            events.append({
                "name": point.get("label", f"point_{i}"),
                "ph": "I",
                "ts": ts,
                "pid": rank_id,
                "tid": "core0",
                "cat": "trace",
                "args": point
            })
            ts += 1.0
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "us"}, f, indent=2, ensure_ascii=False)
    print(f"generated: {args.output}")

if __name__ == "__main__":
    main()
"""


def write_if_needed(path: Path, content: str, force: bool) -> str:
    if path.exists() and not force:
        return f"skip  {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"write {path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate minimal trace toolchain files.")
    parser.add_argument("--build-dir", required=True, help="module build directory, e.g. build/cam/comm_operator")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    base = Path(args.build_dir)
    actions = [
        write_if_needed(base / "trace_preprocessor.py", TRACE_PREPROCESSOR, args.force),
        write_if_needed(base / "trace_utils.py", TRACE_UTILS, args.force),
        write_if_needed(base / "trace_save.py", TRACE_SAVE, args.force),
        write_if_needed(base / "trace_collector.py", TRACE_COLLECTOR, args.force),
    ]

    for a in actions:
        print(a)


if __name__ == "__main__":
    main()
