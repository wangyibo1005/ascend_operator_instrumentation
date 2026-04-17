#!/usr/bin/env python3
"""
Patch compile/build script to inject trace preprocessor command.
Idempotent by marker comments.
"""

from __future__ import annotations

import argparse
from pathlib import Path


START_MARK = "# TRACE_PREPROCESSOR_HOOK_START"
END_MARK = "# TRACE_PREPROCESSOR_HOOK_END"


def inject_hook(script_text: str, cmd: str) -> str:
    if START_MARK in script_text and END_MARK in script_text:
        return script_text

    anchor = "CopyOps \"./ascend_kernels\" \"./${proj_name}\""
    idx = script_text.find(anchor)
    if idx < 0:
        raise RuntimeError("anchor not found in compile script")

    insert_at = script_text.find("\n", idx)
    if insert_at < 0:
        insert_at = len(script_text)

    hook = (
        f"\n    {START_MARK}\n"
        f"    {cmd}\n"
        f"    {END_MARK}\n"
    )
    return script_text[: insert_at + 1] + hook + script_text[insert_at + 1 :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch compile script with trace preprocessor hook.")
    parser.add_argument("--compile-script", required=True, help="path to compile script, e.g. build/.../compile_ascend_proj.sh")
    parser.add_argument("--preprocessor-cmd", required=True, help="command line to run preprocessor")
    args = parser.parse_args()

    path = Path(args.compile_script)
    text = path.read_text(encoding="utf-8")
    patched = inject_hook(text, args.preprocessor_cmd)
    if patched == text:
        print("no change (hook already exists)")
        return
    path.write_text(patched, encoding="utf-8")
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
