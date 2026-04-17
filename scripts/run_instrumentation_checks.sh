#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <file-or-dir>"
  exit 1
fi

python .cursor/skills/ascend-operator-instrumentation/scripts/validate_trace_points.py "$1"
