#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <skill_root> <build_dir> <compile_script>"
  echo "example: $0 .cursor/skills/ascend-operator-instrumentation build/cam/comm_operator build/cam/comm_operator/compile_ascend_proj.sh"
  exit 1
fi

SKILL_ROOT="$1"
BUILD_DIR="$2"
COMPILE_SCRIPT="$3"

python "${SKILL_ROOT}/scripts/bootstrap_trace_toolchain.py" --build-dir "${BUILD_DIR}"

python "${SKILL_ROOT}/scripts/patch_build_pipeline.py" \
  --compile-script "${COMPILE_SCRIPT}" \
  --preprocessor-cmd "python \$SCRIPTS_PATH/comm_operator/trace_preprocessor.py \"./\${proj_name}\" \$BUILD_OUT_PATH/ --modify"

python "${SKILL_ROOT}/scripts/verify_trace_scaffold.py" \
  --build-dir "${BUILD_DIR}" \
  --compile-script "${COMPILE_SCRIPT}"

echo "trace scaffold applied successfully"
