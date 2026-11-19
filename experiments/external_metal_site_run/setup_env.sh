#!/usr/bin/env bash
# Login node only: build a CPU Python environment and the tool folder.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_PYTHON=${BASE_PYTHON:-python3}
ENV_DIR=${ENV_DIR:?Set ENV_DIR to a new folder for the Python environment}
TOOLS=${TOOLS:?Set TOOLS to a new folder for Metal3D code and AllMetal3D weights}
"$BASE_PYTHON" -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install --upgrade pip
"$ENV_DIR/bin/pip" install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
"$ENV_DIR/bin/pip" install -c "$HERE/constraints.txt" allmetal3d==0.7.3 tqdm py3Dmol
"$ENV_DIR/bin/python" "$HERE/prepare_tools.py" --tools "$TOOLS"
