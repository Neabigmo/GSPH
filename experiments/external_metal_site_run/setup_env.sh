#!/usr/bin/env bash
# Login node only: build a CPU Python 3.10 environment and the tool folder.
# Compiled dependencies come from conda packages or manylinux2014 wheels, which
# need only glibc 2.17; pip is not allowed to build them from source.
set -euo pipefail
export PYTHONNOUSERSITE=1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_DIR=${ENV_DIR:?Set ENV_DIR to a new folder for the Python environment}
TOOLS=${TOOLS:?Set TOOLS to a new folder for Metal3D code and AllMetal3D weights}
CONDA=${CONDA:-$(command -v conda || command -v micromamba || true)}
CONDA_CHANNEL=${CONDA_CHANNEL:-conda-forge}
WHEELS_ONLY=numpy,pandas,scipy,scikit-learn,torch,pydantic-core,orjson,rpds-py,contourpy,kiwisolver,pillow,matplotlib,fonttools,msgpack,markupsafe,websockets,hf-xet,pyyaml,charset-normalizer

if [ -n "$CONDA" ]; then
  "$CONDA" create -y -p "$ENV_DIR" --override-channels -c "$CONDA_CHANNEL" \
    python=3.10 numpy=1.26.4 pandas=2.2.3 scipy=1.12.0 pip
else
  "${BASE_PYTHON:-python3.10}" -m venv "$ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
"$PY" -m pip install --upgrade "pip>=23.3"
"$PY" -m pip install --only-binary=:all: torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install --prefer-binary --only-binary "$WHEELS_ONLY" -c "$HERE/constraints.txt" \
  allmetal3d==0.7.3 tqdm py3Dmol typing_extensions
"$PY" "$HERE/prepare_tools.py" --tools "$TOOLS"
HOME="$TOOLS/home" "$PY" -c "import torch, numpy, pandas, moleculekit, allmetal3d; print('environment ready:', torch.__version__, numpy.__version__, pandas.__version__)"
