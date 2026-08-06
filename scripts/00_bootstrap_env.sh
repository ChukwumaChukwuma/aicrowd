#!/usr/bin/env bash
# Rebuild the numeric stack from Git sources only.
#
# Context: the sandbox this work was done in permits outbound HTTPS to
# github.com and nothing else — pypi.org, files.pythonhosted.org,
# huggingface.co and archive.ubuntu.com all answer 403 at the egress gateway —
# and shipped no NumPy. Everything below is therefore built from source.
#
# If you have a working PyPI, you do not need this file: `uv sync` in the
# official starter kit gives you the same flopscope/NumPy pair.
#
#   Usage:  bash scripts/00_bootstrap_env.sh /path/to/workdir
#   Then:   . /path/to/workdir/prefix/runenv.sh
set -euo pipefail

WORK="${1:?usage: 00_bootstrap_env.sh <workdir>}"
B="$WORK/build"; P="$WORK/prefix"
mkdir -p "$B" "$P/bin"

clone() { [ -d "$B/$2" ] || git clone --depth 1 "$1" "$B/$2"; }

# --- pure-Python build tooling (no compiled deps) --------------------------
clone https://github.com/cython/cython.git                cython
clone https://github.com/mesonbuild/meson.git             meson
clone https://github.com/mesonbuild/meson-python.git      meson-python
clone https://github.com/pypa/pyproject-metadata.git      pyproject-metadata
clone https://github.com/pypa/packaging.git               packaging
# --- runtime deps of flopscope (all pure Python) ---------------------------
clone https://github.com/dgasmith/opt_einsum.git          opt_einsum
clone https://github.com/Textualize/rich.git              rich
clone https://github.com/pygments/pygments.git            pygments
clone https://github.com/executablebooks/markdown-it-py.git markdown-it-py
clone https://github.com/executablebooks/mdurl.git        mdurl
# --- the challenge packages ------------------------------------------------
clone https://github.com/AIcrowd/flopscope.git            flopscope
clone https://github.com/AIcrowd/whestbench.git           whestbench
clone https://github.com/AIcrowd/whest-starterkit.git     whest-starterkit

# --- OpenBLAS --------------------------------------------------------------
# NOFORTRAN=1 still yields a full LAPACK via OpenBLAS's C_LAPACK path, which
# matters because this image has gcc/g++ but no gfortran.
if [ ! -d "$B/OpenBLAS" ]; then
  git clone --depth 1 https://github.com/OpenMathLib/OpenBLAS.git "$B/OpenBLAS"
fi
if [ ! -f "$P/lib/libopenblas.so" ]; then
  make -C "$B/OpenBLAS" -j"$(nproc)" NOFORTRAN=1 USE_THREAD=1 NUM_THREADS=8 \
       BUILD_LAPACK_DEPRECATED=0
  make -C "$B/OpenBLAS" PREFIX="$P" NOFORTRAN=1 USE_THREAD=1 NUM_THREADS=8 install
fi

# --- Python venv + build shims --------------------------------------------
[ -d "$P/venv" ] || /usr/bin/python3.12 -m venv "$P/venv"
printf '#!/bin/sh\nexec %s/venv/bin/python %s/meson/meson.py "$@"\n'  "$P" "$B" > "$P/bin/meson"
printf '#!/bin/sh\nexec %s/venv/bin/python %s/cython/cython.py "$@"\n' "$P" "$B" > "$P/bin/cython"
chmod +x "$P/bin/meson" "$P/bin/cython"

# --- NumPy (pinned <2.5 because flopscope 0.10 requires numpy>=2.0,<2.5) ---
if ! "$P/venv/bin/python" -c "import numpy" 2>/dev/null; then
  git -C "$B/numpy" rev-parse HEAD >/dev/null 2>&1 || \
    git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/numpy/numpy.git "$B/numpy"
  git -C "$B/numpy" fetch --depth 1 origin tag v2.3.5
  git -C "$B/numpy" checkout v2.3.5
  git -C "$B/numpy" submodule update --init --recursive --depth 1
  PATH="$P/bin:$PATH" \
  PYTHONPATH="$B/cython:$B/meson:$B/meson-python:$B/pyproject-metadata:$B/packaging/src" \
  PKG_CONFIG_PATH="$P/lib/pkgconfig" LD_LIBRARY_PATH="$P/lib" \
  "$P/venv/bin/python" -m pip install --no-build-isolation --no-deps \
      -Csetup-args=-Dblas=openblas -Csetup-args=-Dlapack=openblas \
      -Csetup-args=-Dallow-noblas=true -Csetup-args=-Dcpu-baseline=avx2 \
      -Ccompile-args=-j"$(nproc)" "$B/numpy"
fi

# --- runtime environment ---------------------------------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cat > "$P/runenv.sh" <<EOF
export LD_LIBRARY_PATH=$P/lib
export PYTHONPATH=$B/flopscope/src:$B/opt_einsum:$B/rich:$B/pygments:$B/markdown-it-py:$B/mdurl:$REPO
export PY=$P/venv/bin/python
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
EOF

# shellcheck disable=SC1090
. "$P/runenv.sh"
"$PY" -c "import numpy, flopscope; print('numpy', numpy.__version__, '| flopscope', flopscope.__version__)"
echo "bootstrap OK — source $P/runenv.sh"
