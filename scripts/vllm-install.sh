#!/bin/bash
# vLLM in its own environment (uv + Python 3.12); PIP_INDEX selects the package index (default PyPI)
set -ex
export PATH=$HOME/.local/bin:$PATH
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
[ -d ~/vllm-env ] || uv venv ~/vllm-env --python 3.12
UV_HTTP_TIMEOUT=300 uv pip install --python ~/vllm-env/bin/python vllm --index-url ${PIP_INDEX:-https://pypi.org/simple}
~/vllm-env/bin/python -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"archs\", torch.cuda.get_arch_list(), \"gpu\", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
echo "=== VLLM INSTALL DONE"
