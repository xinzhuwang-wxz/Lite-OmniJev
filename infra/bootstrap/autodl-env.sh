# Source from the project root. All large files stay on the AutoDL data disk.
export PATH="/root/miniconda3/bin:$PATH"
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv
export UV_PYTHON_INSTALL_DIR=/root/autodl-tmp/python
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT=https://hf-mirror.com
export UV_CONCURRENT_DOWNLOADS=2
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
