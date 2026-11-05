#!/usr/bin/env bash
# 完整检查训练与音频生成的导入闭包；仅在项目内修复缺失依赖。
set -euo pipefail
cd /root/autodl-tmp/IndexTTS25_Minnan
export TMPDIR="$PWD/cache/tmp"
export PIP_CACHE_DIR="$PWD/cache/pip"
export HF_HOME="$PWD/cache/huggingface"
export XDG_CACHE_HOME="$PWD/cache"
export NUMBA_CACHE_DIR="$PWD/cache/numba"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
while [ ! -f logs/setup_complete ]; do sleep 3; done
even/bin/python -m pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r tools/cloud/requirements.txt
even/bin/python -m pip check
PYTHONPATH="$PWD/vendor" even/bin/python -c 'import torch,torchaudio,peft; from indextts.infer_v2_5 import IndexTTS2; assert torch.cuda.is_available(); print("TRAINING_AND_INFERENCE_IMPORTS_OK", torch.__version__, torch.cuda.get_device_name())'
even/bin/python -m pip freeze > logs/environment_freeze.txt
touch logs/runtime_complete
