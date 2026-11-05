#!/usr/bin/env bash
# 仅在用户指定的真实服务器 tmp 数据盘执行；所有下载、安装和缓存留在项目目录。
set -euo pipefail
cd /root/autodl-tmp/IndexTTS25_Minnan
case "$(pwd -P)" in /root/autodl-tmp/IndexTTS25_Minnan) ;; *) exit 20 ;; esac
mkdir -p cache/tmp cache/pip cache/huggingface logs
export TMPDIR="$PWD/cache/tmp"
export PIP_CACHE_DIR="$PWD/cache/pip"
export HF_HOME="$PWD/cache/huggingface"
export XDG_CACHE_HOME="$PWD/cache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
# GPU PyTorch 使用官方 CUDA 源，其他包优先清华，失败后改用阿里云。
even/bin/python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
if ! even/bin/python -m pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r tools/cloud/requirements.txt; then
    even/bin/python -m pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ -r tools/cloud/requirements.txt
fi
even/bin/python -m pip check
even/bin/python -m pip freeze > logs/environment_freeze.txt
even/bin/python -c 'import torch,torchaudio,transformers,peft; assert torch.cuda.is_available(); print(torch.__version__,torchaudio.__version__,transformers.__version__,peft.__version__,torch.cuda.get_device_name())'
touch logs/setup_complete
