#!/bin/bash

# ==========================================
# 评估启动脚本 (Evaluation Launcher)
# ==========================================

# 1. 激活环境
source /home/ubuntu/miniconda3/bin/activate CoPE

# 2. 路径设置
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT

# 指定使用的 GPU (评估通常显存占用较小，单卡串行即可，或者分卡并发)
# 这里我们串行执行，首先评估 RoPE，然后评估 CoPE
export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "开始评估 RoPE (基线模型)"
echo "========================================"
# 运行评估脚本，加载 RoPE 的最新权重
python scripts/eval/eval_v2.py \
    --config configs/rope_selective.yaml \
    --ckpt_path outputs/rope_selective/ckpt_latest.pt

echo ""
echo "========================================"
echo "开始评估 CoPE (实验模型)"
echo "========================================"
# 运行评估脚本，加载 CoPE 的最新权重
python scripts/eval/eval_v2.py \
    --config configs/cope_selective.yaml \
    --ckpt_path outputs/cope_selective/ckpt_latest.pt

echo "评估完成！"
