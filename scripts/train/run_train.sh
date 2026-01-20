#!/bin/bash

# ==========================================
# 训练启动脚本 (Training Launcher)
# ==========================================

# 1. 激活 Conda 环境
# 确保你的环境名称是 'CoPE'
source /home/ubuntu/miniconda3/bin/activate CoPE

# 2. 设置环境变量
# 获取脚本所在目录的上一级目录作为项目根目录
# 目录结构: /project/scripts/train -> /project
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

# 将项目根目录添加到 PYTHONPATH，确保能正确 import src 模块
export PYTHONPATH=$PROJECT_ROOT

echo "当前项目根目录: $PROJECT_ROOT"

# 3. 创建输出目录
mkdir -p outputs

# 4. 并发启动训练任务
# 我们同时启动两个任务：RoPE (基线) 和 CoPE (实验组)
# 使用不同的显卡 (GPU 0 和 GPU 1) 并行加速

# === 任务 1: RoPE (Rotary Position Embedding) ===
echo "正在 GPU 0 上启动 RoPE 训练..."
# 设置 CUDA_VISIBLE_DEVICES=0 指定第一个 GPU
# nohup ... & 将任务放入后台运行，避免终端关闭导致任务中断
export CUDA_VISIBLE_DEVICES=0
nohup python scripts/train/train_v2.py --config configs/rope_selective.yaml > outputs/rope_train.log 2>&1 &
ROPE_PID=$! # 获取进程 ID

# === 任务 2: CoPE (Contextual Position Encoding) ===
echo "正在 GPU 1 上启动 CoPE 训练..."
# 设置 CUDA_VISIBLE_DEVICES=1 指定第二个 GPU
export CUDA_VISIBLE_DEVICES=1
nohup python scripts/train/train_v2.py --config configs/cope_selective.yaml > outputs/cope_train.log 2>&1 &
COPE_PID=$! # 获取进程 ID

echo "训练任务已启动！"
echo "RoPE 进程 ID: $ROPE_PID"
echo "CoPE 进程 ID: $COPE_PID"
echo "----------------------------------------"
echo "请使用以下命令查看日志:"
echo "tail -f outputs/rope_train.log outputs/cope_train.log"

# 等待所有后太任务完成
wait $ROPE_PID
wait $COPE_PID

echo "所有训练任务已完成。"
