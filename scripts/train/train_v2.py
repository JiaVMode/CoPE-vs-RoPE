import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

# 将项目根目录添加到系统路径，以便导入 src 模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# 导入自定义模块
from src.utils import load_config, setup_logger, set_seed
from src.data_gen import get_dataloader
from src.model import GPT, ModelConfig

def main():
    # 参数解析：接收配置文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()
    
    # 1. 加载配置与初始化
    cfg = load_config(args.config)
    set_seed(cfg['experiment']['seed']) # 设置随机种子
    
    # 设置日志
    out_dir = cfg['experiment']['out_dir']
    logger = setup_logger(out_dir)
    logger.info(f"实验配置: {cfg}")
    
    device = cfg['train']['device']
    
    # 2. 准备数据加载器
    logger.info("正在加载数据...")
    data_cfg = cfg['data']
    
    # 训练集加载器
    train_loader = get_dataloader(
        batch_size=cfg['train']['batch_size'],
        num_samples=20000, # 训练样本数
        mode=data_cfg['mode'],
        seq_len=data_cfg['seq_len'],
        vocab_size=data_cfg['vocab_size'],
        **data_cfg.get('hard_mode_kwargs', {}) # 传递高难度参数
    )
    
    # 验证集加载器
    val_loader = get_dataloader(
        batch_size=cfg['train']['batch_size'],
        num_samples=1000, # 验证样本数
        mode=data_cfg['mode'],
        seq_len=data_cfg['seq_len'],
        vocab_size=data_cfg['vocab_size'],
        **data_cfg.get('hard_mode_kwargs', {})
    )
    
    # 3. 构建模型
    logger.info("正在构建模型...")
    model_cfg = cfg['model']
    
    # 转换配置格式
    config = ModelConfig(
        vocab_size=data_cfg['vocab_size'] + 500, # 预留一些 Buffer 以防越界
        n_layer=model_cfg['n_layer'],
        n_head=model_cfg['n_head'],
        n_embd=model_cfg['n_embd'],
        block_size=model_cfg['block_size'],
        use_rope=(model_cfg['position_encoding']['type'] == 'rope'), # 判断是否使用 RoPE
        use_cope=(model_cfg['position_encoding']['type'] == 'cope')  # 判断是否使用 CoPE
    )
    
    # 实例化并移动到 GPU
    model = GPT(config).to(device)
    
    # 打印参数量
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count/1e6:.2f}M")

    # 4. 配置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'])
    
    # 5. 开始训练循环
    max_steps = cfg['train']['max_steps']
    iter_data = iter(train_loader)
    
    model.train() # 切换到训练模式
    start_time = time.time()
    
    logger.info("开始训练...")
    
    for step in range(max_steps):
        # 获取下一个 Batch的数据
        try:
            batch = next(iter_data)
        except StopIteration:
            # 如果 Epoch 结束，重新创建一个迭代器
            iter_data = iter(train_loader)
            batch = next(iter_data)
            
        # 将数据移动到设备
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播 (计算 Loss)
        logits, loss = model(input_ids, targets=labels)
        
        # 反向传播与优化
        optimizer.zero_grad()      # 清空梯度
        loss.backward()            # 计算梯度
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()           # 更新权重
        
        # 打印日志
        if step % cfg['train']['log_interval'] == 0:
            dt = time.time() - start_time
            logger.info(f"Step {step} | Loss: {loss.item():.4f} | Time: {dt:.2f}s")
            start_time = time.time()
            
        # 定期评估与保存模型
        if step > 0 and step % cfg['train']['eval_interval'] == 0:
            evaluate(model, val_loader, device, logger, step)
            # 保存最新 Checkpoint
            ckpt_path = os.path.join(out_dir, "ckpt_latest.pt")
            torch.save(model.state_dict(), ckpt_path)
    
    # 训练结束后的最终保存
    torch.save(model.state_dict(), os.path.join(out_dir, "ckpt_final.pt"))
    logger.info("训练完成！")

@torch.no_grad()
def evaluate(model, dataloader, device, logger, step):
    """ 验证集评估函数 """
    model.eval() # 切换到评估模式 (关闭 Dropout 等)
    losses = []
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播 (不保留梯度)
        logits, loss = model(input_ids, targets=labels)
        losses.append(loss.item())
        
        # 计算准确率 (仅针对最后一个 token)
        # logits[:, -1, :] 是预测的最后一个词的概率分布
        pred = torch.argmax(logits[:, -1, :], dim=-1)
        target = labels[:, -1]
        
        correct += (pred == target).sum().item()
        total += target.size(0)
        
    avg_loss = sum(losses) / len(losses)
    acc = correct / total
    
    logger.info(f"VALIDATION | Step {step} | Loss: {avg_loss:.4f} | Acc: {acc:.2%}")
    model.train() # 恢复训练模式

if __name__ == "__main__":
    main()
