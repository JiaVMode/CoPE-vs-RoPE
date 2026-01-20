import os
import sys
import argparse
import torch
import logging

# 将项目根目录添加到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.utils import load_config, setup_logger, set_seed
from src.data_gen import get_dataloader
from src.model import GPT, ModelConfig

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    单次评估逻辑。
    计算整个 DataLoader 上的平均 Loss 和 Accuracy。
    """
    model.eval()
    losses = []
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        logits, loss = model(input_ids, targets=labels)
        losses.append(loss.item())
        
        # 计算最后一个 token 的准确率
        pred = torch.argmax(logits[:, -1, :], dim=-1)
        target = labels[:, -1]
        correct += (pred == target).sum().item()
        total += target.size(0)
        
    avg_loss = sum(losses) / len(losses)
    acc = correct / total
    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="对应的 YAML 配置文件")
    parser.add_argument("--ckpt_path", type=str, required=True, help="待评估的模型 Checkpoint 路径")
    args = parser.parse_args()
    
    # 1. 加载配置
    cfg = load_config(args.config)
    device = cfg['train']['device']
    
    # 2. 从配置重建模型结构
    model_cfg = cfg['model']
    config = ModelConfig(
        vocab_size=cfg['data']['vocab_size'] + 500,
        n_layer=model_cfg['n_layer'],
        n_head=model_cfg['n_head'],
        n_embd=model_cfg['n_embd'],
        block_size=model_cfg['block_size'],
        use_rope=(model_cfg['position_encoding']['type'] == 'rope'),
        use_cope=(model_cfg['position_encoding']['type'] == 'cope')
    )
    
    # 初始化模型并加载权重
    model = GPT(config).to(device)
    
    if os.path.exists(args.ckpt_path):
        print(f"加载模型权重: {args.ckpt_path}")
        # map_location 确保加载到正确的设备
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    else:
        print(f"错误: 找不到权重文件 {args.ckpt_path}")
        return
    
    print("========================================")
    print(f"评估 Checkpoint: {args.ckpt_path}")
    print("========================================")
    
    # 3. 多长度外推评估
    # 我们评估训练长度 (1024) 和 2倍 (2048) 长度
    base_len = cfg['data']['seq_len']
    lengths = [base_len, base_len * 2]
    
    for length in lengths:
        print(f">> 测试序列长度: {length}")
        try:
            # 动态生成不同长度的测试数据
            dl = get_dataloader(
                batch_size=4,   # 使用小 Batch 避免长序列显存溢出 (OOM)
                num_samples=500,# 样本数量
                mode=cfg['data']['mode'],
                seq_len=length,
                vocab_size=cfg['data']['vocab_size'],
                **cfg['data'].get('hard_mode_kwargs', {})
            )
            # 执行评估
            loss, acc = evaluate(model, dl, device)
            print(f"   [结果] Loss: {loss:.4f} | Acc: {acc:.2%}")
        except Exception as e:
            # 捕获可能的 OOM 或其他运行时错误
            print(f"   [失败] 错误: {e}")

if __name__ == "__main__":
    main()
