import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 将项目根目录添加到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.utils import load_config
from src.model import GPT, ModelConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--ckpt_path", type=str, required=True, help="模型 Checkpoint 路径")
    parser.add_argument("--out_path", type=str, default="outputs/attention_map.png", help="可视化图片保存路径")
    parser.add_argument("--title_suffix", type=str, default="", help="图标标题后缀")
    args = parser.parse_args()
    
    # 1. 加载配置
    cfg = load_config(args.config)
    device = cfg['train']['device']
    
    # 2. 构建模型
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
    
    model = GPT(config).to(device)
    
    # 鲁棒加载
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. 构造特定的测试输入 (选择性计数任务)
    # 指令: "Count A and C"
    # 序列: A B C A D
    # 期望: 对 A, C, A 处的权重较高
    # Token IDs 假设 (需与 data_gen 保持一致): 
    # BOS=1, Count=20, A=10, and=21, C=12, Sep=2, B=11, D=13 ...
    input_seq = [1, 20, 10, 21, 12, 2, 10, 11, 12, 10, 13] 
    token_labels = ["BOS", "Count", "A", "and", "C", "SEP", "A", "B", "C", "A", "D"]
    
    x = torch.tensor([input_seq], dtype=torch.long, device=device)
    
    # 4. 前向传播
    with torch.no_grad():
        _, _ = model(x)
    
    # 5. 提取注意力 (基于模型类型)
    last_layer = model.transformer.h[-1]
    
    if config.use_cope:
        # CoPE: 提取 Gates (Sigmoid后的值)
        # model.transformer.h[layer].attn.cope.last_gates
        data = last_layer.attn.cope.last_gates[0] # 取 batch 0 -> [heads, T, T]
        title_prefix = "CoPE Gates"
        cmap = "Blues"
    else:
        # RoPE: 提取 Attention Weights (Softmax后的值)
        # model.transformer.h[layer].attn.last_attn_weights
        if hasattr(last_layer.attn, 'last_attn_weights'):
            data = last_layer.attn.last_attn_weights[0] # 取 batch 0
        else:
            print("错误: 模型未包含 last_attn_weights 属性，请确保已修改 model.py")
            return
        title_prefix = "RoPE Attention Weights"
        cmap = "Reds" 
    
    # 平均所有头
    data_avg = data.mean(dim=0).cpu().numpy() # [T, T]
    
    # 6. 绘图
    plt.figure(figsize=(10, 8))
    
    # 只需要下三角部分 (因果关注)
    # 修正: RoPE attention weights 已经是下三角的 (masked)
    # 但为了美观，我们可以明确 mask 掉上三角
    mask = np.triu(np.ones_like(data_avg, dtype=bool), k=1)
    
    sns.heatmap(data_avg, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=token_labels, yticklabels=token_labels)
    
    plt.title(f"{title_prefix} {args.title_suffix} (Last Layer Avg)", fontsize=16)
    plt.xlabel("Key Token (Attended Position)", fontsize=12)
    plt.ylabel("Query Token (Current Position)", fontsize=12)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path)
    print(f"可视化结果已保存至: {args.out_path}")

if __name__ == "__main__":
    main()
