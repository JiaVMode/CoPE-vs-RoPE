import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .cope import CoPE

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization) 层。
    相比 LayerNorm，去除了均值项，计算更简单且效果相当。
    """
    def __init__(self, dim, eps=1e-6):
        """
        初始化 RMSNorm。
        
        参数:
            dim (int): 输入的维度。
            eps (float): 防止除零的极小值。
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 (gain)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """ 计算 RMS 归一化 """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """ 前向传播 """
        # 为了数值稳定性，通常在 float32 下计算 norm
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) 旋转位置编码。
    目前大模型的标配位置编码方式。
    """
    def __init__(self, dim, max_seq_len=2048):
        """
        初始化 RoPE。
        
        参数:
            dim (int): 头的维度 (head_dim)。
            max_seq_len (int): 预计算频率矩阵的最大长度。
        """
        super().__init__()
        # 计算旋转频率 theta_i = 10000^(-2i/d)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def forward(self, x, seq_len=None):
        """
        生成对应长度的 cos 和 sin 值。
        """
        if seq_len is None:
            seq_len = x.shape[1]
            
        # 生成位置索引 t = [0, 1, ..., seq_len-1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        # 计算 m * theta
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # 拼接以适配复数旋转 (这里为了适配实数实现的 rotate_half)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    """
    辅助函数：将向量的一半旋转 90 度。
    [-x2, x1]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    """
    应用 RoPE 到查询向量 Q 或 键向量 K 上。
    """
    # 调整 cos/sin 形状以广播: [1, 1, SeqLen, HeadDim]
    cos = cos.view(1, 1, -1, x.shape[-1])
    sin = sin.view(1, 1, -1, x.shape[-1])
    # 标准旋转公式
    return (x * cos) + (rotate_half(x) * sin)

class CausalSelfAttention(nn.Module):
    """
    因果自注意力层 (Causal Self-Attention)。
    支持切换 RoPE 或 CoPE。
    """
    def __init__(self, config):
        """
        初始化注意力层。
        
        参数:
            config: 模型配置对象。
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        # 线性投影层：一次性计算 Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # 读取配置中的标志位
        self.use_rope = getattr(config, 'use_rope', True)
        self.use_cope = getattr(config, 'use_cope', False)
        
        # 互斥逻辑：如果启用 CoPE，则强制关闭 RoPE
        if self.use_cope:
            self.use_rope = False 
            
        # 初始化 RoPE 模块
        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size)
            
        # 初始化 CoPE 模块
        if self.use_cope:
            # CoPE 需要知道最大相对位置范围，这里使用 block_size
            self.cope = CoPE(config.block_size, self.head_dim)

    def forward(self, x, mask=None):
        """ 前向传播 """
        B, T, C = x.size() # Batch, Time(SeqLen), Channels

        # 1. 投影并切分 Q, K, V
        q, k, v = self.c_attn(x).split(C, dim=2)
        
        # 2. 调整形状为多头: [Batch, Heads, SeqLen, HeadDim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) 
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) 
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) 

        # 3. 如果启用 RoPE，对 Q, K 应用位置编码
        if self.use_rope:
            cos, sin = self.rope(x, seq_len=T)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # 4. 计算注意力分数 Logits (Scaled Dot-Product)
        attn_logits = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 5. 应用掩码 (Mask)
        if mask is not None:
             # 如果提供了外部 mask
             attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        else:
            # 默认使用因果掩码 (Causal Mask)：只看过去，不看未来
            bias = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            attn_logits = attn_logits.masked_fill(bias == 0, float('-inf'))
            
        # 6. 如果启用 CoPE，注入位置偏置
        if self.use_cope:
            # CoPE 接收 Query 和 原始 Logits，返回位置偏置 Bias
            # 注意：这里的 attn_logits 已经包含了 Mask (-inf)，
            # CoPE 内部的 Sigmoid(-inf) 会得到 0，正确地忽略被 Mask 掉的 token
            attn_logits = attn_logits + self.cope(q, attn_logits)
        
        # 7. Softmax 归一化
        att = F.softmax(attn_logits, dim=-1)
        
        # 缓存注意力权重以便可视化 (Phase 10)
        self.last_attn_weights = att.detach()

        
        # 8. 这里不做 Dropout (为了简化实验)
        
        # 9. 加权求和
        y = att @ v 
        
        # 10. 重组形状并投影输出
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.c_proj(y)
        
        return y

class MLP(nn.Module):
    """
    标准的前馈神经网络 (Feed-Forward Network)。
    结构: Linear -> SiLU -> Linear
    """
    def __init__(self, config):
        super().__init__()
        # 按照 GPT 标准，隐藏层维度通常是 4倍的 embedding 维度
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.silu(x) # 激活函数
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """
    Transformer 模块 (Block)。
    包含一个 Attention 层和一个 MLP 层，使用 Pre-Norm 结构。
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # 残差连接 (Residual Connection)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ModelConfig:
    """
    模型配置容器类。
    用于传递超参数。
    """
    def __init__(self, **kwargs):
        # 默认参数
        self.n_layer = 12
        self.n_head = 16
        self.n_embd = 1024
        self.block_size = 1024
        self.vocab_size = 50304 
        self.use_rope = True
        self.use_cope = False
        
        # 从 kwargs 更新参数
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPT(nn.Module):
    """
    GPT 主体模型类。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token Embedding 层
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Transformer 层堆叠
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 最终归一化层
            ln_f = RMSNorm(config.n_embd),
        ))
        # 语言模型头 (输出层)，将向量映射回词表大小
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ 参数初始化策略 """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播。
        参数:
            idx: 输入 token 索引，形状 [Batch, SeqLen]
            targets: 目标 token 索引 (可选)，用于计算 Loss
        """
        device = idx.device
        b, t = idx.size()
        
        # Embedding
        tok_emb = self.transformer.wte(idx) 
        x = tok_emb
        
        # 通过所有 Transformer Block
        for block in self.transformer.h:
            x = block(x)
        
        # 最终归一化
        x = self.transformer.ln_f(x)
        
        # 计算 Logits [Batch, SeqLen, VocabSize]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 如果提供了 targets，计算交叉熵损失
            # view(-1, ...) 将 Batch 和 SeqLen 展平，以适配 CrossEntropyLoss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
