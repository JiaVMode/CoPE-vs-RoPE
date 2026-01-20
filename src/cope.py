import torch
import torch.nn as nn
import math

class CoPE(nn.Module):
    """
    CoPE implementation based on the paper snippet provided by user.
    """
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim , npos_max)) 
        
        # 调试用：存储门控用于可视化 (保留此功能以便对比)
        self.last_gates = None

    def forward(self, query, attn_logits):
        # compute positions
        gates = torch.sigmoid(attn_logits)
        
        # 调试：保存门控
        self.last_gates = gates.detach()
        
        # Paper snippet: pos = gates.flip(-1).cumsum(dim=-1).flip(-1) pos = pos.clamp(max=self.npos_max - 1)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        
        logits_int = torch.matmul(query , self.pos_emb) 
        logits_ceil = logits_int.gather(-1, pos_ceil) 				
        logits_floor = logits_int.gather(-1, pos_floor) 
        w = pos - pos_floor  
        
        return logits_ceil * w + logits_floor * (1 - w)
