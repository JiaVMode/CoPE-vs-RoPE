import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class SelectiveCountingTask:
    """
    选择性计数任务生成器 (Phase 8 - High Difficulty)
    
    任务描述：
    我们需要模型仅仅对指令中指定的对象进行计数。
    
    结构：
    1. 指令区 (Directive): "Count [A] and [C]"
    2. 内容区 (Content): 包含 [A], [B], [C], [A], [D]... 以及大量噪声词。
    3. 目标 (Target): 此例中，目标是统计 A 和 C 的总出现次数，忽略 B 和 D。
    
    这是验证 CoPE 能否根据"指令上下文"动态调整关注点的绝佳任务。
    """
    def __init__(self, vocab_size=5000, seq_len=1024, min_directives=1, max_directives=2, num_object_types=10):
        """
        初始化任务生成器。
        
        参数:
            vocab_size: 词表大小。
            seq_len: 生成序列的总长度。
            min_directives: 指令中最少包含的对象种类数 (例如最少只数 1 种)。
            max_directives: 指令中最多包含的对象种类数 (例如最多同时数 2 种)。
            num_object_types: 候选对象的总种类数 (例如 A, B, C... J 共10种)。
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.min_directives = min_directives
        self.max_directives = max_directives
        
        # 定义特殊 Token ID
        self.pad_token = 0      # 填充
        self.bos_token = 1      # 序列开始
        self.sep_token = 2      # 分隔符 (分割指令和内容)
        self.query_token = 3    # 查询符 (提示模型输出答案)
        
        # 对象 Token IDs (用于计数的目标/非目标对象)
        # 假设从 ID 10 开始分配，共 10 个 (10-19)
        self.obj_start_id = 10
        self.obj_end_id = 10 + num_object_types - 1
        
        # 指令关键词 IDs
        self.token_count = 20   # "Count" 单词
        self.token_and = 21     # "and" 单词
        
        # 答案 Token IDs
        # 答案直接用 token ID 表示 (例如 ID 100 代表 0，ID 105 代表 5)
        # 预留一段空间
        self.count_output_start = 100
        
        # 噪声 Token IDs
        # 剩下的 ID 全部作为随机噪声
        self.noise_start = 200
        self.noise_end = vocab_size - 1

    def generate_sample(self):
        """
        生成单个样本。
        
        返回:
            dict: 包含 input_ids, labels 和 真实计数值。
        """
        # ==========================
        # 1. 生成指令部分 (Directive)
        # ==========================
        
        # 随机决定这次要数几个对象 (例如数 A 和 B，num=2)
        num_targets = random.randint(self.min_directives, self.max_directives)
        
        # 获取所有可能的对象列表 (10-19)
        all_objs = list(range(self.obj_start_id, self.obj_end_id + 1))
        
        # 从中随机选出 num_targets 个作为本次的目标
        target_objs = random.sample(all_objs, num_targets)
        
        # 构建指令序列: [BOS] Count [OBJ1] and [OBJ2] [SEP]
        directive_seq = [self.bos_token, self.token_count]
        for i, obj in enumerate(target_objs):
            directive_seq.append(obj)
            # 如果不是最后一个对象，添加 "and" 连接
            if i < len(target_objs) - 1:
                directive_seq.append(self.token_and)
        directive_seq.append(self.sep_token)
        
        # ==========================
        # 2. 生成内容序列 (Content)
        # ==========================
        
        # 计算剩余可用长度
        # 预留 2 个位置给 [QUERY] 和 [ANSWER]
        overhead = len(directive_seq) + 2 
        available_len = self.seq_len - overhead
        
        seq = []
        actual_count = 0 # 真实计数累加器
        
        for _ in range(available_len):
            # 50% 概率生成对象 (Object)，50% 概率生成噪声 (Noise)
            if random.random() < 0.5:
                # 随机选一个对象 (可能是目标，也可能不是)
                obj = random.choice(all_objs)
                seq.append(obj)
                
                # 关键逻辑：只有当该对象在 target_objs 指令列表中时，才计数！
                if obj in target_objs:
                    actual_count += 1
            else:
                # 生成随机噪声
                noise = random.randint(self.noise_start, self.noise_end)
                seq.append(noise)
                
        # ==========================
        # 3. 组合与标签构建
        # ==========================
        
        # 完整输入: 指令 + 内容 + 查询符
        full_seq = directive_seq + seq + [self.query_token]
        
        # 答案 Token
        # 将计数值映射到 token ID
        count_token = self.count_output_start + actual_count
        # 防止越界 (虽然不太可能)
        if count_token >= self.vocab_size:
            count_token = self.vocab_size - 1
            
        # 将答案添加到序列末尾 (用于训练时的 Target)
        full_seq.append(count_token)
        
        # 确保长度不超过 seq_len (防御性截断)
        if len(full_seq) > self.seq_len:
             full_seq = full_seq[:self.seq_len]
        
        # 转换为 Tensor
        input_ids = torch.tensor(full_seq, dtype=torch.long)
        
        # 构建 Labels
        # 我们只希望计算最后一个 token (答案) 的损失，其他位置设为 -100 (忽略)
        labels = torch.full((len(input_ids),), -100, dtype=torch.long)
        labels[-1] = count_token
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "count": actual_count
        }

class LogicalCountingDataset(Dataset):
    """
    PyTorch Dataset 封装
    """
    def __init__(self, num_samples=10000, mode='selective', **kwargs):
        self.num_samples = num_samples
        if mode == 'selective':
            self.task = SelectiveCountingTask(**kwargs)
        else:
            # 如果有其他模式可以在这里扩展
             raise NotImplementedError("目前仅支持 'selective' 模式")
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # 实时生成样本，不需要预先存储
        return self.task.generate_sample()

def collate_fn(batch):
    """
    DataLoader 的整理函数：将多个样本拼接成 Batch。
    """
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # 填充 Pad (batch_first=True)
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "counts": torch.tensor([item['count'] for item in batch])
    }

def get_dataloader(batch_size=32, num_samples=10000, mode='selective', **kwargs):
    """
    获取 DataLoader 工厂函数。
    """
    dataset = LogicalCountingDataset(num_samples=num_samples, mode=mode, **kwargs)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4, # 使用多进程加载数据
        pin_memory=True
    )
