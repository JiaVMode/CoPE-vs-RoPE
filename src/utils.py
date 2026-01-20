import os
import yaml
import logging
import random
import numpy as np
import torch

def load_config(config_path):
    """
    加载 YAML 配置文件。
    
    参数:
        config_path (str): YAML 文件的路径。
        
    返回:
        dict: 解析后的配置字典。
    """
    # 打开并读取 YAML 文件，指定 utf-8 编码以支持中文
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(out_dir, name="Train"):
    """
    配置并初始化日志记录器。
    会将日志同时输出到控制台和指定目录下的 run.log 文件。
    
    参数:
        out_dir (str): 日志文件的输出目录。
        name (str): Logger 的名称，默认为 "Train"。
        
    返回:
        logging.Logger: 配置好的 logger 对象。
    """
    # 确保输出目录存在，如果不存在则创建
    os.makedirs(out_dir, exist_ok=True)
    
    # 获取 logger 实例
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO) # 设置日志级别为 INFO
    
    # 如果 logger 已经有 handler (防止重复添加导致日志重复打印)，则直接返回
    if logger.handlers:
        return logger
        
    # 定义日志格式: 时间 | 级别 | 消息
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 添加文件处理器 (FileHandler): 将日志写入文件中
    file_handler = logging.FileHandler(os.path.join(out_dir, 'run.log'), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 添加控制台处理器 (StreamHandler): 将日志输出到终端
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def set_seed(seed):
    """
    设置全局随机种子，确保实验可复现。
    
    参数:
        seed (int): 随机种子数值。
    """
    # 设置 Python 标准库的 random 种子
    random.seed(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 PyTorch CPU 的随机种子
    torch.manual_seed(seed)
    
    # 如果有 GPU，也设置 CUDA 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
