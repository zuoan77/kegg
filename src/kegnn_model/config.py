#!/usr/bin/env python3
"""
配置类定义
用于KEGG数据训练的配置管理
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """数据配置类"""
    data_dir: str = "kegg_real_processed"
    nodes_file: str = "processed_nodes.csv"
    edges_file: str = "processed_edges.csv"
    weights_file: str = "processed_weights.json"
    
    # 数据预处理参数
    normalize_features: bool = True
    add_self_loops: bool = False
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 负采样参数
    negative_sampling_ratio: float = 1.0
    use_hard_negative_mining: bool = True


@dataclass
class ModelConfig:
    """模型配置类"""
    model_type: str = "GraphSAGE"  # GCN, GraphSAGE, GAT, GIN
    input_dim: int = 8
    hidden_dim: int = 256
    output_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.4
    activation: str = "relu"
    
    # 高级特性
    use_batch_norm: bool = True
    use_residual: bool = True
    use_attention: bool = True
    attention_heads: int = 8
    
    # 边预测配置
    edge_pred_method: str = "enhanced_concat"  # concat, hadamard, cosine, enhanced_concat, enhanced_hadamard
    use_enhanced_predictor: bool = True
    edge_pred_hidden_dim: int = 128


@dataclass
class TrainingConfig:
    """训练配置类"""
    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    
    # 优化器配置
    optimizer: str = "Adam"
    scheduler: str = "StepLR"
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.5
    
    # 早停配置
    early_stopping: bool = True
    patience: int = 50
    min_delta: float = 0.001
    
    # 损失函数配置
    loss_function: str = "BCEWithLogitsLoss"
    pos_weight: Optional[float] = None
    
    # 评估配置
    eval_every: int = 10
    save_best_model: bool = True
    
    # 输出配置
    output_dir: str = "results"
    experiment_name: str = "kegg_training"
    save_plots: bool = True
    verbose: bool = True


def load_config_from_yaml(yaml_path: str) -> tuple[DataConfig, ModelConfig, TrainingConfig]:
    """从YAML文件加载配置"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 创建配置对象
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    
    return data_config, model_config, training_config


def create_default_configs() -> tuple[DataConfig, ModelConfig, TrainingConfig]:
    """创建默认配置"""
    return DataConfig(), ModelConfig(), TrainingConfig()


# 配置验证函数
def validate_configs(data_config: DataConfig, model_config: ModelConfig, training_config: TrainingConfig) -> bool:
    """验证配置的有效性"""
    # 验证数据配置
    if not (0 < data_config.train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")
    if not (0 < data_config.val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1")
    if not (0 < data_config.test_ratio < 1):
        raise ValueError("test_ratio must be between 0 and 1")
    if abs(data_config.train_ratio + data_config.val_ratio + data_config.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    # 验证模型配置
    if model_config.input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if model_config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if model_config.output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if model_config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not (0 <= model_config.dropout <= 1):
        raise ValueError("dropout must be between 0 and 1")
    
    # 验证训练配置
    if training_config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if training_config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if training_config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if training_config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    
    return True