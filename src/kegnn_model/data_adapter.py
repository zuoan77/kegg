#!/usr/bin/env python3
"""
KEGG数据适配器
用于处理KEGG数据的加载、预处理和批处理
"""

import os
import pandas as pd
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class KEGGDataAdapter:
    """KEGG数据适配器类"""
    
    def __init__(self, data_dir: str, normalize_features: bool = True):
        self.data_dir = Path(data_dir)
        self.normalize_features = normalize_features
        self.scaler = StandardScaler() if normalize_features else None
        
        # 数据存储
        self.nodes_df = None
        self.edges_df = None
        self.weights_data = None
        self.node_features = None
        self.edge_index = None
        self.edge_weights = None
        self.node_mapping = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """加载KEGG数据"""
        logger.info("🔧 加载KEGG数据...")
        
        # 文件路径
        nodes_file = self.data_dir / "processed_nodes.csv"
        edges_file = self.data_dir / "processed_edges.csv"
        weights_file = self.data_dir / "processed_weights.json"
        
        # 检查文件是否存在
        for file_path in [nodes_file, edges_file, weights_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 读取数据
        self.nodes_df = pd.read_csv(nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        
        with open(weights_file, 'r') as f:
            self.weights_data = json.load(f)
        
        logger.info(f"   节点数: {len(self.nodes_df)}")
        logger.info(f"   边数: {len(self.edges_df)}")
        logger.info(f"   权重数: {len(self.weights_data)}")
        
        return self.nodes_df, self.edges_df, self.weights_data
    
    def prepare_node_features(self) -> torch.Tensor:
        """准备节点特征"""
        if self.nodes_df is None:
            raise ValueError("请先调用load_data()加载数据")
        
        # 提取数值特征列
        feature_columns = []
        for col in self.nodes_df.columns:
            if col not in ['node_id', 'node_type', 'description'] and self.nodes_df[col].dtype in ['int64', 'float64']:
                feature_columns.append(col)
        
        if not feature_columns:
            # 如果没有数值特征，创建默认特征
            logger.warning("未找到数值特征，使用默认特征")
            features = np.ones((len(self.nodes_df), 1))
        else:
            features = self.nodes_df[feature_columns].values
        
        # 处理缺失值
        features = np.nan_to_num(features, nan=0.0)
        
        # 标准化特征
        if self.normalize_features and self.scaler is not None:
            features = self.scaler.fit_transform(features)
        
        self.node_features = torch.FloatTensor(features)
        logger.info(f"   节点特征维度: {self.node_features.shape}")
        
        return self.node_features
    
    def prepare_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备边数据"""
        if self.edges_df is None:
            raise ValueError("请先调用load_data()加载数据")
        
        # 创建节点映射
        unique_nodes = set(self.edges_df['from_node'].tolist() + self.edges_df['to_node'].tolist())
        self.node_mapping = {node: idx for idx, node in enumerate(sorted(unique_nodes))}
        
        # 转换边索引
        source_indices = [self.node_mapping[node] for node in self.edges_df['from_node']]
        target_indices = [self.node_mapping[node] for node in self.edges_df['to_node']]
        
        # 创建边索引张量 (2, num_edges)
        self.edge_index = torch.LongTensor([source_indices, target_indices])
        
        # 准备边权重
        edge_weights = []
        for _, row in self.edges_df.iterrows():
            edge_key = f"{row['from_node']}-{row['to_node']}"
            weight = self.weights_data.get(edge_key, 1.0)
            edge_weights.append(weight)
        
        self.edge_weights = torch.FloatTensor(edge_weights)
        
        logger.info(f"   边索引形状: {self.edge_index.shape}")
        logger.info(f"   边权重形状: {self.edge_weights.shape}")
        
        return self.edge_index, self.edge_weights
    
    def create_negative_edges(self, num_negative: Optional[int] = None) -> torch.Tensor:
        """创建负样本边"""
        if self.edge_index is None:
            raise ValueError("请先调用prepare_edges()准备边数据")
        
        num_nodes = len(self.node_mapping)
        num_positive = self.edge_index.shape[1]
        
        if num_negative is None:
            num_negative = num_positive
        
        # 创建正边集合用于快速查找
        positive_edges = set()
        for i in range(num_positive):
            src, tgt = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            positive_edges.add((src, tgt))
            positive_edges.add((tgt, src))  # 无向图
        
        # 生成负样本
        negative_edges = []
        attempts = 0
        max_attempts = num_negative * 10
        
        while len(negative_edges) < num_negative and attempts < max_attempts:
            src = np.random.randint(0, num_nodes)
            tgt = np.random.randint(0, num_nodes)
            
            if src != tgt and (src, tgt) not in positive_edges:
                negative_edges.append([src, tgt])
            
            attempts += 1
        
        if len(negative_edges) < num_negative:
            logger.warning(f"只生成了 {len(negative_edges)} 个负样本，目标是 {num_negative}")
        
        return torch.LongTensor(negative_edges).t()
    
    def get_graph_data(self) -> Dict[str, torch.Tensor]:
        """获取完整的图数据"""
        if self.node_features is None:
            self.prepare_node_features()
        if self.edge_index is None:
            self.prepare_edges()
        
        return {
            'x': self.node_features,
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weights,
            'num_nodes': len(self.node_mapping)
        }


class KEGGDataset(Dataset):
    """KEGG数据集类"""
    
    def __init__(self, positive_edges: torch.Tensor, negative_edges: torch.Tensor, 
                 node_features: torch.Tensor, edge_weights: Optional[torch.Tensor] = None):
        self.positive_edges = positive_edges
        self.negative_edges = negative_edges
        self.node_features = node_features
        self.edge_weights = edge_weights
        
        # 创建标签
        self.pos_labels = torch.ones(positive_edges.shape[1])
        self.neg_labels = torch.zeros(negative_edges.shape[1])
        
        # 合并边和标签
        self.all_edges = torch.cat([positive_edges, negative_edges], dim=1)
        self.all_labels = torch.cat([self.pos_labels, self.neg_labels])
        
        # 创建权重
        if edge_weights is not None:
            zero_weights = torch.zeros(negative_edges.shape[1])
            self.all_weights = torch.cat([edge_weights, zero_weights])
        else:
            self.all_weights = torch.ones(self.all_edges.shape[1])
    
    def __len__(self):
        return self.all_edges.shape[1]
    
    def __getitem__(self, idx):
        return {
            'edge': self.all_edges[:, idx],
            'label': self.all_labels[idx],
            'weight': self.all_weights[idx],
            'node_features': self.node_features
        }


def collate_kegg_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """KEGG数据批处理函数"""
    edges = torch.stack([item['edge'] for item in batch]).t()  # (2, batch_size)
    labels = torch.stack([item['label'] for item in batch])
    weights = torch.stack([item['weight'] for item in batch])
    
    # 节点特征在批次中是共享的
    node_features = batch[0]['node_features']
    
    return {
        'edges': edges,
        'labels': labels,
        'weights': weights,
        'node_features': node_features
    }


def create_data_splits(edges: torch.Tensor, train_ratio: float = 0.7, 
                      val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """创建训练/验证/测试数据分割"""
    num_edges = edges.shape[1]
    indices = torch.randperm(num_edges)
    
    train_end = int(train_ratio * num_edges)
    val_end = train_end + int(val_ratio * num_edges)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_edges = edges[:, train_indices]
    val_edges = edges[:, val_indices]
    test_edges = edges[:, test_indices]
    
    return train_edges, val_edges, test_edges


def prepare_kegg_data_for_training(data_dir: str, config) -> Dict[str, Any]:
    """为训练准备KEGG数据的完整流程"""
    logger.info("🚀 开始准备KEGG训练数据...")
    
    # 创建数据适配器
    adapter = KEGGDataAdapter(data_dir, normalize_features=config.normalize_features)
    
    # 加载数据
    adapter.load_data()
    
    # 准备图数据
    graph_data = adapter.get_graph_data()
    
    # 分割边数据
    train_edges, val_edges, test_edges = create_data_splits(
        graph_data['edge_index'], 
        config.train_ratio, 
        config.val_ratio, 
        config.test_ratio
    )
    
    # 创建负样本
    train_neg_edges = adapter.create_negative_edges(train_edges.shape[1])
    val_neg_edges = adapter.create_negative_edges(val_edges.shape[1])
    test_neg_edges = adapter.create_negative_edges(test_edges.shape[1])
    
    # 创建数据集
    train_dataset = KEGGDataset(train_edges, train_neg_edges, graph_data['x'])
    val_dataset = KEGGDataset(val_edges, val_neg_edges, graph_data['x'])
    test_dataset = KEGGDataset(test_edges, test_neg_edges, graph_data['x'])
    
    logger.info("✅ KEGG数据准备完成")
    logger.info(f"   训练集: {len(train_dataset)} 样本")
    logger.info(f"   验证集: {len(val_dataset)} 样本")
    logger.info(f"   测试集: {len(test_dataset)} 样本")
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'graph_data': graph_data,
        'node_mapping': adapter.node_mapping
    }