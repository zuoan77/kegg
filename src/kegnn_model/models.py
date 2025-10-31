#!/usr/bin/env python3
"""
KEGG图神经网络模型
实现用于代谢通路发现的图神经网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, BatchNorm, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
import math
import logging

logger = logging.getLogger(__name__)


class EnhancedEdgePredictor(nn.Module):
    """增强的边预测器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, method: str = "enhanced_concat"):
        super().__init__()
        self.method = method
        
        if method == "enhanced_concat":
            self.predictor = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif method == "enhanced_hadamard":
            self.predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            self.predictor = nn.Linear(input_dim * 2 if method == "concat" else input_dim, 1)
    
    def forward(self, node_embeddings: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: (num_nodes, embedding_dim)
            edges: (2, num_edges) 边的索引
        """
        src_embeddings = node_embeddings[edges[0]]  # (num_edges, embedding_dim)
        tgt_embeddings = node_embeddings[edges[1]]  # (num_edges, embedding_dim)
        
        if self.method == "concat" or self.method == "enhanced_concat":
            edge_features = torch.cat([src_embeddings, tgt_embeddings], dim=1)
        elif self.method == "hadamard" or self.method == "enhanced_hadamard":
            edge_features = src_embeddings * tgt_embeddings
        elif self.method == "cosine":
            edge_features = F.cosine_similarity(src_embeddings, tgt_embeddings, dim=1, keepdim=True)
            return edge_features.squeeze()
        else:
            raise ValueError(f"Unknown edge prediction method: {self.method}")
        
        return self.predictor(edge_features).squeeze()


class GraphSAGELayer(nn.Module):
    """自定义GraphSAGE层"""
    
    def __init__(self, input_dim: int, output_dim: int, use_batch_norm: bool = True, 
                 use_residual: bool = True, dropout: float = 0.1):
        super().__init__()
        self.sage_conv = SAGEConv(input_dim, output_dim)
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual and (input_dim == output_dim)
        
        if use_batch_norm:
            self.batch_norm = BatchNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接的投影层
        if use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # GraphSAGE卷积
        x = self.sage_conv(x, edge_index)
        
        # 批归一化
        if self.use_batch_norm:
            x = self.batch_norm(x)
        
        # 激活函数
        x = F.relu(x)
        
        # 残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            x = x + identity
        
        # Dropout
        x = self.dropout(x)
        
        return x


class AttentionAggregator(nn.Module):
    """注意力聚合器"""
    
    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.output_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # 计算Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)
        
        # 重塑并投影输出
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.output_proj(attended)
        
        return output


class KEGGGraphModel(nn.Module):
    """KEGG图神经网络模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        
        # 构建图卷积层
        self.conv_layers = nn.ModuleList()
        
        # 输入层
        if self.model_type == "GraphSAGE":
            self.conv_layers.append(
                GraphSAGELayer(
                    config.input_dim, 
                    config.hidden_dim,
                    config.use_batch_norm,
                    config.use_residual,
                    config.dropout
                )
            )
        elif self.model_type == "GCN":
            self.conv_layers.append(GCNConv(config.input_dim, config.hidden_dim))
        elif self.model_type == "GAT":
            self.conv_layers.append(
                GATConv(
                    config.input_dim, 
                    config.hidden_dim // config.attention_heads,
                    heads=config.attention_heads,
                    dropout=config.dropout
                )
            )
        elif self.model_type == "GIN":
            gin_nn = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )
            self.conv_layers.append(GINConv(gin_nn))
        
        # 隐藏层
        for i in range(config.num_layers - 2):
            if self.model_type == "GraphSAGE":
                self.conv_layers.append(
                    GraphSAGELayer(
                        config.hidden_dim,
                        config.hidden_dim,
                        config.use_batch_norm,
                        config.use_residual,
                        config.dropout
                    )
                )
            elif self.model_type == "GCN":
                self.conv_layers.append(GCNConv(config.hidden_dim, config.hidden_dim))
            elif self.model_type == "GAT":
                self.conv_layers.append(
                    GATConv(
                        config.hidden_dim,
                        config.hidden_dim // config.attention_heads,
                        heads=config.attention_heads,
                        dropout=config.dropout
                    )
                )
            elif self.model_type == "GIN":
                gin_nn = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim)
                )
                self.conv_layers.append(GINConv(gin_nn))
        
        # 输出层
        if config.num_layers > 1:
            if self.model_type == "GraphSAGE":
                self.conv_layers.append(
                    GraphSAGELayer(
                        config.hidden_dim,
                        config.output_dim,
                        config.use_batch_norm,
                        False,  # 输出层不使用残差连接
                        config.dropout
                    )
                )
            elif self.model_type == "GCN":
                self.conv_layers.append(GCNConv(config.hidden_dim, config.output_dim))
            elif self.model_type == "GAT":
                self.conv_layers.append(
                    GATConv(
                        config.hidden_dim,
                        config.output_dim,
                        heads=1,
                        dropout=config.dropout
                    )
                )
            elif self.model_type == "GIN":
                gin_nn = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.output_dim),
                    nn.ReLU(),
                    nn.Linear(config.output_dim, config.output_dim)
                )
                self.conv_layers.append(GINConv(gin_nn))
        
        # 注意力聚合器
        if config.use_attention:
            self.attention_aggregator = AttentionAggregator(config.output_dim, config.attention_heads)
        
        # 边预测器
        if config.use_enhanced_predictor:
            self.edge_predictor = EnhancedEdgePredictor(
                config.output_dim,
                config.edge_pred_hidden_dim,
                config.edge_pred_method
            )
        else:
            if config.edge_pred_method == "concat":
                self.edge_predictor = nn.Linear(config.output_dim * 2, 1)
            else:
                self.edge_predictor = nn.Linear(config.output_dim, 1)
        
        # Dropout层
        self.dropout = nn.Dropout(config.dropout)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                pred_edges: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 (num_nodes, input_dim)
            edge_index: 图的边索引 (2, num_edges)
            pred_edges: 需要预测的边 (2, num_pred_edges)
        
        Returns:
            edge_predictions: 边预测结果 (num_pred_edges,)
        """
        # 图卷积层
        for i, conv_layer in enumerate(self.conv_layers):
            if self.model_type == "GraphSAGE":
                x = conv_layer(x, edge_index)
            else:
                x = conv_layer(x, edge_index)
                if i < len(self.conv_layers) - 1:  # 不在最后一层应用激活函数和dropout
                    x = F.relu(x)
                    x = self.dropout(x)
        
        # 注意力聚合
        if self.config.use_attention and hasattr(self, 'attention_aggregator'):
            # 为注意力机制重塑张量
            x_reshaped = x.unsqueeze(0)  # (1, num_nodes, output_dim)
            x_attended = self.attention_aggregator(x_reshaped)
            x = x_attended.squeeze(0)  # (num_nodes, output_dim)
        
        # 边预测
        if hasattr(self, 'edge_predictor'):
            edge_predictions = self.edge_predictor(x, pred_edges)
        else:
            # 简单的点积预测
            src_embeddings = x[pred_edges[0]]
            tgt_embeddings = x[pred_edges[1]]
            edge_predictions = torch.sum(src_embeddings * tgt_embeddings, dim=1)
        
        return edge_predictions
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """获取节点嵌入"""
        # 图卷积层
        for i, conv_layer in enumerate(self.conv_layers):
            if self.model_type == "GraphSAGE":
                x = conv_layer(x, edge_index)
            else:
                x = conv_layer(x, edge_index)
                if i < len(self.conv_layers) - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
        
        # 注意力聚合
        if self.config.use_attention and hasattr(self, 'attention_aggregator'):
            x_reshaped = x.unsqueeze(0)
            x_attended = self.attention_aggregator(x_reshaped)
            x = x_attended.squeeze(0)
        
        return x


def create_kegg_model(config) -> KEGGGraphModel:
    """创建KEGG图模型的工厂函数"""
    logger.info(f"🏗️  创建 {config.model_type} 模型...")
    logger.info(f"   输入维度: {config.input_dim}")
    logger.info(f"   隐藏维度: {config.hidden_dim}")
    logger.info(f"   输出维度: {config.output_dim}")
    logger.info(f"   层数: {config.num_layers}")
    logger.info(f"   边预测方法: {config.edge_pred_method}")
    
    model = KEGGGraphModel(config)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"   总参数数: {total_params:,}")
    logger.info(f"   可训练参数数: {trainable_params:,}")
    
    return model