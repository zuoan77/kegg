#!/usr/bin/env python3
"""
KEGG图神经网络预测脚本
使用训练好的模型进行化合物连接预测
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

from config import create_default_configs
from data_adapter import KEGGDataAdapter
from models import create_kegg_model

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KEGGPredictor:
    """KEGG化合物连接预测器"""
    
    def __init__(self, model_path: str, data_dir: str = "kegg_real_processed"):
        """
        初始化预测器
        
        Args:
            model_path: 训练好的模型路径
            data_dir: 数据目录
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载配置 - 使用与训练时相同的配置
        from config import DataConfig, ModelConfig, TrainingConfig
        
        # 使用训练时的简化配置
        self.model_config = ModelConfig(
            model_type="GraphSAGE",
            input_dim=7,  # 实际特征维度是7
            hidden_dim=64,
            output_dim=32,
            num_layers=3,
            dropout=0.3
        )
        
        # 初始化数据适配器
        self.data_adapter = KEGGDataAdapter(data_dir)
        
        # 加载模型
        self.model = None
        self.graph_data = None
        self._load_model()
        self._prepare_data()
    
    def _load_model(self):
        """加载训练好的模型"""
        logger.info(f"🔄 加载模型: {self.model_path}")
        
        # 创建模型
        self.model = create_kegg_model(self.model_config)
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✅ 模型加载完成")
    
    def _prepare_data(self):
        """准备图数据"""
        logger.info("🔄 准备图数据...")
        
        # 加载节点和边数据
        self.data_adapter.load_data()
        self.data_adapter.prepare_node_features()
        self.data_adapter.prepare_edges()
        
        # 获取图数据
        self.graph_data = self.data_adapter.get_graph_data()
        
        # 移动到设备
        for key in self.graph_data:
            if isinstance(self.graph_data[key], torch.Tensor):
                self.graph_data[key] = self.graph_data[key].to(self.device)
        
        logger.info("✅ 图数据准备完成")
        logger.info(f"   节点数量: {self.graph_data['num_nodes']}")
        logger.info(f"   边数量: {self.graph_data['edge_index'].shape[1]}")
    
    def predict_single_pair(self, compound1: str, compound2: str) -> Dict[str, float]:
        """
        预测单个化合物对的连接概率
        
        Args:
            compound1: 化合物1的ID
            compound2: 化合物2的ID
            
        Returns:
            预测结果字典
        """
        # 检查化合物是否存在
        if compound1 not in self.data_adapter.node_mapping:
            raise ValueError(f"化合物 {compound1} 不在数据集中")
        if compound2 not in self.data_adapter.node_mapping:
            raise ValueError(f"化合物 {compound2} 不在数据集中")
        
        # 获取节点索引
        idx1 = self.data_adapter.node_mapping[compound1]
        idx2 = self.data_adapter.node_mapping[compound2]
        
        # 创建边张量
        edge_tensor = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            logit = self.model(
                self.graph_data['x'],
                self.graph_data['edge_index'],
                edge_tensor
            )
            probability = torch.sigmoid(logit).item()
        
        return {
            'compound1': compound1,
            'compound2': compound2,
            'connection_probability': probability,
            'prediction': 'Connected' if probability > 0.5 else 'Not Connected',
            'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low'
        }
    
    def predict_batch_pairs(self, compound_pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """
        批量预测化合物对的连接概率
        
        Args:
            compound_pairs: 化合物对列表
            
        Returns:
            预测结果列表，与输入化合物对一一对应
        """
        results = []
        
        # 分离有效和无效的化合物对
        valid_pairs = []
        valid_indices = []
        
        for i, (comp1, comp2) in enumerate(compound_pairs):
            if comp1 in self.data_adapter.node_mapping and comp2 in self.data_adapter.node_mapping:
                valid_pairs.append((comp1, comp2))
                valid_indices.append(i)
            else:
                logger.warning(f"跳过无效化合物对: {comp1} - {comp2}")
        
        # 初始化结果列表，为所有输入对预留位置
        results = [None] * len(compound_pairs)
        
        # 处理无效化合物对
        for i, (comp1, comp2) in enumerate(compound_pairs):
            if i not in valid_indices:
                results[i] = {
                    'compound1': comp1,
                    'compound2': comp2,
                    'connection_probability': 0.0,
                    'prediction': 'Invalid',
                    'confidence': 'Low'
                }
        
        if not valid_pairs:
            logger.error("没有有效的化合物对")
            return [r for r in results if r is not None]
        
        # 创建批量边张量
        indices1 = [self.data_adapter.node_mapping[pair[0]] for pair in valid_pairs]
        indices2 = [self.data_adapter.node_mapping[pair[1]] for pair in valid_pairs]
        
        edge_tensor = torch.tensor([indices1, indices2], dtype=torch.long).to(self.device)
        
        # 批量预测
        with torch.no_grad():
            logits = self.model(
                self.graph_data['x'],
                self.graph_data['edge_index'],
                edge_tensor
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()
        
        # 整理有效化合物对的结果
        for i, (comp1, comp2) in enumerate(valid_pairs):
            # 处理单个预测结果的情况
            if probabilities.ndim == 0:  # 标量情况
                prob = float(probabilities)
            else:  # 数组情况
                prob = probabilities[i]
            
            # 将结果放到正确的位置
            original_index = valid_indices[i]
            results[original_index] = {
                'compound1': comp1,
                'compound2': comp2,
                'connection_probability': float(prob),
                'prediction': 'Connected' if prob > 0.5 else 'Not Connected',
                'confidence': 'High' if abs(prob - 0.5) > 0.3 else 'Medium' if abs(prob - 0.5) > 0.1 else 'Low'
            }
        
        return results
    
    def find_potential_connections(self, target_compound: str, top_k: int = 10, 
                                 min_probability: float = 0.7) -> List[Dict[str, float]]:
        """
        为指定化合物寻找潜在的连接
        
        Args:
            target_compound: 目标化合物ID
            top_k: 返回前k个最可能的连接
            min_probability: 最小概率阈值
            
        Returns:
            潜在连接列表
        """
        if target_compound not in self.data_adapter.node_mapping:
            raise ValueError(f"化合物 {target_compound} 不在数据集中")
        
        target_idx = self.data_adapter.node_mapping[target_compound]
        all_compounds = list(self.data_adapter.node_mapping.keys())
        
        # 创建所有可能的连接对
        candidate_pairs = []
        for compound in all_compounds:
            if compound != target_compound:
                candidate_pairs.append((target_compound, compound))
        
        # 批量预测
        logger.info(f"🔄 为 {target_compound} 寻找潜在连接...")
        results = self.predict_batch_pairs(candidate_pairs)
        
        # 过滤和排序
        filtered_results = [
            result for result in results 
            if result['connection_probability'] >= min_probability
        ]
        
        # 按概率排序
        filtered_results.sort(key=lambda x: x['connection_probability'], reverse=True)
        
        return filtered_results[:top_k]
    
    def get_compound_info(self) -> pd.DataFrame:
        """获取所有化合物信息"""
        return self.data_adapter.nodes_df
    
    def save_predictions(self, predictions: List[Dict], output_path: str):
        """保存预测结果"""
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
        logger.info(f"💾 预测结果已保存到: {output_path}")


def main():
    """主函数 - 演示预测功能"""
    # 检查模型文件
    model_path = "kegg_training_results/best_model.pth"
    if not Path(model_path).exists():
        logger.error(f"模型文件不存在: {model_path}")
        logger.info("请先运行训练脚本生成模型")
        return
    
    # 创建预测器
    predictor = KEGGPredictor(model_path)
    
    # 获取一些化合物进行演示
    compounds = list(predictor.data_adapter.node_mapping.keys())[:10]
    logger.info(f"📋 可用化合物示例: {compounds}")
    
    # 1. 单个预测示例
    if len(compounds) >= 2:
        logger.info("\n🎯 单个预测示例:")
        result = predictor.predict_single_pair(compounds[0], compounds[1])
        logger.info(f"   {result['compound1']} ↔ {result['compound2']}")
        logger.info(f"   连接概率: {result['connection_probability']:.4f}")
        logger.info(f"   预测结果: {result['prediction']}")
        logger.info(f"   置信度: {result['confidence']}")
    
    # 2. 批量预测示例
    if len(compounds) >= 4:
        logger.info("\n📊 批量预测示例:")
        test_pairs = [
            (compounds[0], compounds[1]),
            (compounds[1], compounds[2]),
            (compounds[2], compounds[3])
        ]
        
        batch_results = predictor.predict_batch_pairs(test_pairs)
        for result in batch_results:
            logger.info(f"   {result['compound1']} ↔ {result['compound2']}: {result['connection_probability']:.4f}")
    
    # 3. 寻找潜在连接示例
    if len(compounds) >= 1:
        logger.info(f"\n🔍 为 {compounds[0]} 寻找潜在连接:")
        potential_connections = predictor.find_potential_connections(
            compounds[0], 
            top_k=5, 
            min_probability=0.6
        )
        
        if potential_connections:
            for i, conn in enumerate(potential_connections, 1):
                logger.info(f"   {i}. {conn['compound2']}: {conn['connection_probability']:.4f} ({conn['confidence']})")
        else:
            logger.info("   未找到高概率的潜在连接")
    
    logger.info("\n✅ 预测演示完成!")
    logger.info("\n💡 使用方法:")
    logger.info("   from predict_kegg import KEGGPredictor")
    logger.info("   predictor = KEGGPredictor('kegg_training_results/best_model.pth')")
    logger.info("   result = predictor.predict_single_pair('compound1', 'compound2')")


if __name__ == "__main__":
    main()