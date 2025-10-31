#!/usr/bin/env python3
"""
基于GNN模型的KEGG节点预测器
整合pathway_discovery功能，支持KO序列到节点预测的完整流程

主要功能：
1. KO序列输入和抽象节点映射
2. 基于GNN模型的节点连接预测
3. 代谢通路推理和路径发现
4. 节点特征分析和可视化
"""

import sys
import os
import logging
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入必要模块
from predict_kegg import KEGGPredictor
from data_adapter import KEGGDataAdapter
from config import ModelConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KEGGNodePredictor:
    """KEGG节点预测器 - 基于GNN模型"""
    
    def __init__(self, 
                 model_path: str = "kegg_training_results/best_model.pth",
                 data_dir: str = "kegg_real_processed"):
        """
        初始化节点预测器
        
        Args:
            model_path: 训练好的GNN模型路径
            data_dir: KEGG数据目录
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("🚀 初始化KEGG节点预测器...")
        logger.info(f"   模型路径: {model_path}")
        logger.info(f"   数据目录: {data_dir}")
        logger.info(f"   计算设备: {self.device}")
        
        # 初始化组件
        self._load_predictor()
        self._load_node_data()
        
    def _load_predictor(self):
        """加载GNN预测器"""
        try:
            logger.info("🔧 加载GNN预测器...")
            self.predictor = KEGGPredictor(self.model_path)
            logger.info("✅ GNN预测器加载成功")
        except Exception as e:
            logger.error(f"❌ GNN预测器加载失败: {e}")
            raise
            
    def _load_node_data(self):
        """加载节点数据"""
        try:
            logger.info("📊 加载节点数据...")
            
            # 加载节点信息
            nodes_file = os.path.join(self.data_dir, "processed_nodes.csv")
            self.nodes_df = pd.read_csv(nodes_file)
            
            # 创建节点映射
            self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes_df['node_id'])}
            self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
            
            # 获取所有节点列表
            self.all_nodes = list(self.nodes_df['node_id'])
            
            logger.info(f"✅ 节点数据加载成功: {len(self.all_nodes)} 个节点")
            
        except Exception as e:
            logger.error(f"❌ 节点数据加载失败: {e}")
            raise
    
    def predict_node_connections(self, 
                               target_node: str, 
                               candidate_nodes: Optional[List[str]] = None,
                               top_k: int = 10,
                               min_probability: float = 0.1) -> Dict:
        """
        预测目标节点与候选节点的连接概率
        
        Args:
            target_node: 目标节点ID
            candidate_nodes: 候选节点列表，如果为None则使用所有节点
            top_k: 返回前K个最可能的连接
            min_probability: 最小概率阈值
            
        Returns:
            预测结果字典
        """
        logger.info(f"🎯 预测节点连接: {target_node}")
        
        if target_node not in self.all_nodes:
            logger.error(f"❌ 目标节点不存在: {target_node}")
            return {'error': f'目标节点不存在: {target_node}'}
        
        # 确定候选节点
        if candidate_nodes is None:
            candidate_nodes = [node for node in self.all_nodes if node != target_node]
        else:
            # 验证候选节点
            valid_candidates = [node for node in candidate_nodes if node in self.all_nodes and node != target_node]
            if len(valid_candidates) != len(candidate_nodes):
                logger.warning(f"⚠️  部分候选节点无效，有效候选节点: {len(valid_candidates)}")
            candidate_nodes = valid_candidates
        
        logger.info(f"📋 候选节点数量: {len(candidate_nodes)}")
        
        # 批量预测连接概率
        pairs = [(target_node, candidate) for candidate in candidate_nodes]
        predictions = self.predictor.predict_batch(pairs)
        
        # 过滤和排序结果
        results = []
        for (source, target), probability in predictions.items():
            if probability >= min_probability:
                confidence = self._get_confidence_level(probability)
                results.append({
                    'target_node': target,
                    'probability': probability,
                    'confidence': confidence,
                    'prediction': 'Connected' if probability > 0.5 else 'Not Connected'
                })
        
        # 按概率排序
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        # 返回前K个结果
        top_results = results[:top_k]
        
        return {
            'query_node': target_node,
            'total_candidates': len(candidate_nodes),
            'valid_predictions': len(results),
            'top_connections': top_results,
            'statistics': {
                'max_probability': max([r['probability'] for r in results]) if results else 0,
                'min_probability': min([r['probability'] for r in results]) if results else 0,
                'avg_probability': np.mean([r['probability'] for r in results]) if results else 0,
                'connected_count': len([r for r in results if r['prediction'] == 'Connected']),
                'high_confidence_count': len([r for r in results if r['confidence'] == 'High'])
            }
        }
    
    def predict_pathway_from_nodes(self, 
                                 node_sequence: List[str],
                                 max_path_length: int = 10,
                                 min_probability: float = 0.1) -> Dict:
        """
        从节点序列预测代谢通路
        
        Args:
            node_sequence: 节点序列
            max_path_length: 最大路径长度
            min_probability: 最小边概率阈值
            
        Returns:
            通路预测结果
        """
        logger.info(f"🛤️  预测代谢通路: {len(node_sequence)} 个节点")
        logger.info(f"📋 节点序列: {node_sequence}")
        
        # 验证节点
        valid_nodes = [node for node in node_sequence if node in self.all_nodes]
        if len(valid_nodes) != len(node_sequence):
            invalid_nodes = [node for node in node_sequence if node not in self.all_nodes]
            logger.warning(f"⚠️  无效节点: {invalid_nodes}")
        
        if len(valid_nodes) < 2:
            return {'error': '至少需要2个有效节点'}
        
        # 生成所有可能的有向边（不包含自环）
        edges_to_predict = []
        for i in range(len(valid_nodes)):
            for j in range(len(valid_nodes)):
                if i != j:
                    edges_to_predict.append((valid_nodes[i], valid_nodes[j]))
        
        logger.info(f"🔗 预测边数量: {len(edges_to_predict)}")
        
        # 批量预测边概率
        edge_predictions = self.predictor.predict_batch_pairs(edges_to_predict)
        
        # 将边预测结果转换为字典格式
        edge_dict = {}
        for i, (node1, node2) in enumerate(edges_to_predict):
            edge_key = (node1, node2)
            edge_dict[edge_key] = edge_predictions[i]['connection_probability']
        
        # 构建路径（使用有向边，路径长度上限不超过唯一节点数）
        pathways = self._build_pathways(valid_nodes, edge_dict, max_path_length, min_probability)
        
        return {
            'input_nodes': node_sequence,
            'valid_nodes': valid_nodes,
            'total_edges_predicted': len(edges_to_predict),
            'pathways': pathways,
            'edge_predictions': {f"{edge[0]}-{edge[1]}": prob for edge, prob in edge_dict.items()},
            'statistics': {
                'total_pathways': len(pathways),
                'avg_pathway_score': np.mean([p['score'] for p in pathways]) if pathways else 0,
                'max_pathway_score': max([p['score'] for p in pathways]) if pathways else 0
            }
        }
    
    def find_connecting_nodes(self, 
                            source_node: str, 
                            target_node: str,
                            max_intermediate_nodes: int = 3,
                            min_probability: float = 0.3) -> Dict:
        """
        寻找连接两个节点的中间节点
        
        Args:
            source_node: 源节点
            target_node: 目标节点
            max_intermediate_nodes: 最大中间节点数
            min_probability: 最小连接概率
            
        Returns:
            连接路径结果
        """
        logger.info(f"🔍 寻找连接路径: {source_node} → {target_node}")
        
        if source_node not in self.all_nodes or target_node not in self.all_nodes:
            return {'error': '源节点或目标节点不存在'}
        
        # 直接连接检查
        direct_prob = self.predictor.predict_single_pair(source_node, target_node)['probability']
        
        paths = []
        
        # 添加直接连接
        if direct_prob >= min_probability:
            paths.append({
                'path': [source_node, target_node],
                'length': 1,
                'total_probability': direct_prob,
                'avg_probability': direct_prob,
                'edges': [{'from': source_node, 'to': target_node, 'probability': direct_prob}]
            })
        
        # 寻找通过中间节点的路径
        for num_intermediate in range(1, max_intermediate_nodes + 1):
            intermediate_paths = self._find_paths_with_intermediates(
                source_node, target_node, num_intermediate, min_probability
            )
            paths.extend(intermediate_paths)
        
        # 按总概率排序
        paths.sort(key=lambda x: x['total_probability'], reverse=True)
        
        return {
            'source_node': source_node,
            'target_node': target_node,
            'direct_connection': {
                'probability': direct_prob,
                'exists': direct_prob >= min_probability
            },
            'connecting_paths': paths[:10],  # 返回前10个最佳路径
            'statistics': {
                'total_paths_found': len(paths),
                'best_path_probability': paths[0]['total_probability'] if paths else 0,
                'avg_path_length': np.mean([p['length'] for p in paths]) if paths else 0
            }
        }
    
    def analyze_node_features(self, node_id: str) -> Dict:
        """
        分析节点特征
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点特征分析结果
        """
        if node_id not in self.all_nodes:
            return {'error': f'节点不存在: {node_id}'}
        
        # 获取节点信息
        node_info = self.nodes_df[self.nodes_df['node_id'] == node_id].iloc[0]
        
        # 获取节点嵌入
        embeddings = self.predictor.get_node_embeddings([node_id])
        
        # 分析连接性
        connections = self.predict_node_connections(node_id, top_k=5)
        
        return {
            'node_id': node_id,
            'node_features': {
                'molecular_diff': float(node_info.get('molecular_diff', 0)),
                'rcu_count': int(node_info.get('rcu_count', 0)),
                'total_count': int(node_info.get('total_count', 0)),
                'substrate_count': int(node_info.get('substrate_count', 0)),
                'product_count': int(node_info.get('product_count', 0)),
                'ko_count': int(node_info.get('ko_count', 0)),
                'next_node_count': int(node_info.get('next_node_count', 0))
            },
            'embeddings': embeddings[node_id].tolist() if node_id in embeddings else None,
            'top_connections': connections['top_connections'],
            'connectivity_stats': connections['statistics']
        }
    
    def _get_confidence_level(self, probability: float) -> str:
        """获取置信度等级"""
        if probability > 0.8 or probability < 0.2:
            return 'High'
        elif probability > 0.6 or probability < 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def _build_pathways(self, nodes: List[str], edge_predictions: Dict, 
                       max_length: int, min_probability: float) -> List[Dict]:
        """构建代谢通路：对小规模节点集枚举全排列，使用有向边概率打分"""
        from itertools import permutations
        pathways = []

        # 过滤有效边
        valid_edges = {edge: prob for edge, prob in edge_predictions.items()
                       if prob >= min_probability}

        if not valid_edges:
            return pathways

        unique_nodes = list(dict.fromkeys(nodes))
        unique_node_count = len(unique_nodes)

        # 对小规模问题（例如4个节点）枚举所有排列
        for perm in permutations(unique_nodes, unique_node_count):
            path_nodes = list(perm)
            path_edges = []
            total_score = 0.0
            valid_path = True

            # 按排列顺序检查相邻有向边
            for i in range(len(path_nodes) - 1):
                edge_key = (path_nodes[i], path_nodes[i + 1])
                prob = valid_edges.get(edge_key, 0.0)
                if prob <= 0.0:
                    valid_path = False
                    break
                path_edges.append({
                    'from': path_nodes[i],
                    'to': path_nodes[i + 1],
                    'probability': prob
                })
                total_score += prob

            if not valid_path:
                continue

            pathways.append({
                'nodes': path_nodes,
                'edges': path_edges,
                'length': len(path_nodes),
                'score': total_score / len(path_edges) if path_edges else 0.0,
                'total_probability': total_score
            })

        # 去重与排序
        unique_pathways = []
        seen = set()
        for p in pathways:
            k = tuple(p['nodes'])
            if k not in seen:
                seen.add(k)
                unique_pathways.append(p)

        unique_pathways.sort(key=lambda x: x['score'], reverse=True)
        return unique_pathways
    
    def _build_single_pathway(self, start_node: str, all_nodes: List[str], 
                            valid_edges: Dict, max_length: int) -> Dict:
        """构建单个路径"""
        path_nodes = [start_node]
        path_edges = []
        total_score = 0
        used_nodes = {start_node}
        
        current_node = start_node
        
        for _ in range(max_length - 1):
            best_next = None
            best_prob = 0
            
            # 寻找最佳下一个节点
            for next_node in all_nodes:
                if next_node in used_nodes:
                    continue
                
                edge_key1 = (current_node, next_node)
                # 仅使用有向边概率
                prob = valid_edges.get(edge_key1, 0)
                
                if prob > best_prob:
                    best_prob = prob
                    best_next = next_node
            
            if best_next is None or best_prob == 0:
                break
            
            # 添加到路径
            path_nodes.append(best_next)
            path_edges.append({
                'from': current_node,
                'to': best_next,
                'probability': best_prob
            })
            total_score += best_prob
            used_nodes.add(best_next)
            current_node = best_next
        
        return {
            'nodes': path_nodes,
            'edges': path_edges,
            'length': len(path_nodes),
            'score': total_score / len(path_edges) if path_edges else 0,
            'total_probability': total_score
        }
    
    def _find_paths_with_intermediates(self, source: str, target: str, 
                                     num_intermediate: int, min_prob: float) -> List[Dict]:
        """寻找通过指定数量中间节点的路径"""
        paths = []
        
        # 简化实现：随机选择中间节点进行测试
        import random
        candidate_intermediates = [node for node in self.all_nodes 
                                 if node not in [source, target]]
        
        # 限制搜索空间
        if len(candidate_intermediates) > 50:
            candidate_intermediates = random.sample(candidate_intermediates, 50)
        
        from itertools import combinations
        
        for intermediate_combo in combinations(candidate_intermediates, num_intermediate):
            # 构建路径：source -> intermediate1 -> ... -> intermediateN -> target
            full_path = [source] + list(intermediate_combo) + [target]
            
            # 检查所有边的概率
            path_edges = []
            total_prob = 1.0
            valid_path = True
            
            for i in range(len(full_path) - 1):
                edge_prob = self.predictor.predict_single_pair(full_path[i], full_path[i + 1])['probability']
                
                if edge_prob < min_prob:
                    valid_path = False
                    break
                
                path_edges.append({
                    'from': full_path[i],
                    'to': full_path[i + 1],
                    'probability': edge_prob
                })
                total_prob *= edge_prob
            
            if valid_path:
                paths.append({
                    'path': full_path,
                    'length': len(full_path) - 1,
                    'total_probability': total_prob,
                    'avg_probability': total_prob ** (1.0 / len(path_edges)),
                    'edges': path_edges
                })
        
        return paths


def main():
    """演示节点预测功能"""
    print("🚀 KEGG节点预测器演示")
    print("=" * 60)
    
    # 初始化预测器
    predictor = KEGGNodePredictor()
    
    # 获取一些示例节点
    sample_nodes = predictor.all_nodes[:10]
    print(f"📋 示例节点: {sample_nodes}")
    
    # 1. 节点连接预测
    print("\n🎯 1. 节点连接预测演示")
    target_node = sample_nodes[0]
    connections = predictor.predict_node_connections(target_node, top_k=5)
    
    print(f"目标节点: {connections['query_node']}")
    print(f"候选节点数: {connections['total_candidates']}")
    print(f"有效预测数: {connections['valid_predictions']}")
    print("前5个连接:")
    for conn in connections['top_connections']:
        print(f"  {conn['target_node']}: {conn['probability']:.4f} ({conn['confidence']})")
    
    # 2. 代谢通路预测
    print("\n🛤️  2. 代谢通路预测演示")
    pathway_nodes = sample_nodes[:4]
    pathways = predictor.predict_pathway_from_nodes(pathway_nodes)
    
    print(f"输入节点: {pathways['input_nodes']}")
    print(f"有效节点: {pathways['valid_nodes']}")
    print(f"发现通路数: {pathways['statistics']['total_pathways']}")
    
    if pathways['pathways']:
        best_pathway = pathways['pathways'][0]
        print(f"最佳通路: {' → '.join(best_pathway['nodes'])}")
        print(f"通路得分: {best_pathway['score']:.4f}")
    
    # 3. 连接路径发现
    print("\n🔍 3. 连接路径发现演示")
    source, target = sample_nodes[0], sample_nodes[3]
    connecting_paths = predictor.find_connecting_nodes(source, target)
    
    print(f"源节点: {connecting_paths['source_node']}")
    print(f"目标节点: {connecting_paths['target_node']}")
    print(f"直接连接概率: {connecting_paths['direct_connection']['probability']:.4f}")
    print(f"发现路径数: {connecting_paths['statistics']['total_paths_found']}")
    
    if connecting_paths['connecting_paths']:
        best_path = connecting_paths['connecting_paths'][0]
        print(f"最佳路径: {' → '.join(best_path['path'])}")
        print(f"路径概率: {best_path['total_probability']:.4f}")
    
    # 4. 节点特征分析
    print("\n📊 4. 节点特征分析演示")
    node_analysis = predictor.analyze_node_features(sample_nodes[0])
    
    print(f"节点ID: {node_analysis['node_id']}")
    print("节点特征:")
    for feature, value in node_analysis['node_features'].items():
        print(f"  {feature}: {value}")
    
    print("连接性统计:")
    stats = node_analysis['connectivity_stats']
    print(f"  最大概率: {stats['max_probability']:.4f}")
    print(f"  平均概率: {stats['avg_probability']:.4f}")
    print(f"  连接数: {stats['connected_count']}")
    
    print("\n✅ 节点预测演示完成!")


if __name__ == "__main__":
    main()