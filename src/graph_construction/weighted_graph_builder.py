#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重图构建模块
基于抽象节点和边权重构建有向图结构
"""

import json
import pandas as pd
import networkx as nx
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedGraphBuilder:
    """权重图构建器"""
    
    @staticmethod
    def get_project_root() -> Path:
        """
        获取项目根目录
        
        Returns:
            项目根目录的Path对象
        """
        # 从当前文件位置向上查找项目根目录
        current_file = Path(__file__).resolve()
        # 当前文件在 src/graph_construction/weighted_graph_builder.py
        # 项目根目录在上两级
        project_root = current_file.parent.parent.parent
        return project_root
    
    def __init__(self, data_dir: str = None):
        """
        初始化图构建器
        
        Args:
            data_dir: 数据目录路径（支持相对路径和绝对路径）
                     相对路径将相对于项目根目录解析
                     默认为 "output/graph_builder_output"
        """
        if data_dir is None:
            # 使用默认的相对路径
            project_root = self.get_project_root()
            self.data_dir = project_root / "output" / "graph_builder_output"
        else:
            # 处理用户提供的路径
            path_obj = Path(data_dir)
            if path_obj.is_absolute():
                self.data_dir = path_obj
            else:
                # 相对路径相对于项目根目录
                project_root = self.get_project_root()
                self.data_dir = project_root / data_dir
        
        self.abstract_nodes_file = self.data_dir / "abstract_nodes_summary.csv"
        self.edge_weights_file = self.data_dir / "edge_weights.json"
        self.metabolic_graph_file = self.data_dir.parent / "metabolic_graph.pkl"
        
        # 图结构
        self.graph = nx.DiGraph()  # 有向图
        self.node_data = {}  # 节点详细信息
        self.edge_data = {}  # 边详细信息
        
    def load_abstract_nodes(self) -> pd.DataFrame:
        """
        加载抽象节点数据
        
        Returns:
            包含抽象节点信息的DataFrame
        """
        if not self.abstract_nodes_file.exists():
            raise FileNotFoundError(f"抽象节点文件不存在: {self.abstract_nodes_file}")
        
        logger.info(f"加载抽象节点数据: {self.abstract_nodes_file}")
        df = pd.read_csv(self.abstract_nodes_file)
        logger.info(f"加载了 {len(df)} 个抽象节点")
        
        return df
    
    def load_edge_weights(self) -> Dict:
        """
        加载边权重数据
        
        Returns:
            包含边权重信息的字典
        """
        if not self.edge_weights_file.exists():
            raise FileNotFoundError(f"边权重文件不存在: {self.edge_weights_file}")
        
        logger.info(f"加载边权重数据: {self.edge_weights_file}")
        with open(self.edge_weights_file, 'r', encoding='utf-8') as f:
            edge_weights = json.load(f)
        
        logger.info(f"加载了 {len(edge_weights)} 条边的权重")
        return edge_weights
    
    def load_ko_mappings(self) -> Dict:
        """
        从原始代谢图中加载KO映射信息
        
        Returns:
            包含抽象节点到KO编号映射的字典
        """
        if not self.metabolic_graph_file.exists():
            logger.warning(f"代谢图文件不存在: {self.metabolic_graph_file}")
            return {}
        
        logger.info(f"加载KO映射信息: {self.metabolic_graph_file}")
        with open(self.metabolic_graph_file, 'rb') as f:
            metabolic_graph = pickle.load(f)
        
        # 提取抽象节点到KO的映射
        ko_mappings = {}
        if hasattr(metabolic_graph, 'nodes'):
            for node in metabolic_graph.nodes():
                if isinstance(node, str) and node.startswith('MD_'):
                    # 从图中获取节点属性
                    node_attr = metabolic_graph.nodes[node]
                    if 'ko_list' in node_attr:
                        ko_mappings[node] = node_attr['ko_list']
                    elif 'ko' in node_attr:
                        ko_mappings[node] = [node_attr['ko']]
        
        logger.info(f"加载了 {len(ko_mappings)} 个抽象节点的KO映射")
        return ko_mappings
    
    def extract_molecular_formula(self, node_name: str) -> str:
        """
        从节点名称中提取分子式
        
        Args:
            node_name: 节点名称，如 "MD_C10H15N3O5S1"
            
        Returns:
            分子式字符串，如 "C10H15N3O5S1"
        """
        if node_name.startswith('MD_'):
            return node_name[3:]  # 去掉 "MD_" 前缀
        return node_name
    
    def add_nodes_to_graph(self, nodes_df: pd.DataFrame, ko_mappings: Dict = None):
        """
        将抽象节点添加到图中
        
        Args:
            nodes_df: 抽象节点DataFrame
            ko_mappings: KO映射字典（可选，从节点数据中获取）
        """
        logger.info("添加抽象节点到图中...")
        
        for _, row in nodes_df.iterrows():
            node_name = row['node_id']
            molecular_formula = self.extract_molecular_formula(node_name)
            
            # 从节点摘要数据中获取KO信息
            ko_list = []
            if pd.notna(row.get('ko_numbers', '')):
                ko_numbers_str = row['ko_numbers']
                if ko_numbers_str and ko_numbers_str != '':
                    ko_list = ko_numbers_str.split(';')
            
            # 节点属性
            node_attributes = {
                'molecular_formula': molecular_formula,
                'ko_list': ko_list,
                'ko_count': len(ko_list),
                'ko_numbers': ';'.join(ko_list) if ko_list else '',
                'total_count': row['total_count'],
                'unique_next_node_count': row['unique_next_node_count'],
                'rcu_count': row['rcu_count'],
                'substrate_count': row['substrate_count'],
                'product_count': row['product_count'],
                'substrates': row.get('substrates', ''),
                'products': row.get('products', ''),
                'node_type': 'abstract_molecular_difference'
            }
            
            # 添加节点到图
            self.graph.add_node(node_name, **node_attributes)
            
            # 处理next_nodes数据
            next_nodes = []
            if pd.notna(row['next_nodes']) and row['next_nodes']:
                try:
                    # 尝试解析为列表或处理为字符串
                    if row['next_nodes'].startswith('[') and row['next_nodes'].endswith(']'):
                        next_nodes = eval(row['next_nodes'])
                    else:
                        # 处理逗号分隔的字符串
                        next_nodes = [x.strip() for x in str(row['next_nodes']).split(',') if x.strip()]
                except:
                    # 如果解析失败，作为单个字符串处理
                    next_nodes = [str(row['next_nodes']).strip()]
            
            # 存储详细信息
            self.node_data[node_name] = {
                'name': node_name,
                'molecular_formula': molecular_formula,
                'ko_list': ko_list,
                'total_count': int(row['total_count']),
                'unique_next_node_count': int(row['unique_next_node_count']),
                'next_nodes': next_nodes
            }
        
        logger.info(f"成功添加 {len(self.node_data)} 个抽象节点")
    
    def add_edges_to_graph(self, edge_weights: Dict):
        """
        将权重边添加到图中
        
        Args:
            edge_weights: 边权重字典
        """
        logger.info("添加权重边到图中...")
        
        for edge_id, edge_info in edge_weights.items():
            source_node = edge_info['source_node']
            target_node = edge_info['target_node']
            weight = edge_info['weight']
            
            # 边属性 - 包含KO信息
            edge_attributes = {
                'weight': weight,
                'count_A_to_B': edge_info['count_A_to_B'],
                'out_A': edge_info['out_A'],
                'V_A': edge_info['V_A'],
                'ko_diversity': edge_info.get('ko_diversity', 0),
                'ko_numbers': edge_info.get('ko_numbers', []),
                'ko_factor': edge_info.get('ko_factor', 1.0),
                'formula': edge_info['formula'],
                'edge_type': 'molecular_transformation'
            }
            
            # 添加有向边
            self.graph.add_edge(source_node, target_node, **edge_attributes)
            
            # 存储详细信息
            self.edge_data[edge_id] = {
                'source': source_node,
                'target': target_node,
                'weight': weight,
                'count_A_to_B': edge_info['count_A_to_B'],
                'out_A': edge_info['out_A'],
                'V_A': edge_info['V_A'],
                'ko_diversity': edge_info.get('ko_diversity', 0),
                'ko_numbers': edge_info.get('ko_numbers', []),
                'ko_factor': edge_info.get('ko_factor', 1.0),
                'calculation_formula': edge_info['formula'],
                'calculation_steps': edge_info.get('calculation_steps', {})
            }
        
        logger.info(f"成功添加 {len(self.edge_data)} 条权重边")
    
    def build_graph(self) -> nx.DiGraph:
        """
        构建完整的权重图
        
        Returns:
            构建完成的有向图
        """
        logger.info("开始构建权重图...")
        
        # 1. 加载数据
        nodes_df = self.load_abstract_nodes()
        edge_weights = self.load_edge_weights()
        ko_mappings = self.load_ko_mappings()
        
        # 2. 构建图
        self.add_nodes_to_graph(nodes_df)
        self.add_edges_to_graph(edge_weights)
        
        # 3. 图统计信息
        logger.info("="*50)
        logger.info("图构建完成!")
        logger.info(f"节点数量: {self.graph.number_of_nodes()}")
        logger.info(f"边数量: {self.graph.number_of_edges()}")
        logger.info(f"平均出度: {sum(dict(self.graph.out_degree()).values()) / self.graph.number_of_nodes():.2f}")
        logger.info(f"平均入度: {sum(dict(self.graph.in_degree()).values()) / self.graph.number_of_nodes():.2f}")
        
        # 权重统计
        weights = [data['weight'] for _, _, data in self.graph.edges(data=True)]
        if weights:
            logger.info(f"边权重范围: {min(weights):.6f} ~ {max(weights):.6f}")
            logger.info(f"平均边权重: {sum(weights)/len(weights):.6f}")
        
        logger.info("="*50)
        
        return self.graph
    
    def get_node_info(self, node_name: str) -> Optional[Dict]:
        """
        获取节点详细信息
        
        Args:
            node_name: 节点名称
            
        Returns:
            节点信息字典
        """
        return self.node_data.get(node_name)
    
    def get_edge_info(self, source: str, target: str) -> Optional[Dict]:
        """
        获取边详细信息
        
        Args:
            source: 源节点
            target: 目标节点
            
        Returns:
            边信息字典
        """
        edge_id = f"{source} -> {target}"
        return self.edge_data.get(edge_id)
    
    def save_graph(self, output_file: str = None):
        """
        保存图到文件
        
        Args:
            output_file: 输出文件路径（支持相对路径和绝对路径）
                        相对路径将相对于项目根目录解析
        """
        if output_file is None:
            output_file = self.data_dir / "weighted_graph.pkl"
        else:
            # 处理用户提供的路径
            path_obj = Path(output_file)
            if not path_obj.is_absolute():
                # 相对路径相对于项目根目录
                project_root = self.get_project_root()
                output_file = project_root / output_file
            else:
                output_file = path_obj
        
        # 确保输出目录存在
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存图到: {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(self.graph, f)
        
        # 同时保存详细信息
        info_file = str(output_file).replace('.pkl', '_info.json')
        graph_info = {
            'nodes': self.node_data,
            'edges': self.edge_data,
            'statistics': {
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges(),
                'avg_out_degree': sum(dict(self.graph.out_degree()).values()) / self.graph.number_of_nodes(),
                'avg_in_degree': sum(dict(self.graph.in_degree()).values()) / self.graph.number_of_nodes()
            }
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(graph_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"图信息保存到: {info_file}")
    
    def get_subgraph_by_ko(self, ko_numbers: List[str], depth: int = 1) -> nx.DiGraph:
        """
        根据KO编号获取子图
        
        Args:
            ko_numbers: KO编号列表
            depth: 搜索深度
            
        Returns:
            子图
        """
        # 找到包含指定KO的节点
        target_nodes = set()
        for node_name, node_info in self.node_data.items():
            ko_list = node_info.get('ko_list', [])
            if any(ko in ko_list for ko in ko_numbers):
                target_nodes.add(node_name)
        
        # 扩展到指定深度
        all_nodes = set(target_nodes)
        current_nodes = target_nodes
        
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                # 添加后继节点
                next_nodes.update(self.graph.successors(node))
                # 添加前驱节点
                next_nodes.update(self.graph.predecessors(node))
            
            all_nodes.update(next_nodes)
            current_nodes = next_nodes
        
        return self.graph.subgraph(all_nodes).copy()
    
    def find_paths(self, source: str, target: str, max_length: int = 5) -> List[List[str]]:
        """
        找到两个节点之间的路径
        
        Args:
            source: 源节点
            target: 目标节点
            max_length: 最大路径长度
            
        Returns:
            路径列表
        """
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
            return paths
        except nx.NetworkXNoPath:
            return []


def main():
    """主函数 - 演示图构建过程"""
    # 创建图构建器
    builder = WeightedGraphBuilder()
    
    # 构建图
    graph = builder.build_graph()
    
    # 保存图
    builder.save_graph()
    
    # 演示一些基本操作
    print("\n=== 图分析示例 ===")
    
    # 查看某个节点的信息
    test_node = "MD_C10H15N3O5S1"
    if test_node in graph.nodes():
        node_info = builder.get_node_info(test_node)
        print(f"\n节点 {test_node} 信息:")
        print(f"  分子式: {node_info['molecular_formula']}")
        print(f"  KO列表: {node_info['ko_list']}")
        print(f"  总连接数: {node_info['total_count']}")
        print(f"  唯一后继节点数: {node_info['unique_next_node_count']}")
        
        # 查看出边
        out_edges = list(graph.out_edges(test_node, data=True))
        print(f"  出边数量: {len(out_edges)}")
        for source, target, data in out_edges:
            print(f"    -> {target}: weight={data['weight']}")
    
    # 查看度数最高的节点
    out_degrees = dict(graph.out_degree())
    top_nodes = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n出度最高的5个节点:")
    for node, degree in top_nodes:
        molecular_formula = builder.extract_molecular_formula(node)
        print(f"  {node} ({molecular_formula}): 出度={degree}")
    
    print(f"\n图构建完成！图文件保存到: {builder.data_dir}/weighted_graph.pkl")


if __name__ == "__main__":
    main()
