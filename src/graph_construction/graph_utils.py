#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图工具函数模块
包含图操作的通用工具函数
"""

import pandas as pd
import networkx as nx
import pickle
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class GraphUtils:
    """图工具类"""
    
    @staticmethod
    def get_project_root() -> Path:
        """
        获取项目根目录
        
        Returns:
            项目根目录的Path对象
        """
        # 从当前文件位置向上查找项目根目录
        current_file = Path(__file__).resolve()
        # 当前文件在 src/graph_construction/graph_utils.py
        # 项目根目录在上两级
        project_root = current_file.parent.parent.parent
        return project_root
    
    @staticmethod
    def resolve_path(file_path: str) -> Path:
        """
        解析路径，支持相对路径和绝对路径
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析后的Path对象
        """
        path_obj = Path(file_path)
        if path_obj.is_absolute():
            return path_obj
        else:
            # 相对路径相对于项目根目录
            project_root = GraphUtils.get_project_root()
            return project_root / file_path
    
    @staticmethod
    def save_graph(graph: nx.DiGraph, file_path: str):
        """
        保存图到pickle文件
        
        Args:
            graph: 要保存的图
            file_path: 保存路径（支持相对路径和绝对路径）
                      相对路径将相对于项目根目录解析
        """
        resolved_path = GraphUtils.resolve_path(file_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(resolved_path, 'wb') as f:
            pickle.dump(graph, f)
        
        logger.info(f"图已保存到: {resolved_path}")
    
    @staticmethod
    def load_graph(file_path: str) -> nx.DiGraph:
        """
        从pickle文件加载图
        
        Args:
            file_path: 图文件路径（支持相对路径和绝对路径）
                      相对路径将相对于项目根目录解析
            
        Returns:
            加载的图
        """
        resolved_path = GraphUtils.resolve_path(file_path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"图文件不存在: {resolved_path}")
        
        with open(resolved_path, 'rb') as f:
            graph = pickle.load(f)
        
        logger.info(f"图已加载: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
        return graph
    
    @staticmethod
    def export_to_csv(graph: nx.DiGraph, 
                     nodes_file: str, 
                     edges_file: str):
        """
        导出图为CSV格式
        
        Args:
            graph: 要导出的图
            nodes_file: 节点CSV文件路径（支持相对路径和绝对路径）
            edges_file: 边CSV文件路径（支持相对路径和绝对路径）
                       相对路径将相对于项目根目录解析
        """
        nodes_path = GraphUtils.resolve_path(nodes_file)
        edges_path = GraphUtils.resolve_path(edges_file)
        
        # 确保目录存在
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        edges_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 导出节点
        nodes_data = []
        for node, attrs in graph.nodes(data=True):
            node_data = {'node_id': node}
            node_data.update(attrs)
            nodes_data.append(node_data)
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(nodes_path, index=False)
        logger.info(f"节点数据已导出到: {nodes_path}")
        
        # 导出边
        edges_data = []
        for source, target, attrs in graph.edges(data=True):
            edge_data = {
                'source': source,
                'target': target
            }
            edge_data.update(attrs)
            edges_data.append(edge_data)
        
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(edges_path, index=False)
        logger.info(f"边数据已导出到: {edges_path}")
    
    @staticmethod
    def export_to_json(graph: nx.DiGraph, file_path: str):
        """
        导出图为JSON格式
        
        Args:
            graph: 要导出的图
            file_path: JSON文件路径（支持相对路径和绝对路径）
                      相对路径将相对于项目根目录解析
        """
        resolved_path = GraphUtils.resolve_path(file_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为JSON兼容的格式
        graph_data = {
            'nodes': [
                {'id': node, **attrs} 
                for node, attrs in graph.nodes(data=True)
            ],
            'edges': [
                {'source': source, 'target': target, **attrs}
                for source, target, attrs in graph.edges(data=True)
            ]
        }
        
        with open(resolved_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"图已导出到JSON文件: {resolved_path}")
    
    @staticmethod
    def get_graph_statistics(graph: nx.DiGraph) -> Dict[str, Any]:
        """
        获取图的统计信息
        
        Args:
            graph: 要分析的图
            
        Returns:
            统计信息字典
        """
        # 基本统计
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        # 度统计
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]
        
        # 连通性
        weakly_connected = list(nx.weakly_connected_components(graph))
        strongly_connected = list(nx.strongly_connected_components(graph))
        
        # 权重统计（如果有）
        weights = []
        for _, _, data in graph.edges(data=True):
            if 'weight' in data:
                weights.append(data['weight'])
        
        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_in_degree': sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            'avg_out_degree': sum(out_degrees) / len(out_degrees) if out_degrees else 0,
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'num_weakly_connected_components': len(weakly_connected),
            'num_strongly_connected_components': len(strongly_connected),
            'density': nx.density(graph),
        }
        
        if weights:
            stats.update({
                'num_weighted_edges': len(weights),
                'avg_weight': sum(weights) / len(weights),
                'min_weight': min(weights),
                'max_weight': max(weights),
            })
        
        # 孤立节点
        isolated = list(nx.isolates(graph))
        stats['num_isolated_nodes'] = len(isolated)
        
        # 最大连通分量大小
        if weakly_connected:
            largest_component = max(weakly_connected, key=len)
            stats['largest_component_size'] = len(largest_component)
        else:
            stats['largest_component_size'] = 0
        
        return stats
    
    @staticmethod
    def filter_nodes_by_degree(graph: nx.DiGraph, 
                              min_degree: int = 0, 
                              max_degree: int = None,
                              degree_type: str = 'total') -> List[str]:
        """
        根据度数过滤节点
        
        Args:
            graph: 图对象
            min_degree: 最小度数
            max_degree: 最大度数，None表示无限制
            degree_type: 度数类型 ('in', 'out', 'total')
            
        Returns:
            符合条件的节点列表
        """
        filtered_nodes = []
        
        for node in graph.nodes():
            if degree_type == 'in':
                degree = graph.in_degree(node)
            elif degree_type == 'out':
                degree = graph.out_degree(node)
            else:  # total
                degree = graph.degree(node)
            
            if degree >= min_degree:
                if max_degree is None or degree <= max_degree:
                    filtered_nodes.append(node)
        
        return filtered_nodes
    
    @staticmethod
    def get_hub_nodes(graph: nx.DiGraph, 
                     top_n: int = 10, 
                     metric: str = 'out_degree') -> List[Tuple[str, float]]:
        """
        获取图中的hub节点（高度连接的节点）
        
        Args:
            graph: 图对象
            top_n: 返回的节点数量
            metric: 评估指标 ('out_degree', 'in_degree', 'total_degree', 'betweenness', 'closeness')
            
        Returns:
            [(节点, 指标值), ...] 按指标值降序排列
        """
        if metric == 'out_degree':
            scores = dict(graph.out_degree())
        elif metric == 'in_degree':
            scores = dict(graph.in_degree())
        elif metric == 'total_degree':
            scores = dict(graph.degree())
        elif metric == 'betweenness':
            scores = nx.betweenness_centrality(graph)
        elif metric == 'closeness':
            scores = nx.closeness_centrality(graph)
        else:
            raise ValueError(f"不支持的指标: {metric}")
        
        # 按分数排序
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_nodes[:top_n]
    
    @staticmethod
    def find_shortest_paths(graph: nx.DiGraph, 
                           source: str, 
                           target: str = None,
                           max_paths: int = 10) -> List[List[str]]:
        """
        查找最短路径
        
        Args:
            graph: 图对象
            source: 源节点
            target: 目标节点，None表示查找从源节点到所有节点的最短路径
            max_paths: 最大返回路径数
            
        Returns:
            路径列表，每个路径是节点列表
        """
        if target is not None:
            # 查找特定源-目标对的最短路径
            try:
                if nx.has_path(graph, source, target):
                    path = nx.shortest_path(graph, source, target)
                    return [path]
                else:
                    return []
            except nx.NetworkXNoPath:
                return []
        else:
            # 查找从源节点到所有可达节点的最短路径
            try:
                paths = nx.single_source_shortest_path(graph, source)
                # 按路径长度排序，返回前max_paths个
                sorted_paths = sorted(paths.items(), key=lambda x: len(x[1]))
                return [path for _, path in sorted_paths[:max_paths]]
            except:
                return []
    
    @staticmethod
    def get_node_neighborhoods(graph: nx.DiGraph, 
                              node: str, 
                              radius: int = 1,
                              direction: str = 'both') -> nx.DiGraph:
        """
        获取节点的邻域子图
        
        Args:
            graph: 图对象
            node: 中心节点
            radius: 邻域半径
            direction: 方向 ('in', 'out', 'both')
            
        Returns:
            邻域子图
        """
        if node not in graph:
            raise ValueError(f"节点不存在: {node}")
        
        # 收集邻域节点
        neighborhood = {node}
        current_layer = {node}
        
        for _ in range(radius):
            next_layer = set()
            
            for n in current_layer:
                if direction in ['out', 'both']:
                    next_layer.update(graph.successors(n))
                if direction in ['in', 'both']:
                    next_layer.update(graph.predecessors(n))
            
            neighborhood.update(next_layer)
            current_layer = next_layer
            
            if not current_layer:  # 没有更多邻居
                break
        
        # 创建子图
        subgraph = graph.subgraph(neighborhood).copy()
        return subgraph
    
    @staticmethod
    def detect_cycles(graph: nx.DiGraph) -> List[List[str]]:
        """
        检测图中的环
        
        Args:
            graph: 图对象
            
        Returns:
            环列表，每个环是节点列表
        """
        try:
            cycles = list(nx.simple_cycles(graph))
            return cycles
        except:
            return []
    
    @staticmethod
    def calculate_node_importance(graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """
        计算节点重要性指标
        
        Args:
            graph: 图对象
            
        Returns:
            节点重要性字典 {节点: {指标: 值, ...}}
        """
        importance = {}
        
        # 度中心性
        in_degree_centrality = nx.in_degree_centrality(graph)
        out_degree_centrality = nx.out_degree_centrality(graph)
        
        # 介数中心性（对于大图可能很慢）
        try:
            betweenness_centrality = nx.betweenness_centrality(graph)
        except:
            betweenness_centrality = {}
        
        # 接近中心性
        try:
            closeness_centrality = nx.closeness_centrality(graph)
        except:
            closeness_centrality = {}
        
        # PageRank
        try:
            pagerank = nx.pagerank(graph)
        except:
            pagerank = {}
        
        # 合并所有指标
        for node in graph.nodes():
            importance[node] = {
                'in_degree_centrality': in_degree_centrality.get(node, 0),
                'out_degree_centrality': out_degree_centrality.get(node, 0),
                'betweenness_centrality': betweenness_centrality.get(node, 0),
                'closeness_centrality': closeness_centrality.get(node, 0),
                'pagerank': pagerank.get(node, 0),
            }
        
        return importance
    
    @staticmethod
    def merge_graphs(graphs: List[nx.DiGraph], 
                    node_attr_strategy: str = 'first',
                    edge_attr_strategy: str = 'sum') -> nx.DiGraph:
        """
        合并多个图
        
        Args:
            graphs: 要合并的图列表
            node_attr_strategy: 节点属性合并策略 ('first', 'last', 'merge')
            edge_attr_strategy: 边属性合并策略 ('first', 'last', 'sum', 'max', 'min')
            
        Returns:
            合并后的图
        """
        if not graphs:
            return nx.DiGraph()
        
        merged = nx.DiGraph()
        
        # 合并节点
        for graph in graphs:
            for node, attrs in graph.nodes(data=True):
                if node in merged:
                    # 处理属性冲突
                    if node_attr_strategy == 'last':
                        merged.nodes[node].update(attrs)
                    elif node_attr_strategy == 'merge':
                        for key, value in attrs.items():
                            if key in merged.nodes[node]:
                                if isinstance(value, (int, float)) and isinstance(merged.nodes[node][key], (int, float)):
                                    merged.nodes[node][key] += value
                                elif isinstance(value, list):
                                    if isinstance(merged.nodes[node][key], list):
                                        merged.nodes[node][key].extend(value)
                                    else:
                                        merged.nodes[node][key] = [merged.nodes[node][key], value]
                            else:
                                merged.nodes[node][key] = value
                else:
                    merged.add_node(node, **attrs)
        
        # 合并边
        for graph in graphs:
            for source, target, attrs in graph.edges(data=True):
                if merged.has_edge(source, target):
                    # 处理属性冲突
                    existing_attrs = merged[source][target]
                    
                    if edge_attr_strategy == 'last':
                        existing_attrs.update(attrs)
                    elif edge_attr_strategy == 'sum':
                        for key, value in attrs.items():
                            if key in existing_attrs and isinstance(value, (int, float)):
                                existing_attrs[key] += value
                            else:
                                existing_attrs[key] = value
                    elif edge_attr_strategy == 'max':
                        for key, value in attrs.items():
                            if key in existing_attrs and isinstance(value, (int, float)):
                                existing_attrs[key] = max(existing_attrs[key], value)
                            else:
                                existing_attrs[key] = value
                    elif edge_attr_strategy == 'min':
                        for key, value in attrs.items():
                            if key in existing_attrs and isinstance(value, (int, float)):
                                existing_attrs[key] = min(existing_attrs[key], value)
                            else:
                                existing_attrs[key] = value
                else:
                    merged.add_edge(source, target, **attrs)
        
        return merged


# 便捷函数
def load_graph(file_path: str) -> nx.DiGraph:
    """加载图的便捷函数"""
    return GraphUtils.load_graph(file_path)


def save_graph(graph: nx.DiGraph, file_path: str):
    """保存图的便捷函数"""
    return GraphUtils.save_graph(graph, file_path)


def get_graph_stats(graph: nx.DiGraph) -> Dict[str, Any]:
    """获取图统计信息的便捷函数"""
    return GraphUtils.get_graph_statistics(graph)


if __name__ == "__main__":
    # 测试代码
    print("GraphUtils 工具模块测试")
    
    # 创建示例图
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B', {'weight': 0.5}), 
                      ('B', 'C', {'weight': 0.8}), 
                      ('C', 'A', {'weight': 0.3})])
    
    # 测试统计信息
    stats = GraphUtils.get_graph_statistics(G)
    print("图统计信息:", stats)
    
    # 测试hub节点
    hubs = GraphUtils.get_hub_nodes(G, top_n=3)
    print("Hub节点:", hubs)
