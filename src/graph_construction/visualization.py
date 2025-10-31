#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块
整合了简单可视化和高级可视化功能
"""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MetabolicGraphVisualizer:
    """代谢图可视化器"""
    
    @staticmethod
    def get_project_root() -> Path:
        """
        获取项目根目录
        
        Returns:
            项目根目录的Path对象
        """
        # 从当前文件位置向上查找项目根目录
        current_file = Path(__file__).resolve()
        # 当前文件在 src/graph_construction/visualization.py
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
            project_root = MetabolicGraphVisualizer.get_project_root()
            return project_root / file_path
    
    def __init__(self, output_dir: str = None):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录（支持相对路径和绝对路径），
                       相对路径将相对于项目根目录解析，
                       默认为output/graph_builder_output目录
        """
        if output_dir is None:
            # 默认输出目录设置为output/graph_builder_output
            project_root = self.get_project_root()
            self.output_dir = project_root / "output" / "graph_builder_output"
        else:
            self.output_dir = self.resolve_path(output_dir)
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"可视化器初始化完成，输出目录: {self.output_dir}")
    
    def visualize_graph_simple(self, graph: nx.DiGraph, output_path: str = None) -> str:
        """
        生成简单的静态图可视化
        
        Args:
            graph: NetworkX图对象
            output_path: 输出文件路径（支持相对路径和绝对路径），
                        相对路径将相对于项目根目录解析，默认自动生成
            
        Returns:
            生成的文件路径
        """
        if output_path is None:
            output_path = self.output_dir / "metabolic_graph_simple.png"
        else:
            output_path = self.resolve_path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            plt.figure(figsize=(12, 8))
            
            # 如果节点太多，使用spring布局
            if graph.number_of_nodes() > 100:
                pos = nx.spring_layout(graph, k=1, iterations=50)
                node_size = 20
                font_size = 6
            else:
                pos = nx.spring_layout(graph)
                node_size = 50
                font_size = 8
            
            # 绘制图
            nx.draw(graph, pos, 
                   node_size=node_size, 
                   alpha=0.6,
                   arrows=True, 
                   arrowsize=10, 
                   edge_color='gray',
                   node_color='lightblue',
                   font_size=font_size)
            
            plt.title(f"代谢网络图 ({graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边)")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"简单静态图已生成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"生成简单静态图失败: {e}")
            raise
    
    def visualize_graph_interactive_pyvis(self, graph: nx.DiGraph, output_path: str = None) -> str:
        """
        使用pyvis生成交互式图
        
        Args:
            graph: NetworkX图对象
            output_path: 输出文件路径（支持相对路径和绝对路径），
                        相对路径将相对于项目根目录解析，默认自动生成
            
        Returns:
            生成的文件路径
        """
        if output_path is None:
            output_path = self.output_dir / "metabolic_graph_interactive.html"
        else:
            output_path = self.resolve_path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            from pyvis.network import Network
            
            # 创建pyvis网络
            net = Network(
                height="600px", 
                width="100%", 
                bgcolor="#ffffff", 
                font_color="black",
                notebook=False
            )
            
            # 添加节点
            for node in graph.nodes():
                # 根据节点度数设置大小
                node_size = max(5, min(25, graph.degree(node) * 2))
                
                # 获取节点属性
                node_attr = graph.nodes[node]
                
                # 构建节点标题信息
                title_parts = [f"Node: {node}"]
                title_parts.append(f"Degree: {graph.degree(node)}")
                
                # 添加KO信息
                ko_list = node_attr.get('ko_list', [])
                if ko_list:
                    ko_count = len(ko_list)
                    ko_display = ', '.join(ko_list[:5])  # 只显示前5个
                    if ko_count > 5:
                        ko_display += f"... (+{ko_count-5} more)"
                    title_parts.append(f"KO Numbers ({ko_count}): {ko_display}")
                
                # 添加分子式
                molecular_formula = node_attr.get('molecular_formula', '')
                if molecular_formula:
                    title_parts.append(f"Molecular Diff: {molecular_formula}")
                
                # 添加RCU信息
                rcu_count = node_attr.get('rcu_count', 0)
                total_count = node_attr.get('total_count', 0)
                if rcu_count > 0:
                    title_parts.append(f"RCU Count: {rcu_count}")
                    title_parts.append(f"Total Count: {total_count}")
                
                # 设置节点颜色 - 根据KO数量
                ko_count = len(ko_list)
                if ko_count == 0:
                    color = "#ffcccc"  # 浅红色 - 无KO
                elif ko_count <= 2:
                    color = "#ffeb9c"  # 浅黄色 - 少量KO
                elif ko_count <= 5:
                    color = "#c5e1a5"  # 浅绿色 - 中等KO
                else:
                    color = "#81c784"  # 绿色 - 大量KO
                
                net.add_node(node, 
                           label=str(node)[:20],  # 限制标签长度
                           size=node_size,
                           color=color,
                           title="\n".join(title_parts))
            
            # 添加边
            for source, target in graph.edges():
                # 获取边权重（如果有的话）
                edge_data = graph[source][target]
                weight = edge_data.get('weight', 1.0)
                
                # 构建边标题信息
                title_parts = [f"{source} -> {target}"]
                title_parts.append(f"Weight: {weight:.4f}")
                
                # 添加KO信息
                ko_numbers = edge_data.get('ko_numbers', [])
                if ko_numbers:
                    ko_count = len(ko_numbers)
                    ko_display = ', '.join(ko_numbers[:3])  # 只显示前3个
                    if ko_count > 3:
                        ko_display += f"... (+{ko_count-3} more)"
                    title_parts.append(f"KO Numbers ({ko_count}): {ko_display}")
                
                # 添加KO因子
                ko_factor = edge_data.get('ko_factor', 1.0)
                if ko_factor != 1.0:
                    title_parts.append(f"KO Factor: {ko_factor:.4f}")
                
                # 添加统计信息
                count_a_to_b = edge_data.get('count_A_to_B', 0)
                if count_a_to_b > 0:
                    title_parts.append(f"Connection Count: {count_a_to_b}")
                
                net.add_edge(source, target, 
                           width=max(1, weight * 5),  # 根据权重设置边宽度
                           title="\n".join(title_parts))
            
            # 设置物理特性
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
              }
            }
            """)
            
            # 保存文件
            net.save_graph(str(output_path))
            
            logger.info(f"交互式图已生成: {output_path}")
            return str(output_path)
            
        except ImportError:
            logger.warning("pyvis未安装，无法生成交互式图")
            raise ImportError("需要安装pyvis: pip install pyvis")
        except Exception as e:
            logger.error(f"生成交互式图失败: {e}")
            raise
    
    def generate_stats_html(self, graph: nx.DiGraph, output_path: str = None) -> str:
        """
        生成图统计信息的HTML页面
        
        Args:
            graph: NetworkX图对象
            output_path: 输出文件路径，默认自动生成
            
        Returns:
            生成的文件路径
        """
        if output_path is None:
            output_path = self.output_dir / "metabolic_graph_stats.html"
        else:
            output_path = Path(output_path)
        
        try:
            # 计算图统计信息
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            
            # 度统计
            degrees = dict(graph.degree())
            avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
            max_degree = max(degrees.values()) if degrees else 0
            
            # 连通分量
            if graph.is_directed():
                weak_components = list(nx.weakly_connected_components(graph))
                num_components = len(weak_components)
                largest_component_size = len(max(weak_components, key=len)) if weak_components else 0
                component_type = "弱连通分量"
            else:
                components = list(nx.connected_components(graph))
                num_components = len(components)
                largest_component_size = len(max(components, key=len)) if components else 0
                component_type = "连通分量"
            
            # 边权重统计（如果有的话）
            edge_weights = []
            for u, v, data in graph.edges(data=True):
                if 'weight' in data:
                    edge_weights.append(data['weight'])
            
            weight_stats = ""
            if edge_weights:
                weight_stats = f"""
                <h3>边权重统计</h3>
                <p><strong>边权重数量:</strong> {len(edge_weights)}</p>
                <p><strong>平均权重:</strong> {sum(edge_weights)/len(edge_weights):.6f}</p>
                <p><strong>最小权重:</strong> {min(edge_weights):.6f}</p>
                <p><strong>最大权重:</strong> {max(edge_weights):.6f}</p>
                """
            
            # 生成HTML内容
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>代谢网络图统计</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 40px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        background-color: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        max-width: 800px;
                        margin: 0 auto;
                    }}
                    h1 {{
                        color: #2c3e50;
                        border-bottom: 3px solid #3498db;
                        padding-bottom: 10px;
                    }}
                    h2, h3 {{
                        color: #34495e;
                        margin-top: 25px;
                    }}
                    p {{
                        margin: 8px 0;
                        line-height: 1.6;
                    }}
                    .stat-box {{
                        background-color: #ecf0f1;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 10px 0;
                    }}
                    .highlight {{
                        color: #e74c3c;
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>代谢网络图统计信息</h1>
                    
                    <div class="stat-box">
                        <h2>基本统计</h2>
                        <p><strong>节点数量:</strong> <span class="highlight">{num_nodes}</span></p>
                        <p><strong>边数量:</strong> <span class="highlight">{num_edges}</span></p>
                        <p><strong>图类型:</strong> {'有向图' if graph.is_directed() else '无向图'}</p>
                    </div>
                    
                    <div class="stat-box">
                        <h2>度统计</h2>
                        <p><strong>平均度:</strong> {avg_degree:.2f}</p>
                        <p><strong>最大度:</strong> {max_degree}</p>
                    </div>
                    
                    <div class="stat-box">
                        <h2>连通性</h2>
                        <p><strong>{component_type}数:</strong> {num_components}</p>
                        <p><strong>最大{component_type}大小:</strong> {largest_component_size}</p>
                    </div>
                    
                    {weight_stats}
                    
                    <div class="stat-box">
                        <h2>生成时间</h2>
                        <p><strong>生成时间:</strong> {Path(__file__).stat().st_mtime}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # 保存HTML文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"统计页面已生成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"生成统计页面失败: {e}")
            raise
    
    def visualize_all(self, graph: nx.DiGraph) -> Dict[str, str]:
        """
        生成所有类型的可视化（不包括PNG图片）
        
        Args:
            graph: NetworkX图对象
            
        Returns:
            生成的文件路径字典
        """
        results = {}
        
        try:
            # 1. 交互式图
            logger.info("生成交互式图...")
            interactive_path = self.visualize_graph_interactive_pyvis(graph)
            results['interactive'] = interactive_path
            
        except Exception as e:
            logger.error(f"生成交互式图失败: {e}")
            results['interactive'] = f"失败: {e}"
        
        try:
            # 2. 统计页面
            logger.info("生成统计页面...")
            stats_path = self.generate_stats_html(graph)
            results['stats'] = stats_path
            
        except Exception as e:
            logger.error(f"生成统计页面失败: {e}")
            results['stats'] = f"失败: {e}"
        
        return results
    
    def quick_visualize(self, graph: nx.DiGraph, viz_type: str = 'all') -> Dict[str, str]:
        """
        快速可视化
        
        Args:
            graph: NetworkX图对象
            viz_type: 可视化类型 ('interactive', 'stats', 'all')
            
        Returns:
            生成的文件路径字典
        """
        if viz_type == 'all':
            return self.visualize_all(graph)
        
        results = {}
        
        if viz_type == 'interactive':
            try:
                path = self.visualize_graph_interactive_pyvis(graph)
                results['interactive'] = path
            except Exception as e:
                results['interactive'] = f"失败: {e}"
        
        elif viz_type == 'stats':
            try:
                path = self.generate_stats_html(graph)
                results['stats'] = path
            except Exception as e:
                results['stats'] = f"失败: {e}"
        
        else:
            logger.error(f"不支持的可视化类型: {viz_type}")
            results['error'] = f"不支持的类型: {viz_type}"
        
        return results


def create_visualizer(output_dir: str = None) -> MetabolicGraphVisualizer:
    """
    创建可视化器实例
    
    Args:
        output_dir: 输出目录
        
    Returns:
        可视化器实例
    """
    return MetabolicGraphVisualizer(output_dir)


# 向后兼容的简单接口
class Visualization:
    """向后兼容的简单可视化类"""
    
    def __init__(self, output_dir: str = None):
        self.visualizer = MetabolicGraphVisualizer(output_dir)
    
    def visualize_graph(self, graph: nx.DiGraph, output_path: str):
        """简单图可视化（向后兼容）"""
        return self.visualizer.visualize_graph_simple(graph, output_path)
    
    def visualize_graph_interactive(self, graph: nx.DiGraph, output_path: str):
        """交互式图可视化（向后兼容）"""
        return self.visualizer.visualize_graph_interactive_pyvis(graph, output_path)


if __name__ == "__main__":
    # 测试代码
    import networkx as nx
    
    # 创建测试图
    G = nx.erdos_renyi_graph(20, 0.3, directed=True)
    
    # 添加权重
    for u, v in G.edges():
        G[u][v]['weight'] = abs(hash(f"{u}{v}")) % 100 / 100.0
    
    # 创建可视化器
    viz = MetabolicGraphVisualizer()
    
    # 生成所有可视化
    results = viz.visualize_all(G)
    
    print("可视化生成完成:")
    for name, path in results.items():
        print(f"  {name}: {path}")
