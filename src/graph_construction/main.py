#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图构建主入口
抽象节点生成 -> 权重计算 -> 图构建 -> 可视化
"""

import os
import sys
from pathlib import Path
import logging
import networkx as nx

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 直接导入各个功能模块的函数和管理器
from generate_abstract_connections import generate_abstract_node_connections
from calculate_edge_weights import calculate_edge_weights
from weighted_graph_builder import WeightedGraphBuilder
from visualization import MetabolicGraphVisualizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录的Path对象
    """
    # 从当前文件位置向上查找项目根目录
    current_file = Path(__file__).resolve()
    # 当前文件在 src/graph_construction/main.py
    # 项目根目录在上两级
    project_root = current_file.parent.parent.parent
    return project_root


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
        project_root = get_project_root()
        return project_root / file_path


def main():
    """
    主函数：完成代谢网络图构建的完整流程
    步骤1：生成抽象节点连接
    步骤2：计算边权重  
    步骤3：构建图结构
    步骤4：生成可视化
    """
    try:
        print("="*60)
        print("代谢网络图构建流程开始")
        print("="*60)
        
        # 创建必要的输出目录
        output_dirs = [
            "output",
            "output/graph_builder_output"
        ]
        for dir_path in output_dirs:
            resolved_path = resolve_path(dir_path)
            resolved_path.mkdir(parents=True, exist_ok=True)
        print("✅ 输出目录已创建")
        
        # 步骤1：生成抽象节点连接
        print("\n🔄 步骤1：生成抽象节点连接...")
        logger.info("开始生成抽象节点连接")
        generate_abstract_node_connections()
        print("✅ 抽象节点连接生成完成")
        
        # 步骤2：计算边权重
        print("\n🔄 步骤2：计算边权重...")
        logger.info("开始计算边权重")
        edge_weights = calculate_edge_weights()
        print("✅ 边权重计算完成")
        
        # 步骤3：构建图结构
        print("\n🔄 步骤3：构建图结构...")
        logger.info("开始构建图结构")
        
        # 使用加权图构建器构建完整图
        weighted_builder = WeightedGraphBuilder()
        graph = weighted_builder.build_graph()
        
        # 输出图统计信息
        print(f"✅ 图构建完成")
        print(f"   节点数量: {graph.number_of_nodes()}")
        print(f"   边数量: {graph.number_of_edges()}")
        print(f"   是否为有向图: {graph.is_directed()}")
        
        # 分析图的连通性
        if graph.is_directed():
            # 有向图分析
            weak_components = list(nx.weakly_connected_components(graph))
            print(f"   弱连通分量数: {len(weak_components)}")
            print(f"   最大弱连通分量大小: {len(max(weak_components, key=len)) if weak_components else 0}")
        else:
            # 无向图分析
            components = list(nx.connected_components(graph))
            print(f"   连通分量数: {len(components)}")
            print(f"   最大连通分量大小: {len(max(components, key=len)) if components else 0}")
        
        # 步骤4：生成可视化
        print("\n🔄 步骤4：生成可视化...")
        logger.info("开始生成可视化")
        
        # 创建可视化器，指定输出到graph_builder_output目录
        output_dir = "output/graph_builder_output"
        visualizer = MetabolicGraphVisualizer(output_dir=output_dir)
        
        try:
            # 使用新的可视化模块生成所有类型的可视化
            viz_results = visualizer.visualize_all(graph)
            
            print("✅ 可视化生成完成")
            print("\n📊 生成的可视化文件:")
            for viz_name, file_path in viz_results.items():
                if not file_path.startswith("失败"):
                    print(f"   {viz_name}: {file_path}")
                else:
                    print(f"   {viz_name}: {file_path}")
                    
        except Exception as e:
            print(f"   ❌ 可视化生成失败: {e}")
            logger.warning(f"可视化失败，尝试简单备用方案: {e}")
            
            # 备用方案：生成简单的HTML统计页面
            try:
                output_dir = resolve_path("output/graph_builder_output")
                
                # 生成简单的统计HTML
                stats_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>代谢网络图统计</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        .stat {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
                    </style>
                </head>
                <body>
                    <h1>代谢网络图统计信息</h1>
                    <div class="stat"><strong>节点数量:</strong> {graph.number_of_nodes()}</div>
                    <div class="stat"><strong>边数量:</strong> {graph.number_of_edges()}</div>
                    <div class="stat"><strong>平均出度:</strong> {sum(dict(graph.out_degree()).values()) / graph.number_of_nodes():.2f}</div>
                    <div class="stat"><strong>平均入度:</strong> {sum(dict(graph.in_degree()).values()) / graph.number_of_nodes():.2f}</div>
                    <div class="stat"><strong>弱连通分量数:</strong> {len(list(nx.weakly_connected_components(graph)))}</div>
                    <div class="stat"><strong>最大连通分量大小:</strong> {len(max(nx.weakly_connected_components(graph), key=len))}</div>
                </body>
                </html>
                """
                
                fallback_path = output_dir / "metabolic_graph_fallback.html"
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    f.write(stats_html)
                print(f"   ✅ 备用统计页面已生成: {fallback_path}")
                
            except Exception as backup_e:
                print(f"   ❌ 备用方案也失败: {backup_e}")
                logger.error(f"所有可视化方案都失败: {backup_e}")
        print("✅ 可视化生成完成")
        
        # 总结
        print("\n" + "="*60)
        print("🎉 代谢网络图构建流程全部完成！")
        print("="*60)
        print("\n📊 最终结果:")
        print(f"   图结构: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
        
        # 检查生成的可视化文件
        output_dir = resolve_path("output/graph_builder_output")
        possible_files = [
            ("交互式图", "metabolic_graph_interactive.html"),
            ("统计页面", "metabolic_graph_stats.html"),
            ("备用统计页面", "metabolic_graph_fallback.html")
        ]
        
        found_files = []
        for name, filename in possible_files:
            file_path = output_dir / filename
            if file_path.exists():
                found_files.append((name, str(file_path)))
        
        if found_files:
            print("\n🎨 生成的可视化文件:")
            for name, path in found_files:
                print(f"   {name}: {path}")
        
        print(f"\n📁 输出目录: {output_dir}")
        
        return graph
        
    except Exception as e:
        logger.error(f"图构建流程中发生错误: {e}")
        print(f"❌ 错误: {e}")
        raise


if __name__ == "__main__":
    main()
