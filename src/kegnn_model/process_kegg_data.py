#!/usr/bin/env python3
"""
专门处理KEGG真实数据的脚本
处理边权重不匹配和重复边问题
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import argparse

def process_kegg_data(nodes_file, edges_file, weights_file, output_dir="kegg_processed"):
    """处理KEGG数据，解决边权重不匹配问题"""
    
    print("🔧 处理KEGG真实数据...")
    print("=" * 50)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. 读取数据
    print("📖 读取数据文件...")
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    
    with open(weights_file, 'r') as f:
        weights_data = json.load(f)
    
    print(f"   节点数: {len(nodes_df)}")
    print(f"   边数: {len(edges_df)}")
    print(f"   权重数: {len(weights_data)}")
    
    # 2. 分析重复边
    print("\n🔍 分析重复边...")
    edge_pairs = edges_df[['from_node', 'to_node']].copy()
    duplicates = edge_pairs.duplicated(keep=False)
    duplicate_count = duplicates.sum()
    unique_edges = len(edges_df) - duplicate_count + len(edge_pairs.drop_duplicates())
    
    print(f"   总边数: {len(edges_df)}")
    print(f"   重复边数: {duplicate_count}")
    print(f"   唯一边数: {unique_edges}")
    
    # 3. 处理权重匹配
    print("\n⚖️ 处理权重匹配...")
    
    # 创建边到权重的映射
    edge_weights = {}
    
    # 从权重文件中提取权重
    for edge_key, weight_info in weights_data.items():
        if isinstance(weight_info, dict) and 'weight' in weight_info:
            weight = weight_info['weight']
        else:
            weight = 1.0  # 默认权重
        
        # 解析边键
        if ' -> ' in edge_key:
            source, target = edge_key.split(' -> ')
            edge_weights[(source, target)] = weight
    
    # 4. 为所有边分配权重
    print("🔗 为边分配权重...")
    processed_edges = []
    
    for _, row in edges_df.iterrows():
        source = row['from_node']
        target = row['to_node']
        
        # 查找权重
        weight = edge_weights.get((source, target), 1.0)
        
        processed_edges.append({
            'from_node': source,
            'to_node': target,
            'weight': weight
        })
    
    processed_edges_df = pd.DataFrame(processed_edges)
    
    # 5. 处理重复边 - 使用平均权重
    print("🔄 处理重复边（使用平均权重）...")
    final_edges = processed_edges_df.groupby(['from_node', 'to_node']).agg({
        'weight': 'mean'
    }).reset_index()
    
    print(f"   处理后边数: {len(final_edges)}")
    
    # 6. 创建权重字典
    final_weights = {}
    for _, row in final_edges.iterrows():
        edge_key = f"{row['from_node']} -> {row['to_node']}"
        final_weights[edge_key] = {
            'source_node': row['from_node'],
            'target_node': row['to_node'],
            'weight': float(row['weight'])
        }
    
    # 7. 保存处理后的数据
    print("\n💾 保存处理后的数据...")
    
    # 保存节点
    nodes_output = output_path / "processed_nodes.csv"
    nodes_df.to_csv(nodes_output, index=False)
    
    # 保存边
    edges_output = output_path / "processed_edges.csv"
    final_edges.to_csv(edges_output, index=False)
    
    # 保存权重
    weights_output = output_path / "processed_weights.json"
    with open(weights_output, 'w') as f:
        json.dump(final_weights, f, indent=2)
    
    # 保存处理报告
    report = {
        "original_data": {
            "nodes": len(nodes_df),
            "edges": len(edges_df),
            "weights": len(weights_data)
        },
        "processed_data": {
            "nodes": len(nodes_df),
            "edges": len(final_edges),
            "weights": len(final_weights)
        },
        "duplicate_edges": {
            "total_duplicates": int(duplicate_count),
            "merge_strategy": "mean"
        },
        "output_files": {
            "nodes": str(nodes_output),
            "edges": str(edges_output),
            "weights": str(weights_output)
        }
    }
    
    report_output = output_path / "processing_report.json"
    with open(report_output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ 数据处理完成！")
    print(f"   输出目录: {output_path}")
    print(f"   节点文件: {nodes_output}")
    print(f"   边文件: {edges_output}")
    print(f"   权重文件: {weights_output}")
    print(f"   报告文件: {report_output}")
    
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description="处理KEGG真实数据")
    parser.add_argument("--nodes", required=False, help="节点文件路径", default="../../output/graph_builder_output/abstract_nodes_summary.csv")
    parser.add_argument("--edges", required=False, help="边文件路径", default="../../output/graph_builder_output/abstract_node_connections.csv")
    parser.add_argument("--weights", required=False, help="权重文件路径", default="../../output/graph_builder_output/edge_weights.json")
    parser.add_argument("--output", default="kegg_real_processed", help="输出目录")
    
    args = parser.parse_args()
    
    process_kegg_data(args.nodes, args.edges, args.weights, args.output)

if __name__ == "__main__":
    main()