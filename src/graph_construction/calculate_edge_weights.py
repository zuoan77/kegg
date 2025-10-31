import pandas as pd
import json
import math
from collections import defaultdict
from pathlib import Path


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录的Path对象
    """
    # 从当前文件位置向上查找项目根目录
    current_file = Path(__file__).resolve()
    # 当前文件在 src/graph_construction/calculate_edge_weights.py
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


def calculate_edge_weights():
    """
    计算抽象节点间的边权重，考虑KO编号频率
    增强公式: w(A→B) = (count(A→B) + 1) / (out(A) + V_A) × count / (count + 2) × ko_factor
    其中 ko_factor = (ko_diversity + 1) / 2，ko_diversity为参与该连接的KO种类数
    """
    
    # 读取抽象节点摘要数据
    summary_path = resolve_path("output/graph_builder_output/abstract_nodes_summary.csv")
    df = pd.read_csv(summary_path)
    
    # 读取连接数据，获取KO信息
    connections_path = resolve_path("output/graph_builder_output/abstract_node_connections.csv")
    connections_df = pd.read_csv(connections_path)
    
    print(f"读取到 {len(df)} 个抽象节点")
    print(f"读取到 {len(connections_df)} 条连接")
    
    # 为每个边聚合KO信息
    edge_ko_info = defaultdict(set)
    for _, conn in connections_df.iterrows():
        edge_key = (conn['from_node'], conn['to_node'])
        
        # 添加来源KO
        if pd.notna(conn.get('from_ko_numbers', '')) and conn.get('from_ko_numbers', '') != '':
            from_kos = conn['from_ko_numbers'].split(';')
            edge_ko_info[edge_key].update(from_kos)
        
        # 添加目标KO
        if pd.notna(conn.get('to_ko_numbers', '')) and conn.get('to_ko_numbers', '') != '':
            to_kos = conn['to_ko_numbers'].split(';')
            edge_ko_info[edge_key].update(to_kos)
    
    # 存储所有边的权重信息
    edge_weights = {}
    total_edges = 0
    
    for idx, row in df.iterrows():
        node_a = row['node_id']
        next_nodes_str = row['next_nodes']
        unique_next_nodes_str = row['unique_next_nodes']
        total_count = row['total_count']  # out(A)
        unique_next_node_count = row['unique_next_node_count']  # V_A
        
        if pd.isna(next_nodes_str) or next_nodes_str == '':
            continue  # 该节点没有出边
            
        # 解析next_nodes，提取目标节点和连接强度
        next_nodes = next_nodes_str.split(';')
        
        for next_node_info in next_nodes:
            if '_' not in next_node_info:
                continue
                
            # 分离节点名和连接强度
            parts = next_node_info.rsplit('_', 1)
            node_b = parts[0]
            count_a_to_b = int(parts[1])  # count(A→B)
            
            # 构建边的标识符
            edge_id = f"{node_a} -> {node_b}"
            
            # 获取该边的KO信息
            edge_key = (node_a, node_b)
            ko_set = edge_ko_info.get(edge_key, set())
            ko_diversity = len(ko_set)
            ko_factor = (ko_diversity + 1) / 2  # KO多样性因子
            
            # 计算权重公式的各个部分
            # w(A→B) = (count(A→B) + 1) / (out(A) + V_A) × count / (count + 2) × ko_factor
            
            part1 = (count_a_to_b + 1) / (total_count + unique_next_node_count)
            part2 = count_a_to_b / (count_a_to_b + 2)
            weight = part1 * part2 * ko_factor
            
            # 将权重保留小数点后五位
            weight_rounded = round(weight, 5)
            
            # 存储详细信息
            edge_weights[edge_id] = {
                "source_node": node_a,
                "target_node": node_b,
                "count_A_to_B": count_a_to_b,
                "out_A": total_count,
                "V_A": unique_next_node_count,
                "ko_diversity": ko_diversity,
                "ko_numbers": list(ko_set),
                "ko_factor": round(ko_factor, 4),
                "formula": f"({count_a_to_b} + 1) / ({total_count} + {unique_next_node_count}) × {count_a_to_b} / ({count_a_to_b} + 2) × {ko_factor:.4f}",
                "calculation_steps": {
                    "part1": f"({count_a_to_b} + 1) / ({total_count} + {unique_next_node_count}) = {part1:.6f}",
                    "part2": f"{count_a_to_b} / ({count_a_to_b} + 2) = {part2:.6f}",
                    "ko_factor": f"({ko_diversity} + 1) / 2 = {ko_factor:.4f}",
                    "final": f"{part1:.6f} × {part2:.6f} × {ko_factor:.4f} = {weight:.6f}"
                },
                "weight": weight_rounded
            }
            
            total_edges += 1
            
        print(f"处理节点 {node_a}: {len(next_nodes)} 条出边")
    
    print(f"总共计算了 {total_edges} 条边的权重")
    
    # 保存权重信息到JSON文件
    output_path = resolve_path("output/graph_builder_output/edge_weights.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(edge_weights, f, indent=2, ensure_ascii=False)
    
    print(f"边权重信息已保存到: {output_path}")
    
    # 统计信息
    weights = [info['weight'] for info in edge_weights.values()]
    print(f"权重统计:")
    print(f"  最小权重: {min(weights):.6f}")
    print(f"  最大权重: {max(weights):.6f}")
    print(f"  平均权重: {sum(weights)/len(weights):.6f}")
    
    # 显示几个示例
    print(f"\n前5个边的权重示例:")
    for i, (edge_id, info) in enumerate(edge_weights.items()):
        if i >= 5:
            break
        print(f"  {edge_id}:")
        print(f"    count(A→B): {info['count_A_to_B']}")
        print(f"    out(A): {info['out_A']}")
        print(f"    V_A: {info['V_A']}")
        print(f"    KO多样性: {info['ko_diversity']} ({', '.join(info['ko_numbers'][:5])}{'...' if len(info['ko_numbers']) > 5 else ''})")
        print(f"    KO因子: {info['ko_factor']}")
        print(f"    公式: {info['formula']}")
        print(f"    权重: {info['weight']:.6f}")
        print()

if __name__ == "__main__":
    calculate_edge_weights()
