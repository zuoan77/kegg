import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录的Path对象
    """
    # 从当前文件位置向上查找项目根目录
    current_file = Path(__file__).resolve()
    # 当前文件在 src/graph_construction/generate_abstract_connections.py
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


def generate_abstract_node_connections():
    # 读取RCU数据
    input_csv_path = resolve_path("output/dataprocess/rcu_nodes.csv")
    rcu_df = pd.read_csv(input_csv_path)
    print(f"加载RCU数据: {len(rcu_df)} 条记录")
    
    # 按molecular_diff分组
    molecular_groups = defaultdict(list)
    for _, row in rcu_df.iterrows():
        molecular_groups[row['molecular_diff']].append(row)
    
    print(f"发现 {len(molecular_groups)} 个不同的molecular_diff")
    
    # 生成抽象节点连接数据
    connections = []
    
    # 遍历每个molecular_diff组
    for from_molecular_diff, from_rcus in molecular_groups.items():
        from_node = f"MD_{from_molecular_diff}"
        
        # 1. 找到count>0的源RCU，收集它们产生的所有产物（包含重复）
        all_products_list = []  # 保留重复的产物列表
        
        for rcu in from_rcus:
            if rcu['count'] > 0:  # 只考虑有实际代谢通量的RCU
                products = eval(rcu['products'])
                for product in products:
                    all_products_list.append(product)
        
        if not all_products_list:
            continue  # 该抽象节点没有活跃的产物
            
        # 2. 为每个产物（包含重复）找到消费它的目标抽象节点
        for product in all_products_list:
            # 找到消费该产物的所有RCU
            consumers = rcu_df[rcu_df['substrates'].apply(lambda x: product in eval(x))]
            
            for idx, consumer_row in consumers.iterrows():
                to_molecular_diff = consumer_row['molecular_diff']
                to_node = f"MD_{to_molecular_diff}"
                to_rcu_id = consumer_row['rcu_id']
                
                # 找到产生该产物的源RCUs
                producing_rcus = []
                for rcu in from_rcus:
                    if rcu['count'] > 0 and product in eval(rcu['products']):
                        producing_rcus.append(rcu['rcu_id'])
                
                # 获取参与该连接的KO编号
                producing_kos = set()
                for rcu in from_rcus:
                    if rcu['count'] > 0 and product in eval(rcu['products']):
                        ko_numbers = eval(rcu['ko_numbers']) if rcu['ko_numbers'] != '[]' else []
                        producing_kos.update(ko_numbers)
                
                consuming_kos = eval(consumer_row['ko_numbers']) if consumer_row['ko_numbers'] != '[]' else []
                
                # 创建连接记录（每个产物-消费者对一条记录）
                connections.append({
                    'from_node': from_node,
                    'to_node': to_node,
                    'from_molecular_diff': from_molecular_diff,
                    'to_molecular_diff': to_molecular_diff,
                    'common_compounds': product,
                    'from_rcu_ids': ';'.join(producing_rcus),
                    'to_rcu_ids': to_rcu_id,
                    'from_ko_numbers': ';'.join(sorted(producing_kos)) if producing_kos else '',
                    'to_ko_numbers': ';'.join(consuming_kos) if consuming_kos else '',
                    'connection_strength': 1,  # 每个产物-消费者对记为1个连接强度
                    'compound_count': 1,
                    'ko_count': len(producing_kos) + len(consuming_kos)
                })
    
    # 保存连接数据
    if connections:
        connections_df = pd.DataFrame(connections)
        output_path = resolve_path("output/graph_builder_output/abstract_node_connections.csv")
        connections_df.to_csv(output_path, index=False)
        print(f"抽象节点连接表已保存: {output_path}")
        print(f"共 {len(connections)} 条连接")
        
        # 显示统计信息
        unique_edges = len(connections_df[['from_node', 'to_node']].drop_duplicates())
        print(f"唯一边数: {unique_edges}")
        print(f"重复边数: {len(connections) - unique_edges}")
    
    # 构建抽象节点的下一个节点映射，基于连接强度
    next_nodes_map = defaultdict(list)
    if connections:
        for conn in connections:
            from_node = conn['from_node']
            to_node = conn['to_node']
            connection_strength = conn['connection_strength']
            # 根据连接强度添加重复的连接
            for _ in range(connection_strength):
                next_nodes_map[from_node].append(to_node)
    
    # 生成抽象节点摘要
    node_summary = []
    for molecular_diff, rcus in molecular_groups.items():
        node_id = f"MD_{molecular_diff}"
        
        all_substrates = set()
        all_products = set() 
        all_ko_numbers = set()
        total_count = 0
        rcu_ids = []
        
        for rcu in rcus:
            all_substrates.update(eval(rcu['substrates']))
            all_products.update(eval(rcu['products']))
            ko_numbers = eval(rcu['ko_numbers']) if rcu['ko_numbers'] != '[]' else []
            all_ko_numbers.update(ko_numbers)
            total_count += rcu['count']
            rcu_ids.append(rcu['rcu_id'])
        
        # 获取下一个抽象节点（包含重复）
        next_nodes = next_nodes_map.get(node_id, [])
        
        # 统计每个下游节点的出现次数
        next_node_counts = Counter(next_nodes)
        
        # 格式化为 节点名_count数 的格式
        next_nodes_with_count = []
        for node, count in next_node_counts.items():
            next_nodes_with_count.append(f"{node}_{count}")
        next_nodes_str = ';'.join(sorted(next_nodes_with_count)) if next_nodes_with_count else ''
        
        # 获取唯一的下一个抽象节点
        unique_next_nodes = list(set(next_nodes))
        unique_next_nodes_str = ';'.join(sorted(unique_next_nodes)) if unique_next_nodes else ''
        
        node_summary.append({
            'node_id': node_id,
            'molecular_diff': molecular_diff,
            'rcu_count': len(rcus),
            'total_count': total_count,
            'substrate_count': len(all_substrates),
            'product_count': len(all_products),
            'ko_count': len(all_ko_numbers),
            'ko_numbers': ';'.join(sorted(all_ko_numbers)) if all_ko_numbers else '',
            'next_nodes': next_nodes_str,
            'unique_next_nodes': unique_next_nodes_str,
            'next_node_count': len(next_nodes),
            'unique_next_node_count': len(unique_next_nodes),
            'rcu_ids': ';'.join(rcu_ids),
            'substrates': ';'.join(sorted(all_substrates)),
            'products': ';'.join(sorted(all_products))
        })
    
    # 保存节点摘要
    if node_summary:
        summary_df = pd.DataFrame(node_summary)
        summary_path = resolve_path("output/graph_builder_output/abstract_nodes_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"抽象节点摘要已保存: {summary_path}")
        print(f"共 {len(node_summary)} 个抽象节点")

if __name__ == "__main__":
    generate_abstract_node_connections()
