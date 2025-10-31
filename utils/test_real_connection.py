#!/usr/bin/env python3
"""
真实边测试工具
用于验证代谢网络中分子差异节点之间的直接连接关系
使用方法：
    python test_real_connection.py  MD_C6H12N2O1  MD_H3N1O1
"""

import pandas as pd
import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录的Path对象
    """
    # 从当前文件位置向上查找项目根目录
    current_file = Path(__file__).resolve()
    # 当前文件在 utils/test_real_connection.py
    # 项目根目录在上一级
    project_root = current_file.parent.parent
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


class RealConnectionTester:
    """真实连接测试器 - 简化版，只检查直接连接"""
    
    def __init__(self, data_dir: str = "output"):
        """
        初始化测试器
        
        Args:
            data_dir: 数据目录路径（支持相对路径和绝对路径）
        """
        # 解析数据目录路径
        self.data_dir = resolve_path(data_dir)
        self.connections_file = self.data_dir / "graph_builder_output" / "abstract_node_connections.csv"
        
        # 加载数据
        self.connections_df = None
        self._load_data()
    
    def _load_data(self):
        """加载连接数据文件"""
        try:
            if self.connections_file.exists():
                self.connections_df = pd.read_csv(self.connections_file)
                print(f"✓ 加载连接数据: {len(self.connections_df)} 条记录")
            else:
                print(f"✗ 连接文件不存在: {self.connections_file}")
                
        except Exception as e:
            print(f"✗ 数据加载错误: {e}")
    
    def check_direct_connection(self, node1: str, node2: str) -> Dict:
        """
        检查两个节点之间是否存在直接连接（有向图）
        
        Args:
            node1: 第一个节点ID (格式: MD_C20H34N7O14P3S1)
            node2: 第二个节点ID (格式: MD_C20H34N7O14P3S1)
            
        Returns:
            连接检查结果，包含方向性信息
        """
        if self.connections_df is None:
            return {
                'node1': node1,
                'node2': node2,
                'error': '连接数据未加载',
                'has_direct_connection': False
            }
        
        # 提取分子差异部分（去掉MD_前缀）
        mol1 = node1.replace('MD_', '') if node1.startswith('MD_') else node1
        mol2 = node2.replace('MD_', '') if node2.startswith('MD_') else node2
        
        results = {
            'node1': node1,
            'node2': node2,
            'molecule1': mol1,
            'molecule2': mol2,
            'has_direct_connection': False,
            'connection_type': 'none',  # none, forward, reverse, bidirectional
            'connection_details': []
        }
        
        # 分别检查正向连接 (node1 -> node2)
        forward_connections = self.connections_df[
            (self.connections_df['from_molecular_diff'] == mol1) & 
            (self.connections_df['to_molecular_diff'] == mol2)
        ]
        
        # 分别检查反向连接 (node2 -> node1)
        reverse_connections = self.connections_df[
            (self.connections_df['from_molecular_diff'] == mol2) & 
            (self.connections_df['to_molecular_diff'] == mol1)
        ]
        
        has_forward = not forward_connections.empty
        has_reverse = not reverse_connections.empty
        
        if has_forward or has_reverse:
            results['has_direct_connection'] = True
            
            # 确定连接类型
            if has_forward and has_reverse:
                results['connection_type'] = 'bidirectional'
            elif has_forward:
                results['connection_type'] = 'forward'
            else:
                results['connection_type'] = 'reverse'
            
            # 添加正向连接信息（如果存在）
            if has_forward:
                row = forward_connections.iloc[0]
                results['connection_details'].append({
                    'direction': 'forward',
                    'direction_text': f"{node1} -> {node2}",
                    'from_mol': row['from_molecular_diff'],
                    'to_mol': row['to_molecular_diff'],
                    'common_compounds': row['common_compounds'],
                    'connection_strength': int(row['connection_strength']),
                    'compound_count': int(row['compound_count']),
                    'ko_count': int(row['ko_count']),
                    'connections_in_direction': int(len(forward_connections))
                })
            
            # 添加反向连接信息（如果存在）
            if has_reverse:
                row = reverse_connections.iloc[0]
                results['connection_details'].append({
                    'direction': 'reverse',
                    'direction_text': f"{node2} -> {node1}",
                    'from_mol': row['from_molecular_diff'],
                    'to_mol': row['to_molecular_diff'],
                    'common_compounds': row['common_compounds'],
                    'connection_strength': int(row['connection_strength']),
                    'compound_count': int(row['compound_count']),
                    'ko_count': int(row['ko_count']),
                    'connections_in_direction': int(len(reverse_connections))
                })
            
            # 添加总连接数信息
            total_connections = len(forward_connections) + len(reverse_connections)
            results['total_connections_found'] = total_connections
        
        return results

    def check_self_loop(self, node: str) -> Dict:
        """
        检查单个节点的自环连接
        
        Args:
            node: 节点ID (格式: MD_C20H34N7O14P3S1)
            
        Returns:
            自环连接检查结果
        """
        if self.connections_df is None:
            return {
                'node': node,
                'error': '连接数据未加载',
                'has_self_loop': False
            }
        
        # 提取分子差异部分（去掉MD_前缀）
        mol = node.replace('MD_', '') if node.startswith('MD_') else node
        
        results = {
            'node': node,
            'molecule': mol,
            'has_self_loop': False,
            'self_loop_details': []
        }
        
        # 搜索自环连接 (from_molecular_diff == to_molecular_diff)
        self_loops = self.connections_df[
            (self.connections_df['from_molecular_diff'] == mol) & 
            (self.connections_df['to_molecular_diff'] == mol)
        ]
        
        if not self_loops.empty:
            results['has_self_loop'] = True
            results['total_self_loops'] = len(self_loops)
            
            # 只显示第一个自环连接的详情
            row = self_loops.iloc[0]
            results['self_loop_details'].append({
                'direction_text': f"{node} -> {node}",
                'from_mol': row['from_molecular_diff'],
                'to_mol': row['to_molecular_diff'],
                'common_compounds': row['common_compounds'],
                'connection_strength': int(row['connection_strength']),
                'compound_count': int(row['compound_count']),
                'ko_count': int(row['ko_count'])
            })
        
        return results

    def batch_check_self_loops(self, nodes: List[str]) -> Dict:
        """
        批量检查多个节点的自环连接
        
        Args:
            nodes: 节点ID列表
            
        Returns:
            批量自环检查结果，格式为 {node_id: result_dict}
        """
        results = {}
        
        for node in nodes:
            loop_result = self.check_self_loop(node)
            results[node] = loop_result
        
        return results

    def has_self_loop(self, node: str) -> bool:
        """
        简单的自循环判断函数，只返回True或False
        
        Args:
            node: 节点ID (格式: MD_C20H34N7O14P3S1)
            
        Returns:
            bool: 如果节点存在自循环返回True，否则返回False
        """
        if self.connections_df is None:
            return False
        
        # 提取分子差异部分（去掉MD_前缀）
        mol = node.replace('MD_', '') if node.startswith('MD_') else node
        
        # 搜索自环连接 (from_molecular_diff == to_molecular_diff)
        self_loops = self.connections_df[
            (self.connections_df['from_molecular_diff'] == mol) & 
            (self.connections_df['to_molecular_diff'] == mol)
        ]
        
        return not self_loops.empty


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='真实边连接测试工具 - 简化版')
    parser.add_argument('--data-dir', default='output', help='数据目录路径')
    parser.add_argument('node1', help='第一个节点ID (格式: MD_C20H34N7O14P3S1)')
    parser.add_argument('node2', nargs='?', help='第二个节点ID (格式: MD_C20H34N7O14P3S1)，如果与node1相同则检查自环')
    parser.add_argument('--output', help='输出结果到JSON文件')
    
    args = parser.parse_args()
    
    # 初始化测试器
    tester = RealConnectionTester(args.data_dir)
    
    # 判断是自环测试还是直接连接测试
    if args.node2 is None or args.node1 == args.node2:
        # 自环测试
        print(f"\n=== 检查自环连接: {args.node1} ===")
        
        result = tester.check_self_loop(args.node1)
        
        if 'error' in result:
            print(f"错误: {result['error']}")
            return
        
        print(f"节点: {result['node']} (分子: {result['molecule']})")
        print(f"存在自环连接: {'是' if result['has_self_loop'] else '否'}")
        
        if result['has_self_loop']:
            total_loops = result.get('total_self_loops', 0)
            print(f"自环连接数: {total_loops}")
            
            # 显示自环连接详情
            for i, loop in enumerate(result['self_loop_details']):
                print(f"\n--- 自环连接 {i+1}: {loop['direction_text']} ---")
                print(f"  连接强度: {loop['connection_strength']}")
                print(f"  共同化合物: {loop['common_compounds']}")
                print(f"  化合物数量: {loop['compound_count']}")
                print(f"  酶数量: {loop['ko_count']}")
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")
    
    else:
        # 直接连接测试
        print(f"\n=== 检查直接连接: {args.node1} <-> {args.node2} ===")
        
        result = tester.check_direct_connection(args.node1, args.node2)
        
        if 'error' in result:
            print(f"错误: {result['error']}")
            return
        
        print(f"节点1: {result['node1']} (分子: {result['molecule1']})")
        print(f"节点2: {result['node2']} (分子: {result['molecule2']})")
        print(f"存在直接连接: {'是' if result['has_direct_connection'] else '否'}")
        
        if result['has_direct_connection']:
            connection_type = result['connection_type']
            total_connections = result.get('total_connections_found', 0)
            
            # 显示连接类型
            type_text = {
                'forward': '单向连接 (正向)',
                'reverse': '单向连接 (反向)', 
                'bidirectional': '双向连接'
            }
            print(f"连接类型: {type_text.get(connection_type, connection_type)}")
            print(f"总连接数: {total_connections}")
            
            # 显示每个方向的连接详情
            for i, conn in enumerate(result['connection_details']):
                print(f"\n--- 连接 {i+1}: {conn['direction_text']} ---")
                print(f"  方向: {conn['direction']}")
                print(f"  连接强度: {conn['connection_strength']}")
                print(f"  共同化合物: {conn['common_compounds']}")
                print(f"  化合物数量: {conn['compound_count']}")
                print(f"  酶数量: {conn['ko_count']}")
                print(f"  该方向连接数: {conn['connections_in_direction']}")
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()