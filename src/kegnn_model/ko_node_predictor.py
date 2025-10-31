#!/usr/bin/env python3
"""
KO序列到节点预测的完整流程
整合pathway_discovery功能，支持从KO编号到代谢通路预测

主要功能：
1. KO序列输入和验证
2. KO到抽象节点的映射
3. 基于GNN模型的节点连接预测
4. 代谢通路推理和路径优化
5. 结果可视化和报告生成
"""

import sys
import os
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
import itertools
import argparse

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入必要模块
from node_predictor import KEGGNodePredictor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KONodePredictor:
    """KO序列到节点预测的完整流程"""
    
    def __init__(self, 
                 model_path: str = None,
                 data_dir: str = None,
                 abstract_nodes_file: str = None,
                 abstract_connections_file: str = None):
        """
        初始化KO节点预测器
        
        Args:
            model_path: GNN模型路径
            data_dir: KEGG数据目录
            abstract_nodes_file: 抽象节点摘要文件
            abstract_connections_file: 抽象节点连接文件
        """
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent
        
        # 设置默认路径（使用绝对路径）
        self.model_path = model_path or str(current_dir / "kegg_training_results" / "best_model.pth")
        self.data_dir = data_dir or str(current_dir / "kegg_real_processed")
        self.abstract_nodes_file = abstract_nodes_file or str(current_dir.parent.parent / "output" / "graph_builder_output" / "abstract_nodes_summary.csv")
        self.abstract_connections_file = abstract_connections_file or str(current_dir.parent.parent / "output" / "graph_builder_output" / "abstract_node_connections.csv")
        
        logger.info("🚀 初始化KO节点预测器...")
        logger.info(f"   模型路径: {self.model_path}")
        logger.info(f"   数据目录: {self.data_dir}")
        logger.info(f"   抽象节点文件: {self.abstract_nodes_file}")
        
        # 初始化组件
        self._load_node_predictor()
        self._load_abstract_data()
        self._build_ko_mapping()
        
    def _load_node_predictor(self):
        """加载节点预测器"""
        try:
            logger.info("🔧 加载节点预测器...")
            self.node_predictor = KEGGNodePredictor(self.model_path, self.data_dir)
            logger.info("✅ 节点预测器加载成功")
        except Exception as e:
            logger.error(f"❌ 节点预测器加载失败: {e}")
            raise
    
    def _load_abstract_data(self):
        """加载节点数据"""
        try:
            logger.info("📊 加载节点数据...")
            
            # 使用实际的节点数据而不是抽象节点数据
            nodes_file = os.path.join(self.data_dir, "processed_nodes.csv")
            self.nodes_df = pd.read_csv(nodes_file)
            logger.info(f"✅ 节点数据: {len(self.nodes_df)} 个节点")
            
            # 加载边数据
            edges_file = os.path.join(self.data_dir, "processed_edges.csv")
            self.edges_df = pd.read_csv(edges_file)
            logger.info(f"✅ 边数据: {len(self.edges_df)} 个边")
            
        except Exception as e:
            logger.error(f"❌ 节点数据加载失败: {e}")
            raise
    
    def _build_ko_mapping(self):
        """构建KO到节点的映射"""
        logger.info("🗺️  构建KO映射...")
        
        self.ko_to_nodes = {}
        self.node_to_kos = {}
        
        for _, row in self.nodes_df.iterrows():
            node_id = row['node_id']
            ko_numbers = str(row['ko_numbers']).split(';') if pd.notna(row['ko_numbers']) else []
            
            # 清理KO编号
            ko_numbers = [ko.strip() for ko in ko_numbers if ko.strip()]
            
            self.node_to_kos[node_id] = ko_numbers
            
            for ko in ko_numbers:
                if ko not in self.ko_to_nodes:
                    self.ko_to_nodes[ko] = []
                self.ko_to_nodes[ko].append(node_id)
        
        logger.info(f"✅ KO映射构建完成: {len(self.ko_to_nodes)} 个KO编号")
        logger.info(f"✅ 节点映射构建完成: {len(self.node_to_kos)} 个节点")
    
    def predict_from_ko_sequence(self, 
                                ko_sequence: List[str],
                                max_combinations: int = 100,
                                min_probability: float = 0.1,
                                top_k_pathways: int = 10) -> Dict:
        """
        从KO序列预测代谢通路
        
        Args:
            ko_sequence: KO编号序列
            max_combinations: 最大组合数
            min_probability: 最小边概率阈值
            top_k_pathways: 返回前K个最佳通路
            
        Returns:
            预测结果字典
        """
        logger.info("🎯 开始KO序列预测")
        logger.info(f"📋 输入KO序列: {ko_sequence}")
        
        # 1. KO到抽象节点映射
        mapping_result = self._map_kos_to_abstract_nodes(ko_sequence)
        
        if mapping_result['error']:
            return mapping_result
        
        # 2. 生成节点组合
        node_combinations = self._generate_node_combinations(
            mapping_result['ko_mappings'], max_combinations
        )
        
        # 3. 对每个组合进行通路预测
        pathway_results = []
        
        for i, combination in enumerate(node_combinations):
            logger.info(f"🔄 处理组合 {i+1}/{len(node_combinations)}: {combination}")
            
            # 预测该组合的通路
            pathway_result = self.node_predictor.predict_pathway_from_nodes(
                combination, min_probability=min_probability
            )
            # 收集该组合下的所有通路到全局结果
            if pathway_result.get('pathways'):
                for pathway in pathway_result['pathways']:
                    pathway_results.append({
                        'combination': combination,
                        'pathway': pathway,
                        'ko_mapping': {node: self.node_to_kos.get(node, []) for node in combination},
                        'edge_predictions': pathway_result.get('edge_predictions', {})
                    })
        
        # 4. 排序和筛选最佳通路
        pathway_results.sort(key=lambda x: x['pathway']['score'], reverse=True)
        top_pathways = pathway_results[:top_k_pathways]
        
        # 5. 生成综合结果
        result = {
            'input_ko_sequence': ko_sequence,
            'ko_mapping_summary': mapping_result,
            'total_combinations_tested': len(node_combinations),
            'total_pathways_found': len(pathway_results),
            'top_pathways': top_pathways,
            'all_pathways': pathway_results,
            'statistics': {
                'valid_kos': len(mapping_result['valid_kos']),
                'invalid_kos': len(mapping_result['invalid_kos']),
                'mapped_nodes': len(mapping_result['all_mapped_nodes']),
                'avg_pathway_score': np.mean([p['pathway']['score'] for p in pathway_results]) if pathway_results else 0,
                'max_pathway_score': max([p['pathway']['score'] for p in pathway_results]) if pathway_results else 0
            }
        }
        
        return result
    
    def _map_kos_to_abstract_nodes(self, ko_sequence: List[str]) -> Dict:
        """将KO序列映射到抽象节点"""
        logger.info("🗺️  映射KO到抽象节点...")
        
        valid_kos = []
        invalid_kos = []
        ko_mappings = {}
        all_mapped_nodes = set()
        
        for ko in ko_sequence:
            if ko in self.ko_to_nodes:
                mapped_nodes = self.ko_to_nodes[ko]
                ko_mappings[ko] = mapped_nodes
                all_mapped_nodes.update(mapped_nodes)
                valid_kos.append(ko)
                logger.info(f"  ✅ {ko} → {mapped_nodes}")
            else:
                invalid_kos.append(ko)
                logger.warning(f"  ❌ {ko} 未找到映射")
        
        if not valid_kos:
            return {
                'error': '没有找到有效的KO映射',
                'invalid_kos': invalid_kos,
                'valid_kos': [],
                'ko_mappings': {},
                'all_mapped_nodes': []
            }
        
        return {
            'error': None,
            'valid_kos': valid_kos,
            'invalid_kos': invalid_kos,
            'ko_mappings': ko_mappings,
            'all_mapped_nodes': list(all_mapped_nodes)
        }
    
    def _generate_node_combinations(self, ko_mappings: Dict, max_combinations: int) -> List[List[str]]:
        """生成节点组合"""
        logger.info("🔄 生成节点组合...")
        
        # 获取每个KO对应的节点列表
        node_lists = list(ko_mappings.values())
        
        # 生成笛卡尔积组合
        combinations = list(itertools.product(*node_lists))
        
        # 限制组合数量
        if len(combinations) > max_combinations:
            logger.warning(f"⚠️  组合数量过多 ({len(combinations)})，限制为 {max_combinations}")
            combinations = combinations[:max_combinations]
        
        # 转换为列表格式并去重
        unique_combinations = []
        seen = set()
        
        for combo in combinations:
            # 去除重复节点并排序
            unique_nodes = sorted(list(set(combo)))
            combo_key = tuple(unique_nodes)
            
            if combo_key not in seen and len(unique_nodes) >= 2:
                seen.add(combo_key)
                unique_combinations.append(unique_nodes)
        
        logger.info(f"✅ 生成 {len(unique_combinations)} 个唯一组合")
        return unique_combinations
    
    def analyze_ko_coverage(self, ko_sequence: List[str]) -> Dict:
        """分析KO序列的覆盖情况"""
        logger.info("📊 分析KO覆盖情况...")
        
        coverage_stats = {
            'total_kos': len(ko_sequence),
            'mapped_kos': 0,
            'unmapped_kos': 0,
            'total_nodes': 0,
            'ko_details': {},
            'node_details': {}
        }
        
        for ko in ko_sequence:
            if ko in self.ko_to_nodes:
                mapped_nodes = self.ko_to_nodes[ko]
                coverage_stats['mapped_kos'] += 1
                coverage_stats['total_nodes'] += len(mapped_nodes)
                
                coverage_stats['ko_details'][ko] = {
                    'status': 'mapped',
                    'mapped_nodes': mapped_nodes,
                    'node_count': len(mapped_nodes)
                }
                
                # 收集节点详细信息
                for node in mapped_nodes:
                    if node not in coverage_stats['node_details']:
                        node_info = self.nodes_df[
                            self.nodes_df['node_id'] == node
                        ].iloc[0] if len(self.nodes_df[
                            self.nodes_df['node_id'] == node
                        ]) > 0 else None
                        
                        if node_info is not None:
                            coverage_stats['node_details'][node] = {
                                'ko_numbers': str(node_info['ko_numbers']).split(';'),
                                'substrates': str(node_info['substrates']).split(';'),
                                'products': str(node_info['products']).split(';'),
                                'rcu_count': int(node_info['rcu_count']),
                                'total_count': int(node_info['total_count'])
                            }
            else:
                coverage_stats['unmapped_kos'] += 1
                coverage_stats['ko_details'][ko] = {
                    'status': 'unmapped',
                    'mapped_nodes': [],
                    'node_count': 0
                }
        
        coverage_stats['mapping_rate'] = coverage_stats['mapped_kos'] / coverage_stats['total_kos']
        
        return coverage_stats
    
    def find_ko_pathways(self, 
                        ko_sequence: List[str],
                        pathway_length_range: Tuple[int, int] = (2, 6),
                        min_edge_probability: float = 0.2) -> Dict:
        """
        寻找KO序列的最优代谢通路
        
        Args:
            ko_sequence: KO编号序列
            pathway_length_range: 通路长度范围 (最小, 最大)
            min_edge_probability: 最小边概率阈值
            
        Returns:
            通路发现结果
        """
        logger.info("🔍 寻找KO代谢通路...")
        
        # 获取映射结果
        mapping_result = self._map_kos_to_abstract_nodes(ko_sequence)
        
        if mapping_result['error']:
            return mapping_result
        
        # 获取所有映射的节点
        all_nodes = mapping_result['all_mapped_nodes']
        
        if len(all_nodes) < pathway_length_range[0]:
            return {
                'error': f'映射节点数量 ({len(all_nodes)}) 少于最小通路长度 ({pathway_length_range[0]})'
            }
        
        # 寻找不同长度的通路
        pathways_by_length = {}
        
        for length in range(pathway_length_range[0], min(pathway_length_range[1] + 1, len(all_nodes) + 1)):
            logger.info(f"🔄 寻找长度为 {length} 的通路...")
            
            # 生成该长度的所有节点组合
            from itertools import combinations
            node_combinations = list(combinations(all_nodes, length))
            
            pathways = []
            
            for combo in node_combinations[:50]:  # 限制组合数量
                combo_list = list(combo)
                pathway_result = self.node_predictor.predict_pathway_from_nodes(
                    combo_list, min_probability=min_edge_probability
                )
                
                if pathway_result.get('pathways'):
                    for pathway in pathway_result['pathways']:
                        pathways.append({
                            'nodes': pathway['nodes'],
                            'edges': pathway['edges'],
                            'score': pathway['score'],
                            'length': pathway['length'],
                            'ko_mapping': {node: self.node_to_kos.get(node, []) for node in pathway['nodes']}
                        })
            
            # 排序并保留最佳通路
            pathways.sort(key=lambda x: x['score'], reverse=True)
            pathways_by_length[length] = pathways[:10]
        
        # 找到全局最佳通路
        all_pathways = []
        for pathways in pathways_by_length.values():
            all_pathways.extend(pathways)
        
        all_pathways.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'input_ko_sequence': ko_sequence,
            'mapping_summary': mapping_result,
            'pathways_by_length': pathways_by_length,
            'best_pathways': all_pathways[:10],
            'statistics': {
                'total_pathways_found': len(all_pathways),
                'avg_pathway_score': np.mean([p['score'] for p in all_pathways]) if all_pathways else 0,
                'max_pathway_score': max([p['score'] for p in all_pathways]) if all_pathways else 0,
                'pathway_lengths': list(pathways_by_length.keys())
            }
        }
    
    def generate_prediction_report(self, result: Dict, output_file: Optional[str] = None) -> str:
        """生成预测结果报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = f"""# KO序列节点预测报告
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 输入信息
- **KO序列**: {', '.join(result['input_ko_sequence'])}
- **序列长度**: {len(result['input_ko_sequence'])}

## 映射统计
- **有效KO**: {result['statistics']['valid_kos']} / {len(result['input_ko_sequence'])}
- **映射节点数**: {result['statistics']['mapped_nodes']}
- **映射成功率**: {result['statistics']['valid_kos'] / len(result['input_ko_sequence']) * 100:.1f}%

## 预测结果
- **测试组合数**: {result['total_combinations_tested']}
- **发现通路数**: {result['total_pathways_found']}
- **平均通路得分**: {result['statistics']['avg_pathway_score']:.4f}
- **最高通路得分**: {result['statistics']['max_pathway_score']:.4f}

## 最佳通路

"""
        # 汇总一次性输出：边概率一览（有向）
        merged_edge_predictions = {}
        edge_source_list = result.get('all_pathways', []) or result.get('top_pathways', [])
        for pr in edge_source_list:
            ep = pr.get('edge_predictions', {})
            for k, v in ep.items():
                merged_edge_predictions[k] = max(v, merged_edge_predictions.get(k, v))

        if merged_edge_predictions:
            report += "**边概率一览（有向）**:\n"
            for edge_key, prob in sorted(merged_edge_predictions.items(), key=lambda kv: kv[1], reverse=True):
                try:
                    src, dst = edge_key.split('-')
                except ValueError:
                    src, dst = edge_key, ''
                report += f"- {src} → {dst}: {prob:.4f}\n"
            report += "\n"

        # 汇总一次性输出：节点对应输入KO（从所有通路合并映射并按节点顺序展示）
        node_source_list = result.get('all_pathways', []) or result.get('top_pathways', [])
        if node_source_list:
            # 汇总节点顺序：以首个通路为先，随后补齐其它通路中的节点
            primary_nodes = node_source_list[0]['pathway'].get('nodes', [])
            all_nodes = []
            seen_nodes = set()
            for n in primary_nodes:
                if n not in seen_nodes:
                    seen_nodes.add(n)
                    all_nodes.append(n)
            for pr in node_source_list:
                for n in pr['pathway'].get('nodes', []):
                    if n not in seen_nodes:
                        seen_nodes.add(n)
                        all_nodes.append(n)

            # 合并所有通路的ko_mapping
            combined_mapping = {}
            for pr in node_source_list:
                km = pr.get('ko_mapping', {})
                for node, kos in km.items():
                    combined_mapping[node] = list(set(combined_mapping.get(node, [])) | set(kos))

            report += "**节点对应输入KO**:\n"
            input_kos_set = set(result.get('input_ko_sequence', []))
            for node in all_nodes:
                kos_all = combined_mapping.get(node, [])
                kos_filtered = [ko for ko in kos_all if ko in input_kos_set]
                report += f"- {node}: {', '.join(kos_filtered) if kos_filtered else '无输入KO映射'}\n"
            report += "\n"

        # 通路分数总览表
        if result.get('top_pathways'):
            report += "## 通路分数总览\n\n"
            report += "| 排名 | 分数 | 代谢通路 |\n"
            report += "|------|------|----------|\n"
            for idx, pathway_result in enumerate(result['top_pathways'][:10], 1):
                p = pathway_result['pathway']
                path_str = ' → '.join(p.get('nodes', []))
                report += f"| {idx} | {p.get('score', 0):.4f} | {path_str} |\n"

        # 按组合分类显示所有通路详单
        all_list = result.get('all_pathways', [])
        if all_list:
            report += "\n## 按组合分类的通路详单\n\n"
            
            # 按组合对通路进行分组
            combination_groups = {}
            for pathway_result in all_list:
                combination = pathway_result.get('combination', [])
                combination_key = tuple(sorted(combination)) if combination else tuple()
                if combination_key not in combination_groups:
                    combination_groups[combination_key] = []
                combination_groups[combination_key].append(pathway_result)
            
            # 按组合显示
            for combo_idx, (combination_key, pathways) in enumerate(combination_groups.items(), 1):
                combination_list = list(combination_key)
                report += f"### 组合{combo_idx}：[{', '.join(combination_list)}]\n\n"
                
                # 显示该组合的统计信息
                combo_scores = [p['pathway'].get('score', 0) for p in pathways]
                report += f"- **通路数量**: {len(pathways)}\n"
                report += f"- **最高得分**: {max(combo_scores):.4f}\n"
                report += f"- **平均得分**: {sum(combo_scores)/len(combo_scores):.4f}\n"
                report += f"- **最低得分**: {min(combo_scores):.4f}\n\n"
                
                # 显示该组合的前5条最佳通路
                sorted_pathways = sorted(pathways, key=lambda x: x['pathway'].get('score', 0), reverse=True)
                report += "**最佳通路（前5条）**:\n\n"
                
                for i, pathway_result in enumerate(sorted_pathways[:5], 1):
                    pathway = pathway_result['pathway']
                    report += f"#### 通路 {combo_idx}.{i}\n"
                    report += f"- **节点序列**: {' → '.join(pathway.get('nodes', []))}\n"
                    report += f"- **通路得分**: {pathway.get('score', 0):.4f}\n"
                    report += f"- **通路长度**: {pathway.get('length', 0)}\n"
                    
                    # 边连接（仅列出当前通路使用到的边及概率）
                    if pathway.get('edges'):
                        report += "- **边连接**:\n"
                        for edge in pathway['edges']:
                            report += f"  - {edge['from']} → {edge['to']}: {edge['probability']:.4f}\n"
                    report += "\n"
                
                # 如果该组合有超过5条通路，显示其他通路的简要信息
                if len(pathways) > 5:
                    report += f"**其他通路（{len(pathways)-5}条）**:\n\n"
                    report += "| 序号 | 得分 | 通路 |\n"
                    report += "|------|------|------|\n"
                    for i, pathway_result in enumerate(sorted_pathways[5:], 6):
                        pathway = pathway_result['pathway']
                        path_str = ' → '.join(pathway.get('nodes', []))
                        report += f"| {combo_idx}.{i} | {pathway.get('score', 0):.4f} | {path_str} |\n"
                    report += "\n"
                
                report += "---\n\n"

        

        # 保存报告
        if output_file:
            # 确保使用项目根目录的prediction_output文件夹
            project_root = Path(__file__).parent.parent.parent  # 从src/kegnn_model回到项目根目录
            output_dir = project_root / "prediction_output"
            
            # 创建输出目录（如果不存在）
            output_dir.mkdir(exist_ok=True)
            
            # 构建完整的输出文件路径
            output_path = output_dir / output_file
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"📄 报告已保存: {output_path}")
        
        return report


def main():
    """主函数"""
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='KO序列节点预测工具')
    parser.add_argument('--ko_sequence', type=str, help='KO序列，用空格分隔（例如：K14755 K01904 K01598 K00816 K03418）')
    parser.add_argument('--max_combinations', type=int, default=20, help='最大组合数量（默认：20）')
    parser.add_argument('--interactive', action='store_true', help='强制使用交互式输入模式')
    
    args = parser.parse_args()
    
    print("🚀 KO序列节点预测演示")
    print("=" * 60)
    
    # 初始化预测器
    predictor = KONodePredictor()
    
    # 确定KO序列输入方式
    if args.ko_sequence and not args.interactive:
        # 使用命令行参数
        ko_sequence = args.ko_sequence.split()
        ko_sequences = [ko_sequence]
        print(f"✅ 使用命令行参数KO序列: {ko_sequence}")
    else:
        # 使用交互式输入
        print("请输入KO序列（用空格分隔，例如：K14755 K01904 K01598 K00816 K03418）:")
        user_input = input("KO序列: ").strip()
        

            # 解析用户输入的KO序列
        ko_sequence = user_input.split()
        ko_sequences = [ko_sequence]
        print(f"✅ 已输入KO序列: {ko_sequence}")


   
    
    for i, ko_sequence in enumerate(ko_sequences, 1):
        print(f"\n {i}: KO序列预测")
        print(f"KO序列: {ko_sequence}")
        
        # 1. 分析KO覆盖情况
        coverage = predictor.analyze_ko_coverage(ko_sequence)
        print(f"映射成功率: {coverage['mapping_rate']*100:.1f}%")
        print(f"映射节点数: {coverage['total_nodes']}")
        
        # 2. 进行通路预测
        if coverage['mapped_kos'] > 0:
            result = predictor.predict_from_ko_sequence(ko_sequence, max_combinations=args.max_combinations if 'args' in locals() else 20)
            
            if not result.get('error'):
                print(f"发现通路数: {result['total_pathways_found']}")
                
                if result['top_pathways']:
                    best_pathway = result['top_pathways'][0]
                    pathway = best_pathway['pathway']
                    print(f"最佳通路: {' → '.join(pathway['nodes'])}")
                    print(f"通路得分: {pathway['score']:.4f}")
                    
                    # 生成报告
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_file = f"ko_prediction_report_{timestamp}.md"
                    predictor.generate_prediction_report(result, report_file)
                    print(f"报告已生成: prediction_output/{report_file}")
                else:
                    print("未找到有效通路")
            else:
                print(f"预测失败: {result['error']}")
        else:
            print("没有有效的KO映射，跳过预测")
        
        print("-" * 40)
    
    print("\n✅ KO节点预测完成!")


if __name__ == "__main__":
    main()