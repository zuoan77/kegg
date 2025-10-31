"""
数据预处理模块
负责从KEGG/MetaCyc数据库提取和处理代谢反应数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KEGGDataProcessor:
    """KEGG数据处理器"""
    
    @staticmethod
    def get_project_root() -> Path:
        """
        获取项目根目录
        
        Returns:
            项目根目录的Path对象
        """
        # 从当前文件位置向上查找项目根目录
        current_file = Path(__file__).resolve()
        # 当前文件在 src/data_processing/kegg_processor.py
        # 项目根目录在上两级
        project_root = current_file.parent.parent.parent
        return project_root
    
    def __init__(self, data_path: str):
        """
        初始化数据处理器
        
        Args:
            data_path: 数据文件路径（支持相对路径和绝对路径）
                      相对路径将相对于项目根目录解析
        """
        # 处理路径：如果是相对路径，则相对于项目根目录解析
        path_obj = Path(data_path)
        if path_obj.is_absolute():
            self.data_path = path_obj
        else:
            # 相对路径相对于项目根目录
            project_root = self.get_project_root()
            self.data_path = project_root / data_path
            
        self.reactions_df = None
        self.compounds_df = None
        
    def load_reaction_data(self) -> pd.DataFrame:
        """
        加载反应数据
        
        Returns:
            包含反应信息的DataFrame
        """
        try:
            if self.data_path.suffix == '.xlsx':
                self.reactions_df = pd.read_excel(self.data_path)
            elif self.data_path.suffix == '.csv':
                self.reactions_df = pd.read_csv(self.data_path)
            else:
                raise ValueError(f"不支持的文件格式: {self.data_path.suffix}")
                
            logger.info(f"成功加载 {len(self.reactions_df)} 条反应数据")
            return self.reactions_df
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
            
    def extract_reaction_features(self, reaction_row: pd.Series) -> Dict:
        """
        提取单个反应的特征
        
        Args:
            reaction_row: 反应数据行
            
        Returns:
            包含反应特征的字典
        """
        features = {
            'substrates': self._parse_compounds(reaction_row.get('From', '')),
            'products': self._parse_compounds(reaction_row.get('To', '')),
            'ko_numbers': self._parse_ko_numbers(reaction_row.get('Orthology', '')),
            # 直接从文件中读取差异分子式和差异分子量
            'molecular_diff': reaction_row.get('diff_formula', ''),
            'mass_diff': reaction_row.get('diff_mass', 0.0),
            'atom_diff': self._parse_atom_diff(reaction_row.get('diff_formula', ''))  # 解析原子差异
        }
        
        return features
        
    def _parse_compounds(self, compound_str: str) -> List[str]:
        """解析化合物字符串"""
        if pd.isna(compound_str) or not compound_str:
            return []
        return [c.strip() for c in str(compound_str).split(',') if c.strip()]
        
    def _parse_ko_numbers(self, ko_str: str) -> List[str]:
        """解析KO编号字符串"""
        if pd.isna(ko_str) or not ko_str:
            return []
        # 匹配KO编号格式 (K##### 或 ko:#####)
        ko_pattern = r'K\d{5}|ko:\d{5}'
        return re.findall(ko_pattern, str(ko_str))
        
    def _parse_atom_diff(self, diff_formula: str) -> Dict[str, int]:
        """
        解析差异分子式中的原子差异
        
        Args:
            diff_formula: 差异分子式字符串，如 "C-1" 或 "C10H10N2O2"
            
        Returns:
            原子差异字典，如 {'C': -1} 或 {'C': 10, 'H': 10, 'N': 2, 'O': 2}
        """
        if pd.isna(diff_formula) or not diff_formula:
            return {}
            
        atom_diff = {}
        # 使用正则表达式匹配原子和数量
        # 匹配模式：元素符号 + 可选的正负号 + 数字
        pattern = r'([A-Z][a-z]?)(\-?\d+)'
        matches = re.findall(pattern, str(diff_formula))
        
        for element, count in matches:
            atom_diff[element] = int(count)
            
        return atom_diff
        
    # 移除或注释掉不再使用的 _calculate_reaction_diff 方法
    # def _calculate_reaction_diff(self, features: Dict) -> Dict:
    #     """
    #     计算反应的差异分子式和差异分子量
    #     （已移除，因为文件中已提供）
    #     """
    #     try:
    #         # 这里需要根据实际的化合物数据库来实现
    #         # 暂时返回空值，后续可以结合KEGG Compound数据库
    #         return {
    #             'molecular_diff': '',
    #             'mass_diff': 0.0,
    #             'atom_diff': {}
    #         }
    #     except Exception as e:
    #         logger.warning(f"计算反应差异失败: {e}")
    #         return {
    #             'molecular_diff': '',
    #             'mass_diff': 0.0,
    #             'atom_diff': {}
    #         }
            
    def create_rcu_nodes(self) -> pd.DataFrame:
        """
        创建RCU节点
        
        RCU (Reaction Class Unit) 的定义：
        具有相同差异分子式和差异分子量的反应归为同一个RCU节点
        
        Returns:
            包含RCU节点信息的DataFrame
        """
        if self.reactions_df is None:
            raise ValueError("请先加载反应数据")
        
        # 首先提取所有反应的特征
        all_reactions = []
        for idx, row in self.reactions_df.iterrows():
            features = self.extract_reaction_features(row)
            features['original_index'] = idx
            all_reactions.append(features)
        
        # 基于差异分子式和差异分子量进行分组
        rcu_groups = {}
        
        for reaction in all_reactions:
            # 创建RCU键（基于差异分子式和差异分子量）
            molecular_diff = reaction['molecular_diff']
            mass_diff = reaction['mass_diff']
            
            # 处理缺失值
            if pd.isna(molecular_diff):
                molecular_diff = ''
            if pd.isna(mass_diff):
                mass_diff = 0.0
            
            rcu_key = (str(molecular_diff), float(mass_diff))
            
            if rcu_key not in rcu_groups:
                rcu_groups[rcu_key] = []
            rcu_groups[rcu_key].append(reaction)
        
        logger.info(f"基于差异分子式和分子量分组，共找到 {len(rcu_groups)} 个唯一的RCU类型")
        
        # 为每个RCU组创建一个RCU节点
        rcu_nodes = []
        
        for rcu_idx, (rcu_key, reactions) in enumerate(rcu_groups.items()):
            molecular_diff, mass_diff = rcu_key
            
            # 合并同一RCU组内的所有底物、产物和KO编号
            all_substrates = set()
            all_products = set()
            all_ko_numbers = set()
            all_atom_diffs = {}
            
            for reaction in reactions:
                all_substrates.update(reaction['substrates'])
                all_products.update(reaction['products'])
                all_ko_numbers.update(reaction['ko_numbers'])
                
                # 合并原子差异（应该都相同，取第一个非空的）
                if reaction['atom_diff'] and not all_atom_diffs:
                    all_atom_diffs = reaction['atom_diff']
            
            # 创建RCU节点
            rcu_node = {
                'rcu_id': f"RCU_{rcu_idx:06d}",
                'substrates': sorted(list(all_substrates)),
                'products': sorted(list(all_products)),
                'ko_numbers': sorted(list(all_ko_numbers)),
                'molecular_diff': molecular_diff,
                'mass_diff': mass_diff,
                'atom_diff': all_atom_diffs,
                'reaction_count': len(reactions),  # 该RCU包含的反应数量
                'original_indices': [r['original_index'] for r in reactions]  # 原始反应索引
            }
            
            rcu_nodes.append(rcu_node)
        
        rcu_df = pd.DataFrame(rcu_nodes)
        logger.info(f"创建了 {len(rcu_df)} 个RCU节点，覆盖了 {len(all_reactions)} 个原始反应")
        
        # 添加 count 列，统计每一行的生成物在反应物列中出现的次数
        rcu_df['count'] = rcu_df['products'].apply(
            lambda products: sum(
                1 for substrates in rcu_df['substrates'] 
                if set(products).intersection(set(substrates))
            )
        )
        
        return rcu_df
        
    def save_processed_data(self, rcu_df: pd.DataFrame, output_path: str):
        """
        保存处理后的数据
        
        Args:
            rcu_df: RCU节点DataFrame
            output_path: 输出路径（支持相对路径和绝对路径）
                        相对路径将相对于项目根目录解析
        """
        # 处理路径：如果是相对路径，则相对于项目根目录解析
        path_obj = Path(output_path)
        if path_obj.is_absolute():
            output_file = path_obj
        else:
            # 相对路径相对于项目根目录
            project_root = self.get_project_root()
            output_file = project_root / output_path
            
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.suffix == '.csv':
            rcu_df.to_csv(output_file, index=False)
        elif output_file.suffix == '.xlsx':
            rcu_df.to_excel(output_file, index=False)
        else:
            # 默认保存为pickle格式
            rcu_df.to_pickle(output_file)
            
        logger.info(f"数据已保存到: {output_file}")


def main():
    """
    示例用法：展示如何使用相对路径
    """
    # 使用相对路径（相对于项目根目录）
    data_path = "data/all_reaction.xlsx"  # 相对路径
    output_path = "output/dataprocess/rcu_nodes.csv"  # 修改为图构建脚本期望的路径
    
    try:
        # 初始化处理器
        processor = KEGGDataProcessor(data_path)
        
        # 加载数据
        logger.info("开始加载反应数据...")
        processor.load_reaction_data()
        
        # 创建RCU节点
        logger.info("开始创建RCU节点...")
        rcu_df = processor.create_rcu_nodes()
        
        # 保存处理后的数据
        logger.info("开始保存处理后的数据...")
        processor.save_processed_data(rcu_df, output_path)
        
        logger.info("数据预处理完成！")
        
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        raise


if __name__ == "__main__":
    main()


