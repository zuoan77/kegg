#!/usr/bin/env python3
"""
KEGG数据训练脚本 - 简化版本
专门用于处理KEGG真实数据的训练
"""

import os
import pandas as pd
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from config import DataConfig, ModelConfig, TrainingConfig
from data_adapter import KEGGDataAdapter, KEGGDataset, collate_kegg_batch, prepare_kegg_data_for_training
from models import KEGGGraphModel, create_kegg_model
from trainer import LinkPredictionTrainer, create_trainer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_kegg_data(data_dir):
    """加载处理后的KEGG数据"""
    logger.info("🔧 加载KEGG数据...")
    
    # 读取节点数据
    nodes_file = Path(data_dir) / "processed_nodes.csv"
    edges_file = Path(data_dir) / "processed_edges.csv"
    weights_file = Path(data_dir) / "processed_weights.json"
    
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    
    with open(weights_file, 'r') as f:
        weights_data = json.load(f)
    
    logger.info(f"   节点数: {len(nodes_df)}")
    logger.info(f"   边数: {len(edges_df)}")
    logger.info(f"   权重数: {len(weights_data)}")
    
    return nodes_df, edges_df, weights_data

def create_simple_config(nodes_count, edges_count):
    """创建简化的配置"""
    
    # 数据配置
    data_config = DataConfig(
        data_dir="kegg_real_processed",
        nodes_file="processed_nodes.csv",
        edges_file="processed_edges.csv",
        weights_file="processed_weights.json"
    )
    
    # 模型配置 - 简化参数
    model_config = ModelConfig(
        model_type="GraphSAGE",
        input_dim=8,  # 根据实际特征维度调整
        hidden_dim=64,
        output_dim=32,
        num_layers=3,
        dropout=0.3
    )
    
    # 训练配置 - 简化训练
    training_config = TrainingConfig(
        epochs=500,       # 适当的训练轮数
        learning_rate=0.01,
        weight_decay=1e-4,
        patience=10,
        early_stopping=True
    )
    
    return data_config, model_config, training_config

def train_kegg_model():
    """训练KEGG模型"""
    
    # 创建结果目录
    results_dir = "kegg_training_results"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # 1. 加载数据
        nodes_df, edges_df, weights_data = load_kegg_data("kegg_real_processed")
        
        # 2. 创建配置
        data_config, model_config, training_config = create_simple_config(
            len(nodes_df), len(edges_df)
        )
        
        # 3. 准备数据
        logger.info("🔄 准备训练数据...")
        from data_adapter import prepare_kegg_data_for_training
        
        data_dict = prepare_kegg_data_for_training("kegg_real_processed", data_config)
        
        train_dataset = data_dict['train_dataset']
        val_dataset = data_dict['val_dataset']
        test_dataset = data_dict['test_dataset']
        graph_data = data_dict['graph_data']
        
        # 4. 创建数据加载器
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=True,
            collate_fn=collate_kegg_batch
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=False,
            collate_fn=collate_kegg_batch
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=False,
            collate_fn=collate_kegg_batch
        )
        
        # 5. 创建模型
        logger.info("🧠 创建模型...")
        from models import create_kegg_model
        
        # 更新模型配置的输入维度
        model_config.input_dim = graph_data['x'].shape[1]
        model = create_kegg_model(model_config)
        
        # 6. 创建训练器
        logger.info("🏃 创建训练器...")
        from trainer import create_trainer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"   使用设备: {device}")
        
        trainer = create_trainer(model, training_config, device)
        
        # 7. 开始训练
        logger.info("🚀 开始训练...")
        training_results = trainer.train(train_loader, val_loader, graph_data)
        
        # 8. 评估模型
        logger.info("📊 评估模型...")
        test_results = trainer.evaluate(test_loader, graph_data)
        
        # 9. 保存结果
        results = {
            'training_results': training_results,
            'test_results': test_results,
            'model_config': model_config.__dict__,
            'training_config': training_config.__dict__,
            'data_config': data_config.__dict__
        }
        
        results_file = os.path.join(results_dir, "training_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 10. 保存模型
        model_file = os.path.join(results_dir, "best_model.pth")
        trainer.save_model(model_file)
        
        # 11. 绘制训练曲线
        if training_config.save_plots:
            plot_file = os.path.join(results_dir, "training_curves.png")
            trainer.plot_training_curves(plot_file)
        
        logger.info("✅ 训练完成!")
        logger.info(f"   最佳验证AUC: {training_results['best_val_auc']:.4f}")
        logger.info(f"   测试AUC: {test_results.get('auc_roc', 0):.4f}")
        logger.info(f"   结果保存在: {results_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    train_kegg_model()