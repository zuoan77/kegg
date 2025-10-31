#!/usr/bin/env python3
"""
链接预测训练器
用于训练和评估KEGG图神经网络模型
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_recall_curve
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 50, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前验证分数（越高越好）
            model: 模型实例
            
        Returns:
            是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """保存最佳模型权重"""
        self.best_weights = model.state_dict().copy()


class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """计算各种评估指标"""
        metrics = {}
        
        # AUC-ROC
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = 0.0
        
        # AUC-PR
        try:
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auc_pr'] = 0.0
        
        # 准确率
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 精确率、召回率、F1分数
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        try:
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        except ValueError:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0
        
        return metrics
    
    @staticmethod
    def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """找到最佳阈值"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        y_pred_best = (y_prob >= best_threshold).astype(int)
        best_metrics = MetricsCalculator.calculate_metrics(y_true, y_pred_best, y_prob)
        
        return best_threshold, best_metrics


class LinkPredictionTrainer:
    """链接预测训练器"""
    
    def __init__(self, model: nn.Module, config, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # 优化器
        if config.optimizer == "Adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "SGD":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        
        # 学习率调度器
        if config.scheduler == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.scheduler_step_size,
                gamma=config.scheduler_gamma
            )
        elif config.scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=20,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # 损失函数
        if config.loss_function == "BCEWithLogitsLoss":
            pos_weight = torch.tensor([config.pos_weight]) if config.pos_weight else None
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif config.loss_function == "BCELoss":
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported loss function: {config.loss_function}")
        
        # 早停
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.patience,
                min_delta=config.min_delta
            )
        else:
            self.early_stopping = None
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_auc': [],
            'val_loss': [],
            'val_auc': [],
            'val_metrics': []
        }
        
        # 最佳模型
        self.best_val_auc = 0.0
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader, graph_data: Dict) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in train_loader:
            # 移动数据到设备
            edges = batch['edges'].to(self.device)
            labels = batch['labels'].to(self.device)
            node_features = batch['node_features'].to(self.device)
            
            # 图数据
            edge_index = graph_data['edge_index'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(node_features, edge_index, edges)
            
            # 计算损失
            loss = self.criterion(predictions, labels.float())
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            
            # 计算预测概率
            with torch.no_grad():
                probs = torch.sigmoid(predictions)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算平均损失和AUC
        avg_loss = total_loss / len(train_loader)
        
        try:
            train_auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            train_auc = 0.0
        
        return avg_loss, train_auc
    
    def validate(self, val_loader: DataLoader, graph_data: Dict) -> Tuple[float, float, Dict[str, float]]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # 移动数据到设备
                edges = batch['edges'].to(self.device)
                labels = batch['labels'].to(self.device)
                node_features = batch['node_features'].to(self.device)
                
                # 图数据
                edge_index = graph_data['edge_index'].to(self.device)
                
                # 前向传播
                predictions = self.model(node_features, edge_index, edges)
                
                # 计算损失
                loss = self.criterion(predictions, labels.float())
                total_loss += loss.item()
                
                # 计算预测概率
                probs = torch.sigmoid(predictions)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 找到最佳阈值并计算指标
        best_threshold, metrics = MetricsCalculator.find_best_threshold(all_labels, all_preds)
        
        return avg_loss, metrics['auc_roc'], metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              graph_data: Dict) -> Dict[str, Any]:
        """完整训练流程"""
        logger.info("🚀 开始训练...")
        logger.info(f"   训练轮数: {self.config.epochs}")
        logger.info(f"   批大小: {self.config.batch_size}")
        logger.info(f"   学习率: {self.config.learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_auc = self.train_epoch(train_loader, graph_data)
            
            # 验证
            if epoch % self.config.eval_every == 0:
                val_loss, val_auc, val_metrics = self.validate(val_loader, graph_data)
                
                # 记录历史
                self.train_history['train_loss'].append(train_loss)
                self.train_history['train_auc'].append(train_auc)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_auc'].append(val_auc)
                self.train_history['val_metrics'].append(val_metrics)
                
                # 保存最佳模型
                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    self.best_model_state = self.model.state_dict().copy()
                
                # 学习率调度
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_auc)
                    else:
                        self.scheduler.step()
                
                # 早停检查
                if self.early_stopping:
                    if self.early_stopping(val_auc, self.model):
                        logger.info(f"早停在第 {epoch+1} 轮")
                        break
                
                # 打印进度
                if self.config.verbose:
                    epoch_time = time.time() - epoch_start
                    logger.info(
                        f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                        f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | "
                        f"Time: {epoch_time:.2f}s"
                    )
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        total_time = time.time() - start_time
        logger.info(f"✅ 训练完成! 总时间: {total_time:.2f}s")
        logger.info(f"   最佳验证AUC: {self.best_val_auc:.4f}")
        
        return {
            'best_val_auc': self.best_val_auc,
            'train_history': self.train_history,
            'total_time': total_time
        }
    
    def evaluate(self, test_loader: DataLoader, graph_data: Dict) -> Dict[str, float]:
        """评估模型"""
        logger.info("📊 评估模型...")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                edges = batch['edges'].to(self.device)
                labels = batch['labels'].to(self.device)
                node_features = batch['node_features'].to(self.device)
                edge_index = graph_data['edge_index'].to(self.device)
                
                predictions = self.model(node_features, edge_index, edges)
                probs = torch.sigmoid(predictions)
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 计算指标
        best_threshold, metrics = MetricsCalculator.find_best_threshold(all_labels, all_preds)
        metrics['best_threshold'] = best_threshold
        
        logger.info("📈 测试结果:")
        for metric, value in metrics.items():
            logger.info(f"   {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_auc': self.best_val_auc,
            'train_history': self.train_history
        }, filepath)
        logger.info(f"💾 模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        self.train_history = checkpoint.get('train_history', {})
        logger.info(f"📂 模型已从 {filepath} 加载")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """绘制训练曲线"""
        if not self.train_history['train_loss']:
            logger.warning("没有训练历史数据可绘制")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 损失曲线
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='训练损失')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='验证损失')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # AUC曲线
        ax2.plot(epochs, self.train_history['train_auc'], 'b-', label='训练AUC')
        ax2.plot(epochs, self.train_history['val_auc'], 'r-', label='验证AUC')
        ax2.set_title('训练和验证AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()
        ax2.grid(True)
        
        # 验证指标
        if self.train_history['val_metrics']:
            precision_scores = [m.get('precision', 0) for m in self.train_history['val_metrics']]
            recall_scores = [m.get('recall', 0) for m in self.train_history['val_metrics']]
            f1_scores = [m.get('f1', 0) for m in self.train_history['val_metrics']]
            
            ax3.plot(epochs, precision_scores, 'g-', label='Precision')
            ax3.plot(epochs, recall_scores, 'orange', label='Recall')
            ax3.plot(epochs, f1_scores, 'purple', label='F1-Score')
            ax3.set_title('验证指标')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Score')
            ax3.legend()
            ax3.grid(True)
        
        # 学习率曲线（如果有调度器）
        ax4.text(0.5, 0.5, f'最佳验证AUC: {self.best_val_auc:.4f}', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('训练总结')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 训练曲线已保存到: {save_path}")
        
        plt.show()


def create_trainer(model, training_config, device='cpu'):
    """创建训练器的工厂函数"""
    return LinkPredictionTrainer(
        model=model,
        config=training_config,
        device=device
    )