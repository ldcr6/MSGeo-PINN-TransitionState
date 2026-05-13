#!/usr/bin/env python3
"""
基于处理后数据的模型训练脚本
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 添加src目录到路径
sys.path.append('src')
from model import TransitionStatePredictor, TSPredictionLoss


class ProcessedDataTrainer:
    """基于处理后数据的训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = TransitionStatePredictor(config['model']).to(self.device)
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 损失函数和优化器
        self.criterion = TSPredictionLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['model']['learning_rate'],
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.7
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def load_processed_data(self, data_dir: str = "data/processed"):
        """加载处理后的数据"""
        data = {}
        
        for split in ['train', 'val', 'test']:
            file_path = Path(data_dir) / f"{split}_data.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    data[split] = pickle.load(f)
                print(f"✅ 加载 {split} 数据: {len(data[split])} 个样本")
            else:
                print(f"❌ 找不到 {split} 数据文件: {file_path}")
        
        return data
    
    def create_batch_data(self, samples, batch_size: int):
        """创建批次数据"""
        batches = []
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            batches.append(batch_samples)
        return batches
    
    def process_batch(self, batch_samples):
        """处理单个批次"""
        reactant_graphs = []
        product_graphs = []
        ts_coords_list = []
        reactant_coords_list = []
        product_coords_list = []
        
        for sample in batch_samples:
            # 移动图数据到设备
            r_graph = sample['reactant_graph'].to(self.device)
            p_graph = sample['product_graph'].to(self.device)
            
            reactant_graphs.append(r_graph)
            product_graphs.append(p_graph)
            
            # 移动坐标数据到设备
            ts_coords_list.append(sample['ts_pos'].to(self.device))
            reactant_coords_list.append(sample['reactant_pos'].to(self.device))
            product_coords_list.append(sample['product_pos'].to(self.device))
        
        return reactant_graphs, product_graphs, ts_coords_list, reactant_coords_list, product_coords_list
    
    def train_epoch(self, train_batches):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_coord_loss = 0
        total_smooth_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_batches, desc="训练中")
        
        for batch_samples in progress_bar:
            try:
                # 处理批次数据
                r_graphs, p_graphs, ts_coords, r_coords, p_coords = self.process_batch(batch_samples)
                
                self.optimizer.zero_grad()
                batch_loss = 0
                batch_coord_loss = 0
                batch_smooth_loss = 0
                
                # 逐个样本处理（因为图大小不同）
                for i in range(len(r_graphs)):
                    # 前向传播
                    pred_coords = self.model(r_graphs[i], p_graphs[i])
                    
                    # 计算损失
                    loss, loss_dict = self.criterion(
                        pred_coords, ts_coords[i], r_coords[i], p_coords[i]
                    )
                    
                    batch_loss += loss
                    batch_coord_loss += loss_dict['coord_loss']
                    batch_smooth_loss += loss_dict['smooth_loss']
                
                # 平均损失
                batch_loss = batch_loss / len(r_graphs)
                batch_coord_loss = batch_coord_loss / len(r_graphs)
                batch_smooth_loss = batch_smooth_loss / len(r_graphs)
                
                # 反向传播
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 累计损失
                total_loss += batch_loss.item()
                total_coord_loss += batch_coord_loss
                total_smooth_loss += batch_smooth_loss
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{batch_loss.item():.6f}',
                    'coord': f'{batch_coord_loss:.6f}',
                    'smooth': f'{batch_smooth_loss:.6f}'
                })
                
            except Exception as e:
                print(f"批次训练错误: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_coord_loss = total_coord_loss / num_batches if num_batches > 0 else 0
        avg_smooth_loss = total_smooth_loss / num_batches if num_batches > 0 else 0
        
        return avg_loss, avg_coord_loss, avg_smooth_loss
    
    def validate_epoch(self, val_batches):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_coord_loss = 0
        total_smooth_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_batches, desc="验证中")
            
            for batch_samples in progress_bar:
                try:
                    r_graphs, p_graphs, ts_coords, r_coords, p_coords = self.process_batch(batch_samples)
                    
                    batch_loss = 0
                    batch_coord_loss = 0
                    batch_smooth_loss = 0
                    
                    for i in range(len(r_graphs)):
                        pred_coords = self.model(r_graphs[i], p_graphs[i])
                        loss, loss_dict = self.criterion(
                            pred_coords, ts_coords[i], r_coords[i], p_coords[i]
                        )
                        
                        batch_loss += loss
                        batch_coord_loss += loss_dict['coord_loss']
                        batch_smooth_loss += loss_dict['smooth_loss']
                    
                    batch_loss = batch_loss / len(r_graphs)
                    batch_coord_loss = batch_coord_loss / len(r_graphs)
                    batch_smooth_loss = batch_smooth_loss / len(r_graphs)
                    
                    total_loss += batch_loss.item()
                    total_coord_loss += batch_coord_loss
                    total_smooth_loss += batch_smooth_loss
                    num_batches += 1
                    
                    progress_bar.set_postfix({
                        'val_loss': f'{batch_loss.item():.6f}'
                    })
                    
                except Exception as e:
                    print(f"批次验证错误: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_coord_loss = total_coord_loss / num_batches if num_batches > 0 else 0
        avg_smooth_loss = total_smooth_loss / num_batches if num_batches > 0 else 0
        
        return avg_loss, avg_coord_loss, avg_smooth_loss
    
    def train(self, data_dir: str = "data/processed", epochs: int = 100):
        """完整训练流程"""
        print("="*60)
        print("开始训练过渡态预测模型")
        print("="*60)
        
        # 加载数据
        data = self.load_processed_data(data_dir)
        
        if 'train' not in data or 'val' not in data:
            print("❌ 缺少训练或验证数据")
            return
        
        train_data = data['train']
        val_data = data['val']
        
        print(f"训练样本: {len(train_data)}")
        print(f"验证样本: {len(val_data)}")
        
        # 创建批次
        batch_size = self.config['model']['batch_size']
        train_batches = self.create_batch_data(train_data, batch_size)
        val_batches = self.create_batch_data(val_data, batch_size)
        
        print(f"训练批次: {len(train_batches)}")
        print(f"验证批次: {len(val_batches)}")
        
        # 训练循环
        patience = self.config['training'].get('early_stopping_patience', 15)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_coord_loss, train_smooth_loss = self.train_epoch(train_batches)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_coord_loss, val_smooth_loss = self.validate_epoch(val_batches)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印结果
            print(f"训练损失: {train_loss:.6f} (坐标: {train_coord_loss:.6f}, 平滑: {train_smooth_loss:.6f})")
            print(f"验证损失: {val_loss:.6f} (坐标: {val_coord_loss:.6f}, 平滑: {val_smooth_loss:.6f})")
            print(f"学习率: {current_lr:.8f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(epoch)
                print("🎉 新的最佳模型已保存!")
            else:
                self.patience_counter += 1
                print(f"验证损失未改善 ({self.patience_counter}/{patience})")
            
            # 早停
            if self.patience_counter >= patience:
                print(f"早停: {patience} 个epoch后验证损失未改善")
                break
        
        # 绘制训练曲线
        self.plot_training_curves()
        print("\n🎉 训练完成!")
    
    def save_model(self, epoch: int):
        """保存模型"""
        save_dir = Path("models")
        save_dir.mkdir(exist_ok=True)
        
        save_path = save_dir / "best_ts_model.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, save_path)
        
        print(f"模型已保存到: {save_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失', color='blue')
        plt.plot(self.val_losses, label='验证损失', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学习率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('训练损失')
        plt.title('训练损失趋势')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("训练曲线已保存到: training_curves.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练过渡态预测模型')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='处理后数据目录')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'model': {
            'hidden_dim': args.hidden_dim,
            'num_layers': 3,
            'dropout': 0.1,
            'learning_rate': args.lr,
            'batch_size': args.batch_size
        },
        'training': {
            'early_stopping_patience': 15
        }
    }
    
    # 创建训练器并开始训练
    trainer = ProcessedDataTrainer(config)
    trainer.train(data_dir=args.data_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()