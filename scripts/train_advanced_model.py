#!/usr/bin/env python3
"""
训练高级过渡态预测模型
融合多种先进技术
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
import time

# 添加src目录到路径
sys.path.append('src')
from advanced_ts_model import AdvancedTransitionStatePredictor, AdvancedTSPredictionLoss


class AdvancedTrainer:
    """高级模型训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化高级模型
        self.model = AdvancedTransitionStatePredictor(config['model']).to(self.device)
        print(f"高级模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 高级损失函数
        self.criterion = AdvancedTSPredictionLoss(
            coord_weight=1.0,
            smooth_weight=0.1,
            distance_weight=0.2,
            angle_weight=0.1,
            physics_weight=0.15,
            uncertainty_weight=0.05
        )
        
        # 优化器（使用AdamW + 学习率预热）
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config['model']['learning_rate'],
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器（余弦退火 + 预热）
        self.warmup_epochs = 5
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=15, T_mult=2, eta_min=1e-7
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 性能监控
        self.training_times = []
        self.memory_usage = []
    
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
    
    def warmup_lr(self, epoch):
        """学习率预热"""
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config['model']['learning_rate'] * lr_scale
    
    def train_epoch(self, train_batches, epoch):
        """训练一个epoch"""
        self.model.train()
        start_time = time.time()
        
        total_losses = {
            'total': 0, 'coord': 0, 'smooth': 0, 'distance': 0, 
            'angle': 0, 'physics': 0, 'uncertainty': 0
        }
        num_batches = 0
        
        progress_bar = tqdm(train_batches, desc=f"训练Epoch {epoch+1}")
        
        for batch_samples in progress_bar:
            try:
                # 处理批次数据
                r_graphs, p_graphs, ts_coords, r_coords, p_coords = self.process_batch(batch_samples)
                
                self.optimizer.zero_grad()
                batch_losses = {key: 0 for key in total_losses.keys()}
                
                # 逐个样本处理
                for i in range(len(r_graphs)):
                    # 前向传播
                    pred_coords, uncertainty = self.model(r_graphs[i], p_graphs[i])
                    
                    # 计算损失
                    loss, loss_dict = self.criterion(
                        pred_coords, ts_coords[i], r_coords[i], p_coords[i], uncertainty
                    )
                    
                    batch_losses['total'] += loss.item()
                    for key in ['coord', 'smooth', 'distance', 'angle', 'physics', 'uncertainty']:
                        batch_losses[key] += loss_dict[f'{key}_loss']
                    
                    # 反向传播（累积梯度）
                    loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 优化器步骤
                self.optimizer.step()
                
                # 平均损失
                for key in batch_losses:
                    batch_losses[key] /= len(r_graphs)
                    total_losses[key] += batch_losses[key]
                
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{batch_losses["total"]:.6f}',
                    'coord': f'{batch_losses["coord"]:.6f}',
                    'physics': f'{batch_losses["physics"]:.6f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
            except Exception as e:
                print(f"批次训练错误: {e}")
                continue
        
        # 计算平均损失
        avg_losses = {key: total_losses[key] / num_batches if num_batches > 0 else 0 
                     for key in total_losses}
        
        # 记录训练时间
        epoch_time = time.time() - start_time
        self.training_times.append(epoch_time)
        
        return avg_losses
    
    def validate_epoch(self, val_batches):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_coord_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_batches, desc="验证中")
            
            for batch_samples in progress_bar:
                try:
                    r_graphs, p_graphs, ts_coords, r_coords, p_coords = self.process_batch(batch_samples)
                    
                    batch_loss = 0
                    batch_coord_loss = 0
                    
                    for i in range(len(r_graphs)):
                        pred_coords, uncertainty = self.model(r_graphs[i], p_graphs[i])
                        loss, loss_dict = self.criterion(
                            pred_coords, ts_coords[i], r_coords[i], p_coords[i], uncertainty
                        )
                        
                        batch_loss += loss.item()
                        batch_coord_loss += loss_dict['coord_loss']
                    
                    batch_loss /= len(r_graphs)
                    batch_coord_loss /= len(r_graphs)
                    
                    total_loss += batch_loss
                    total_coord_loss += batch_coord_loss
                    num_batches += 1
                    
                    progress_bar.set_postfix({
                        'val_loss': f'{batch_loss:.6f}'
                    })
                    
                except Exception as e:
                    print(f"批次验证错误: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_coord_loss = total_coord_loss / num_batches if num_batches > 0 else 0
        
        return avg_loss, avg_coord_loss
    
    def train(self, data_dir: str = "data/processed", epochs: int = 30):
        """完整训练流程"""
        print("="*60)
        print("开始训练高级过渡态预测模型")
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
        patience = self.config['training'].get('early_stopping_patience', 25)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 学习率预热
            if epoch < self.warmup_epochs:
                self.warmup_lr(epoch)
            
            # 训练
            train_losses = self.train_epoch(train_batches, epoch)
            self.train_losses.append(train_losses['total'])
            
            # 验证
            val_loss, val_coord_loss = self.validate_epoch(val_batches)
            self.val_losses.append(val_loss)
            
            # 学习率调度（预热后）
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印详细结果
            print(f"训练损失: {train_losses['total']:.6f}")
            print(f"  - 坐标: {train_losses['coord']:.6f}")
            print(f"  - 平滑: {train_losses['smooth']:.6f}")
            print(f"  - 距离: {train_losses['distance']:.6f}")
            print(f"  - 角度: {train_losses['angle']:.6f}")
            print(f"  - 物理: {train_losses['physics']:.6f}")
            print(f"  - 不确定性: {train_losses['uncertainty']:.6f}")
            print(f"验证损失: {val_loss:.6f} (坐标: {val_coord_loss:.6f})")
            print(f"学习率: {current_lr:.8f}")
            print(f"训练时间: {self.training_times[-1]:.1f}s")
            
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
        self.print_training_summary()
        print("\n🎉 高级模型训练完成!")
    
    def save_model(self, epoch: int):
        """保存模型"""
        save_dir = Path("models")
        save_dir.mkdir(exist_ok=True)
        
        save_path = save_dir / "best_advanced_ts_model.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'training_times': self.training_times
        }, save_path)
        
        print(f"模型已保存到: {save_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_losses, label='训练损失', color='blue')
        axes[0, 0].plot(self.val_losses, label='验证损失', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 学习率曲线
        epochs = range(1, len(self.train_losses) + 1)
        lrs = []
        for epoch in epochs:
            if epoch <= self.warmup_epochs:
                lr = self.config['model']['learning_rate'] * epoch / self.warmup_epochs
            else:
                # 简化的余弦退火计算
                lr = self.config['model']['learning_rate'] * 0.5 * (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / 15))
            lrs.append(lr)
        
        axes[0, 1].plot(epochs, lrs, color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('学习率')
        axes[0, 1].set_title('学习率调度')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 训练时间
        if self.training_times:
            axes[1, 0].plot(self.training_times, color='orange')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('时间 (秒)')
            axes[1, 0].set_title('每轮训练时间')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 损失改善
        if len(self.val_losses) > 1:
            improvements = [0]
            for i in range(1, len(self.val_losses)):
                improvement = self.val_losses[i-1] - self.val_losses[i]
                improvements.append(improvement)
            
            axes[1, 1].plot(improvements, color='purple')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('损失改善')
            axes[1, 1].set_title('验证损失改善')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_training_curves.png', dpi=300, bbox_inches='tight')
        print("训练曲线已保存到: advanced_training_curves.png")
    
    def print_training_summary(self):
        """打印训练总结"""
        print("\n" + "="*60)
        print("训练总结")
        print("="*60)
        
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"总训练轮数: {len(self.train_losses)}")
        
        if self.training_times:
            total_time = sum(self.training_times)
            avg_time = np.mean(self.training_times)
            print(f"总训练时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
            print(f"平均每轮时间: {avg_time:.1f}秒")
        
        # 性能改善
        if len(self.val_losses) > 1:
            initial_loss = self.val_losses[0]
            final_loss = min(self.val_losses)
            improvement = (initial_loss - final_loss) / initial_loss * 100
            print(f"验证损失改善: {improvement:.1f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练高级过渡态预测模型')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='处理后数据目录')
    parser.add_argument('--epochs', type=int, default=25,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏层维度')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'model': {
            'hidden_dim': args.hidden_dim,
            'num_layers': 4,
            'dropout': 0.1,
            'learning_rate': args.lr,
            'batch_size': args.batch_size
        },
        'training': {
            'early_stopping_patience': 25
        }
    }
    
    # 创建训练器并开始训练
    trainer = AdvancedTrainer(config)
    trainer.train(data_dir=args.data_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()