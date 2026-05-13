"""
模型训练脚本
"""

import os
import yaml
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_processing import MolecularDataProcessor
from model import TransitionStatePredictor, TSPredictionLoss


class TSTrainer:
    """过渡态预测模型训练器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['training']['device'] 
                                 if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 初始化数据处理器
        self.data_processor = MolecularDataProcessor(config_path)
        
        # 初始化模型
        self.model = TransitionStatePredictor(self.config['model']).to(self.device)
        
        # 损失函数和优化器
        self.criterion = TSPredictionLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['model']['learning_rate']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def prepare_batch_data(self, batch_data):
        """准备批次数据"""
        reactant_graphs = []
        product_graphs = []
        ts_coords = []
        reactant_coords = []
        product_coords = []
        
        for data in batch_data:
            reactant_graphs.append(data['reactant_graph'])
            product_graphs.append(data['product_graph'])
            ts_coords.append(data['ts_pos'])
            reactant_coords.append(data['reactant_pos'])
            product_coords.append(data['product_pos'])
        
        # 转换为张量
        ts_coords = torch.stack(ts_coords).to(self.device)
        reactant_coords = torch.stack(reactant_coords).to(self.device)
        product_coords = torch.stack(product_coords).to(self.device)
        
        return reactant_graphs, product_graphs, ts_coords, reactant_coords, product_coords
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_data in progress_bar:
            try:
                # 准备数据
                r_graphs, p_graphs, ts_coords, r_coords, p_coords = self.prepare_batch_data(batch_data)
                
                # 前向传播
                self.optimizer.zero_grad()
                
                # 注意：这里需要根据实际的批处理方式调整
                # 简化版本：逐个处理
                batch_loss = 0
                for i in range(len(r_graphs)):
                    pred_coords = self.model(r_graphs[i], p_graphs[i])
                    loss, loss_dict = self.criterion(
                        pred_coords, ts_coords[i], r_coords[i], p_coords[i]
                    )
                    batch_loss += loss
                
                batch_loss = batch_loss / len(r_graphs)
                
                # 反向传播
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': batch_loss.item()})
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                try:
                    r_graphs, p_graphs, ts_coords, r_coords, p_coords = self.prepare_batch_data(batch_data)
                    
                    batch_loss = 0
                    for i in range(len(r_graphs)):
                        pred_coords = self.model(r_graphs[i], p_graphs[i])
                        loss, _ = self.criterion(
                            pred_coords, ts_coords[i], r_coords[i], p_coords[i]
                        )
                        batch_loss += loss
                    
                    batch_loss = batch_loss / len(r_graphs)
                    total_loss += batch_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def train(self):
        """完整训练流程"""
        print("Loading training data...")
        
        # 优先使用Transition1x数据
        if 'transition1x_path' in self.config['data'] and os.path.exists(self.config['data']['transition1x_path']):
            print("Using Transition1x HDF5 dataset...")
            train_data = self.data_processor.load_dataset(
                self.config['data']['transition1x_path'], is_training=True
            )
        else:
            print("Using traditional XYZ dataset...")
            train_data = self.data_processor.load_dataset(
                self.config['data']['train_path'], is_training=True
            )
        
        if len(train_data) == 0:
            print("No training data found!")
            return
        
        # 划分训练集和验证集
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # 创建数据加载器（简化版本）
        batch_size = self.config['model']['batch_size']
        train_batches = [train_data[i:i+batch_size] for i in range(0, len(train_data), batch_size)]
        val_batches = [val_data[i:i+batch_size] for i in range(0, len(val_data), batch_size)]
        
        # 训练循环
        epochs = self.config['model']['epochs']
        patience = self.config['training']['early_stopping_patience']
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_batches)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate_epoch(val_batches)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model()
                print("New best model saved!")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def save_model(self):
        """保存模型"""
        os.makedirs(os.path.dirname(self.config['training']['save_path']), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }, self.config['training']['save_path'])
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    print("Starting transition state prediction model training...")
    
    trainer = TSTrainer()
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()