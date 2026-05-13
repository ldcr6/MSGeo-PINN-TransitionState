#!/usr/bin/env python3
"""
优化训练脚本 - 目标：RMSE ≤ 0.2Å
改进策略：
1. 更大的模型容量
2. 更长的训练时间
3. 数据增强
4. 改进的损失函数
5. 学习率调度优化
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('src')

from advanced_ts_model import AdvancedTransitionStatePredictor, AdvancedTSPredictionLoss
from ase.io import read

class OptimizedConfig:
    """优化的配置"""
    def __init__(self):
        # 模型配置 - 增加容量
        self.hidden_dim = 384  # 从256增加到384
        self.num_layers = 6    # 从4增加到6
        self.dropout = 0.15    # 轻微增加dropout
        
        # 训练配置
        self.batch_size = 8    # 从4增加到8
        self.epochs = 200      # 从100增加到200
        self.learning_rate = 1e-4  # 降低初始学习率
        self.weight_decay = 5e-5   # 增加正则化
        
        # 早停配置
        self.patience = 50     # 从25增加到50
        
        # 损失权重优化
        self.coord_weight = 1.0
        self.smooth_weight = 0.05   # 减少，避免过度平滑
        self.distance_weight = 0.3  # 增加距离保持
        self.angle_weight = 0.15    # 增加角度保持
        self.physics_weight = 0.2   # 增加物理约束
        self.uncertainty_weight = 0.1

def create_molecular_graph(xyz_file, device='cuda'):
    """Create molecular graph from XYZ file"""
    atoms = read(xyz_file)
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    
    n_atoms = len(atomic_numbers)
    atom_type_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 14: 5, 15: 6, 16: 7, 17: 8, 35: 9, 53: 10}
    
    node_features = []
    for atom_num in atomic_numbers:
        feat = [0.0] * 11
        if atom_num in atom_type_map:
            feat[atom_type_map[atom_num]] = 1.0
        node_features.append(feat)
    
    x = torch.tensor(node_features, dtype=torch.float32)
    pos = torch.tensor(positions, dtype=torch.float32)
    
    # Create edges
    edge_indices = []
    cutoff = 3.5  # 稍微增加cutoff
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < cutoff:
                edge_indices.append([i, j])
                edge_indices.append([j, i])
    
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor([[i, i] for i in range(n_atoms)], dtype=torch.long).t()
    
    data = Data(x=x, edge_index=edge_index, pos=pos)
    return data.to(device), atomic_numbers

def load_training_data(data_dir, max_samples=None):
    """加载训练数据"""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return []
    
    rxn_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if max_samples:
        rxn_dirs = rxn_dirs[:max_samples]
    
    print(f"Loading {len(rxn_dirs)} reactions...")
    
    samples = []
    for rxn_dir in tqdm(rxn_dirs, desc="Loading"):
        r_file = rxn_dir / "r.xyz"
        p_file = rxn_dir / "p.xyz"
        ts_file = rxn_dir / "ts.xyz"
        
        if not all([r_file.exists(), p_file.exists(), ts_file.exists()]):
            continue
        
        try:
            # 读取真实坐标
            ts_atoms = read(str(ts_file))
            ts_coords = torch.tensor(ts_atoms.get_positions(), dtype=torch.float32)
            
            samples.append({
                'r_file': str(r_file),
                'p_file': str(p_file),
                'ts_coords': ts_coords,
                'rxn_name': rxn_dir.name
            })
        except:
            continue
    
    print(f"Loaded {len(samples)} valid samples")
    return samples

def data_augmentation(coords, noise_level=0.02):
    """数据增强：添加小噪声"""
    noise = torch.randn_like(coords) * noise_level
    return coords + noise

def train_optimized_model():
    """训练优化模型"""
    print("="*80)
    print("Optimized Training - Target: RMSE <= 0.2A")
    print("="*80)
    
    # 配置
    config = OptimizedConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # 导入模型（使用绝对路径导入）
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from advanced_ts_model import AdvancedTransitionStatePredictor
    
    # 加载数据
    data_dir = "data/tar_extracted/train_data"
    if not os.path.exists(data_dir):
        print(f"使用备用数据：data/merged_training_data")
        data_dir = "data/merged_training_data"
    
    samples = load_training_data(data_dir, max_samples=None)
    
    if len(samples) < 10:
        print("Training data too small!")
        return
    
    # 划分训练集和验证集
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"\nTraining samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    
    # 创建模型
    model_config = {
        'hidden_dim': config.hidden_dim,
        'num_layers': config.num_layers,
        'dropout': config.dropout
    }
    
    model = AdvancedTransitionStatePredictor(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,}")
    
    # 损失函数
    criterion = AdvancedTSPredictionLoss(
        coord_weight=config.coord_weight,
        smooth_weight=config.smooth_weight,
        distance_weight=config.distance_weight,
        angle_weight=config.angle_weight,
        physics_weight=config.physics_weight,
        uncertainty_weight=config.uncertainty_weight
    )
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器 - 使用OneCycleLR以获得更好的性能
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate * 10,
        epochs=config.epochs,
        steps_per_epoch=len(train_samples) // config.batch_size + 1,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmsd': [],
        'learning_rates': [],
        'best_epoch': 0
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 训练循环
    print(f"\n{'='*80}")
    print("Starting Training...")
    print(f"{'='*80}\n")
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []
        
        # Shuffle training data
        np.random.shuffle(train_samples)
        
        for i in range(0, len(train_samples), config.batch_size):
            batch_samples = train_samples[i:i+config.batch_size]
            
            batch_loss = 0
            for sample in batch_samples:
                try:
                    # Create graphs
                    r_graph, _ = create_molecular_graph(sample['r_file'], device)
                    p_graph, _ = create_molecular_graph(sample['p_file'], device)
                    ts_coords = sample['ts_coords'].to(device)
                    
                    # Forward
                    pred_coords, uncertainty = model(r_graph, p_graph)
                    
                    # 数据增强（训练时）
                    if epoch < config.epochs * 0.7:  # 前70%训练使用增强
                        r_coords_aug = data_augmentation(r_graph.pos)
                        p_coords_aug = data_augmentation(p_graph.pos)
                    else:
                        r_coords_aug = r_graph.pos
                        p_coords_aug = p_graph.pos
                    
                    # Loss
                    loss, _ = criterion(
                        pred_coords, ts_coords,
                        r_coords_aug, p_coords_aug,
                        uncertainty
                    )
                    
                    batch_loss += loss
                except Exception as e:
                    continue
            
            if batch_loss > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_losses.append(batch_loss.item() / len(batch_samples))
        
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        
        # Validation
        model.eval()
        val_losses = []
        val_rmsds = []
        
        with torch.no_grad():
            for sample in val_samples:
                try:
                    r_graph, _ = create_molecular_graph(sample['r_file'], device)
                    p_graph, _ = create_molecular_graph(sample['p_file'], device)
                    ts_coords = sample['ts_coords'].to(device)
                    
                    pred_coords, uncertainty = model(r_graph, p_graph)
                    
                    loss, _ = criterion(
                        pred_coords, ts_coords,
                        r_graph.pos, p_graph.pos,
                        uncertainty
                    )
                    
                    # Calculate RMSD
                    rmsd = torch.sqrt(torch.mean((pred_coords - ts_coords) ** 2)).item()
                    
                    val_losses.append(loss.item())
                    val_rmsds.append(rmsd)
                except:
                    continue
        
        avg_val_loss = np.mean(val_losses) if val_losses else 0
        avg_val_rmsd = np.mean(val_rmsds) if val_rmsds else 0
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_rmsd'].append(avg_val_rmsd)
        history['learning_rates'].append(current_lr)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Val RMSD: {avg_val_rmsd:.4f} A")
        print(f"  LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            history['best_epoch'] = epoch + 1
            
            # 保存模型
            save_path = 'models/optimized_ts_model.pth'
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_rmsd': avg_val_rmsd,
                'config': model_config,
                'history': history
            }, save_path)
            
            print(f"  [BEST] Model saved! RMSD: {avg_val_rmsd:.4f}A")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        print()
    
    # 保存训练历史
    with open('models/optimized_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_rmsd'])
    plt.axhline(y=0.2, color='r', linestyle='--', label='Target (0.2A)')
    plt.axhline(y=0.5, color='orange', linestyle='--', label='Success (0.5A)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSD (A)')
    plt.legend()
    plt.title('Validation RMSD')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rates'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/optimized_training_curves.png', dpi=150)
    print(f"\nTraining curves saved to: models/optimized_training_curves.png")
    
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best val RMSD: {min(history['val_rmsd']):.4f} A")
    print(f"{'='*80}")

if __name__ == "__main__":
    train_optimized_model()

