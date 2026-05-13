"""
工具函数
"""

import numpy as np
import torch
from typing import Tuple, List
from ase import Atoms
from ase.io import read, write
import os
import logging
from pathlib import Path


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    计算两个坐标集之间的RMSD
    使用Kabsch算法进行最优对齐
    """
    try:
        from rmsd import kabsch_rmsd
        return kabsch_rmsd(coords1, coords2)
    except ImportError:
        # 简化版RMSD计算（不进行对齐）
        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def align_molecules(coords1: np.ndarray, coords2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Kabsch算法对齐两个分子结构
    """
    try:
        from rmsd import kabsch
        # 中心化
        coords1_centered = coords1 - np.mean(coords1, axis=0)
        coords2_centered = coords2 - np.mean(coords2, axis=0)
        
        # 计算旋转矩阵
        rotation_matrix = kabsch(coords1_centered, coords2_centered)
        
        # 应用旋转
        coords1_aligned = np.dot(coords1_centered, rotation_matrix)
        
        return coords1_aligned, coords2_centered
    except ImportError:
        print("Warning: rmsd package not available, returning original coordinates")
        return coords1, coords2


def create_xyz_file(atomic_numbers: List[int], coordinates: np.ndarray, filename: str):
    """
    创建XYZ格式文件
    """
    atoms = Atoms(numbers=atomic_numbers, positions=coordinates)
    write(filename, atoms)


def read_xyz_file(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取XYZ文件
    返回: (原子序数, 坐标)
    """
    try:
        atoms = read(filename)
        return atoms.get_atomic_numbers(), atoms.get_positions()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None


def validate_xyz_format(filename: str) -> bool:
    """
    验证XYZ文件格式是否正确
    """
    try:
        atomic_numbers, positions = read_xyz_file(filename)
        if atomic_numbers is None or positions is None:
            return False
        
        # 检查原子数是否匹配
        if len(atomic_numbers) != len(positions):
            return False
        
        # 检查坐标维度
        if positions.shape[1] != 3:
            return False
        
        return True
    except:
        return False


def batch_rmsd_calculation(pred_coords_list: List[np.ndarray], 
                          true_coords_list: List[np.ndarray]) -> List[float]:
    """
    批量计算RMSD
    """
    rmsd_values = []
    
    for pred_coords, true_coords in zip(pred_coords_list, true_coords_list):
        try:
            rmsd_val = calculate_rmsd(pred_coords, true_coords)
            rmsd_values.append(rmsd_val)
        except Exception as e:
            print(f"Error calculating RMSD: {e}")
            rmsd_values.append(float('inf'))
    
    return rmsd_values


def calculate_success_rate(rmsd_values: List[float], threshold: float = 0.5) -> float:
    """
    计算成功率（RMSD小于阈值的比例）
    """
    if not rmsd_values:
        return 0.0
    
    success_count = sum(1 for rmsd in rmsd_values if rmsd <= threshold)
    return success_count / len(rmsd_values) * 100


def create_sample_data():
    """
    创建示例数据用于测试
    """
    # 创建一个简单的水分子示例
    atomic_numbers = [8, 1, 1]  # O, H, H
    
    # 反应物坐标
    reactant_coords = np.array([
        [0.0, 0.0, 0.0],      # O
        [0.96, 0.0, 0.0],     # H
        [-0.24, 0.93, 0.0]    # H
    ])
    
    # 产物坐标（稍微不同）
    product_coords = np.array([
        [0.0, 0.0, 0.0],      # O
        [0.98, 0.0, 0.0],     # H
        [-0.26, 0.95, 0.0]    # H
    ])
    
    # 过渡态坐标（中间状态）
    ts_coords = (reactant_coords + product_coords) / 2
    
    # 创建目录
    sample_dir = "data/sample/reaction_001"
    os.makedirs(sample_dir, exist_ok=True)
    
    # 保存文件
    create_xyz_file(atomic_numbers, reactant_coords, 
                   os.path.join(sample_dir, "reactant.xyz"))
    create_xyz_file(atomic_numbers, product_coords, 
                   os.path.join(sample_dir, "product.xyz"))
    create_xyz_file(atomic_numbers, ts_coords, 
                   os.path.join(sample_dir, "ts.xyz"))
    
    print(f"Sample data created in {sample_dir}")


def print_model_summary(model):
    """
    打印模型摘要信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print("=" * 50)


if __name__ == "__main__":
    # 创建示例数据
    create_sample_data()
    print("Utility functions ready!")

def setup_logging(log_file: str = 'training.log') -> logging.Logger:
    """设置日志记录"""
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """加载检查点"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']