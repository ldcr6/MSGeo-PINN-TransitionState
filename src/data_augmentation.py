#!/usr/bin/env python3
"""
数据增强模块
包含坐标微扰、反应类型混合等提升模型鲁棒性的方法
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from scipy.spatial.transform import Rotation


class CoordinateAugmentation:
    """坐标数据增强"""
    
    def __init__(self, config: Dict):
        self.noise_std = config.get('noise_std', 0.1)
        self.rotation_prob = config.get('rotation_prob', 0.5)
        self.translation_std = config.get('translation_std', 0.5)
        self.scaling_range = config.get('scaling_range', (0.95, 1.05))
        
    def add_gaussian_noise(self, coords: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(coords) * self.noise_std
        return coords + noise
    
    def random_rotation(self, coords: torch.Tensor) -> torch.Tensor:
        """随机旋转"""
        if random.random() > self.rotation_prob:
            return coords
        
        # 生成随机旋转矩阵
        rotation = Rotation.random()
        rotation_matrix = torch.tensor(rotation.as_matrix(), dtype=coords.dtype, device=coords.device)
        
        # 计算质心
        center = coords.mean(dim=0)
        
        # 旋转坐标
        centered_coords = coords - center
        rotated_coords = torch.matmul(centered_coords, rotation_matrix.T)
        
        return rotated_coords + center
    
    def random_translation(self, coords: torch.Tensor) -> torch.Tensor:
        """随机平移"""
        translation = torch.randn(3, device=coords.device) * self.translation_std
        return coords + translation
    
    def random_scaling(self, coords: torch.Tensor) -> torch.Tensor:
        """随机缩放"""
        scale_factor = random.uniform(*self.scaling_range)
        center = coords.mean(dim=0)
        return (coords - center) * scale_factor + center
    
    def augment_coordinates(self, coords: torch.Tensor, augment_types: List[str] = None) -> torch.Tensor:
        """应用多种坐标增强"""
        if augment_types is None:
            augment_types = ['noise', 'rotation', 'translation', 'scaling']
        
        augmented_coords = coords.clone()
        
        for aug_type in augment_types:
            if aug_type == 'noise':
                augmented_coords = self.add_gaussian_noise(augmented_coords)
            elif aug_type == 'rotation':
                augmented_coords = self.random_rotation(augmented_coords)
            elif aug_type == 'translation':
                augmented_coords = self.random_translation(augmented_coords)
            elif aug_type == 'scaling':
                augmented_coords = self.random_scaling(augmented_coords)
        
        return augmented_coords


class ReactionTypeMixing:
    """反应类型混合增强"""
    
    def __init__(self, config: Dict):
        self.mix_prob = config.get('mix_prob', 0.3)
        self.mix_alpha = config.get('mix_alpha', 0.2)
        
    def mixup_reactions(self, batch_data: List[Dict]) -> List[Dict]:
        """Mixup数据增强"""
        if len(batch_data) < 2:
            return batch_data
        
        augmented_batch = []
        
        for i, sample in enumerate(batch_data):
            if random.random() < self.mix_prob and i < len(batch_data) - 1:
                # 选择另一个样本进行混合
                j = random.randint(0, len(batch_data) - 1)
                if i != j:
                    other_sample = batch_data[j]
                    
                    # 生成混合系数
                    lam = np.random.beta(self.mix_alpha, self.mix_alpha)
                    
                    # 混合坐标
                    mixed_sample = {
                        'reactant_coords': lam * sample['reactant_coords'] + (1 - lam) * other_sample['reactant_coords'],
                        'product_coords': lam * sample['product_coords'] + (1 - lam) * other_sample['product_coords'],
                        'ts_coords': lam * sample['ts_coords'] + (1 - lam) * other_sample['ts_coords'],
                        'atom_types': sample['atom_types'],  # 保持原子类型不变
                        'mix_ratio': lam,
                        'mixed_with': j
                    }
                    augmented_batch.append(mixed_sample)
                else:
                    augmented_batch.append(sample)
            else:
                augmented_batch.append(sample)
        
        return augmented_batch
    
    def cutmix_reactions(self, batch_data: List[Dict]) -> List[Dict]:
        """CutMix数据增强（原子级别）"""
        if len(batch_data) < 2:
            return batch_data
        
        augmented_batch = []
        
        for i, sample in enumerate(batch_data):
            if random.random() < self.mix_prob and i < len(batch_data) - 1:
                j = random.randint(0, len(batch_data) - 1)
                if i != j:
                    other_sample = batch_data[j]
                    
                    # 随机选择要替换的原子
                    n_atoms = sample['reactant_coords'].shape[0]
                    n_replace = random.randint(1, n_atoms // 4)  # 替换1/4的原子
                    replace_indices = random.sample(range(n_atoms), n_replace)
                    
                    mixed_sample = {
                        'reactant_coords': sample['reactant_coords'].clone(),
                        'product_coords': sample['product_coords'].clone(),
                        'ts_coords': sample['ts_coords'].clone(),
                        'atom_types': sample['atom_types'].clone()
                    }
                    
                    # 替换选定的原子坐标
                    for idx in replace_indices:
                        mixed_sample['reactant_coords'][idx] = other_sample['reactant_coords'][idx]
                        mixed_sample['product_coords'][idx] = other_sample['product_coords'][idx]
                        mixed_sample['ts_coords'][idx] = other_sample['ts_coords'][idx]
                        mixed_sample['atom_types'][idx] = other_sample['atom_types'][idx]
                    
                    augmented_batch.append(mixed_sample)
                else:
                    augmented_batch.append(sample)
            else:
                augmented_batch.append(sample)
        
        return augmented_batch


class PhysicsAwareAugmentation:
    """物理感知的数据增强"""
    
    def __init__(self, config: Dict):
        self.bond_constraint_weight = config.get('bond_constraint_weight', 1.0)
        self.energy_constraint_weight = config.get('energy_constraint_weight', 0.5)
        
    def constrained_perturbation(self, coords: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        """约束性坐标扰动（保持化学合理性）"""
        perturbed_coords = coords.clone()
        
        # 计算当前键长
        distances = torch.cdist(coords, coords)
        
        # 定义合理的键长范围
        bond_ranges = {
            (1, 1): (0.7, 0.8),    # H-H
            (1, 6): (1.0, 1.2),    # H-C
            (6, 6): (1.3, 1.6),    # C-C
            (6, 8): (1.2, 1.5),    # C-O
            (6, 7): (1.3, 1.5),    # C-N
        }
        
        # 对每个原子添加约束性扰动
        for i in range(len(coords)):
            # 计算与邻居的距离
            neighbor_distances = distances[i]
            
            # 生成候选扰动
            perturbation = torch.randn(3, device=coords.device) * 0.1
            candidate_coord = coords[i] + perturbation
            
            # 检查键长约束
            valid_perturbation = True
            for j in range(len(coords)):
                if i != j:
                    new_distance = torch.norm(candidate_coord - coords[j])
                    atom_pair = tuple(sorted([int(atom_types[i]), int(atom_types[j])]))
                    
                    if atom_pair in bond_ranges:
                        min_dist, max_dist = bond_ranges[atom_pair]
                        if new_distance < min_dist * 0.8 or new_distance > max_dist * 1.2:
                            valid_perturbation = False
                            break
            
            if valid_perturbation:
                perturbed_coords[i] = candidate_coord
        
        return perturbed_coords
    
    def energy_guided_augmentation(self, coords: torch.Tensor, atom_types: torch.Tensor, 
                                 energy_model: Optional[nn.Module] = None) -> torch.Tensor:
        """能量引导的数据增强"""
        if energy_model is None:
            return self.constrained_perturbation(coords, atom_types)
        
        # 使用能量模型评估扰动的合理性
        original_energy = energy_model(coords, atom_types)
        
        best_coords = coords.clone()
        best_energy = original_energy
        
        # 尝试多个扰动
        for _ in range(10):
            candidate_coords = self.constrained_perturbation(coords, atom_types)
            candidate_energy = energy_model(candidate_coords, atom_types)
            
            # 选择能量变化合理的扰动
            energy_diff = abs(candidate_energy - original_energy)
            if energy_diff < 0.5:  # 能量变化不超过0.5 eV
                best_coords = candidate_coords
                break
        
        return best_coords


class TemporalAugmentation:
    """时间序列数据增强（用于反应路径）"""
    
    def __init__(self, config: Dict):
        self.time_warp_prob = config.get('time_warp_prob', 0.3)
        self.interpolation_noise = config.get('interpolation_noise', 0.05)
        
    def time_warping(self, reaction_path: List[torch.Tensor]) -> List[torch.Tensor]:
        """时间扭曲增强"""
        if random.random() > self.time_warp_prob:
            return reaction_path
        
        n_frames = len(reaction_path)
        if n_frames < 3:
            return reaction_path
        
        # 生成扭曲的时间索引
        original_indices = np.linspace(0, n_frames - 1, n_frames)
        warp_strength = 0.2
        warped_indices = original_indices + np.random.normal(0, warp_strength, n_frames)
        warped_indices = np.clip(warped_indices, 0, n_frames - 1)
        warped_indices = np.sort(warped_indices)
        
        # 插值生成新的路径
        warped_path = []
        for i in range(n_frames):
            # 找到最近的帧进行插值
            idx = warped_indices[i]
            lower_idx = int(np.floor(idx))
            upper_idx = min(int(np.ceil(idx)), n_frames - 1)
            
            if lower_idx == upper_idx:
                warped_frame = reaction_path[lower_idx]
            else:
                alpha = idx - lower_idx
                warped_frame = (1 - alpha) * reaction_path[lower_idx] + alpha * reaction_path[upper_idx]
            
            warped_path.append(warped_frame)
        
        return warped_path
    
    def path_interpolation_noise(self, reaction_path: List[torch.Tensor]) -> List[torch.Tensor]:
        """路径插值噪声"""
        noisy_path = []
        
        for coords in reaction_path:
            noise = torch.randn_like(coords) * self.interpolation_noise
            noisy_coords = coords + noise
            noisy_path.append(noisy_coords)
        
        return noisy_path


class AdaptiveAugmentation:
    """自适应数据增强"""
    
    def __init__(self, config: Dict):
        self.coord_aug = CoordinateAugmentation(config)
        self.reaction_mix = ReactionTypeMixing(config)
        self.physics_aug = PhysicsAwareAugmentation(config)
        self.temporal_aug = TemporalAugmentation(config)
        
        self.augmentation_history = []
        self.performance_threshold = config.get('performance_threshold', 0.1)
        
    def select_augmentation_strategy(self, current_loss: float, epoch: int) -> List[str]:
        """根据当前性能选择增强策略"""
        strategies = []
        
        # 基础增强（总是应用）
        strategies.append('coordinate')
        
        # 根据损失值决定是否应用更强的增强
        if current_loss > self.performance_threshold:
            strategies.extend(['reaction_mix', 'physics_aware'])
        
        # 训练后期减少增强强度
        if epoch > 50:
            strategies = ['coordinate']  # 只保留基础增强
        
        return strategies
    
    def apply_augmentation(self, batch_data: List[Dict], strategies: List[str]) -> List[Dict]:
        """应用选定的增强策略"""
        augmented_data = batch_data.copy()
        
        for strategy in strategies:
            if strategy == 'coordinate':
                for sample in augmented_data:
                    sample['reactant_coords'] = self.coord_aug.augment_coordinates(sample['reactant_coords'])
                    sample['product_coords'] = self.coord_aug.augment_coordinates(sample['product_coords'])
                    sample['ts_coords'] = self.coord_aug.augment_coordinates(sample['ts_coords'])
            
            elif strategy == 'reaction_mix':
                augmented_data = self.reaction_mix.mixup_reactions(augmented_data)
            
            elif strategy == 'physics_aware':
                for sample in augmented_data:
                    sample['ts_coords'] = self.physics_aug.constrained_perturbation(
                        sample['ts_coords'], sample['atom_types']
                    )
        
        return augmented_data
    
    def update_performance_history(self, loss: float, accuracy: float):
        """更新性能历史"""
        self.augmentation_history.append({
            'loss': loss,
            'accuracy': accuracy
        })
        
        # 保持最近100个记录
        if len(self.augmentation_history) > 100:
            self.augmentation_history.pop(0)


class AugmentationScheduler:
    """增强调度器"""
    
    def __init__(self, config: Dict):
        self.initial_strength = config.get('initial_strength', 1.0)
        self.final_strength = config.get('final_strength', 0.3)
        self.decay_epochs = config.get('decay_epochs', 100)
        
    def get_augmentation_strength(self, epoch: int) -> float:
        """获取当前epoch的增强强度"""
        if epoch >= self.decay_epochs:
            return self.final_strength
        
        # 线性衰减
        decay_ratio = epoch / self.decay_epochs
        strength = self.initial_strength * (1 - decay_ratio) + self.final_strength * decay_ratio
        
        return strength
    
    def adjust_augmentation_config(self, config: Dict, strength: float) -> Dict:
        """根据强度调整增强配置"""
        adjusted_config = config.copy()
        
        # 调整各种增强参数
        adjusted_config['noise_std'] = config.get('noise_std', 0.1) * strength
        adjusted_config['mix_prob'] = config.get('mix_prob', 0.3) * strength
        adjusted_config['rotation_prob'] = config.get('rotation_prob', 0.5) * strength
        
        return adjusted_config