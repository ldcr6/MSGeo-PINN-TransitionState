#!/usr/bin/env python3
"""
增强特征工程模块
包含几何特征、物理化学描述符、SOAP、ACSF等
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
import math


class GeometricFeatureExtractor(nn.Module):
    """几何特征提取器"""
    
    def __init__(self, max_atoms: int = 50):
        super().__init__()
        self.max_atoms = max_atoms
        
    def extract_bond_features(self, coords: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        """提取键长、键角特征"""
        n_atoms = coords.shape[0]
        
        # 计算距离矩阵
        distances = torch.cdist(coords, coords)
        
        # 键长特征 (只考虑合理的键长范围 0.5-3.0 Å)
        bond_mask = (distances > 0.5) & (distances < 3.0)
        bond_lengths = distances[bond_mask]
        
        # 键角特征
        angles = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                for k in range(j+1, n_atoms):
                    if bond_mask[i,j] and bond_mask[j,k]:
                        # 计算角度 i-j-k
                        v1 = coords[i] - coords[j]
                        v2 = coords[k] - coords[j]
                        cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
                        angle = torch.acos(torch.clamp(cos_angle, -1, 1))
                        angles.append(angle)
        
        # 统计特征
        bond_stats = torch.tensor([
            bond_lengths.mean() if len(bond_lengths) > 0 else 0,
            bond_lengths.std() if len(bond_lengths) > 0 else 0,
            bond_lengths.min() if len(bond_lengths) > 0 else 0,
            bond_lengths.max() if len(bond_lengths) > 0 else 0
        ])
        
        angle_stats = torch.tensor([
            torch.stack(angles).mean() if angles else 0,
            torch.stack(angles).std() if angles else 0
        ])
        
        return torch.cat([bond_stats, angle_stats])
    
    def center_of_mass_alignment(self, coords: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        """质心对齐"""
        # 原子质量 (简化版本)
        atomic_masses = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 16: 32.065}
        masses = torch.tensor([atomic_masses.get(int(t), 12.0) for t in atom_types])
        
        # 计算质心
        center_of_mass = torch.sum(coords * masses.unsqueeze(1), dim=0) / torch.sum(masses)
        
        # 平移到质心
        centered_coords = coords - center_of_mass
        
        return centered_coords
    
    def forward(self, coords: torch.Tensor, atom_types: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取所有几何特征"""
        # 质心对齐
        aligned_coords = self.center_of_mass_alignment(coords, atom_types)
        
        # 键长键角特征
        bond_features = self.extract_bond_features(coords, atom_types)
        
        # 分子尺寸特征
        size_features = torch.tensor([
            torch.norm(coords, dim=1).max(),  # 最大原子距离
            torch.norm(coords, dim=1).mean(), # 平均原子距离
            coords.std(dim=0).mean()          # 坐标标准差
        ])
        
        return {
            'aligned_coords': aligned_coords,
            'bond_features': bond_features,
            'size_features': size_features
        }


class SOAPDescriptor(nn.Module):
    """SOAP (Smooth Overlap of Atomic Positions) 描述符"""
    
    def __init__(self, n_max: int = 8, l_max: int = 6, r_cut: float = 5.0):
        super().__init__()
        self.n_max = n_max
        self.l_max = l_max
        self.r_cut = r_cut
        
    def radial_basis(self, distances: torch.Tensor) -> torch.Tensor:
        """径向基函数"""
        n_indices = torch.arange(1, self.n_max + 1, dtype=torch.float32)
        basis = torch.sin(n_indices * math.pi * distances.unsqueeze(-1) / self.r_cut) / distances.unsqueeze(-1)
        return basis
    
    def forward(self, coords: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        """计算SOAP描述符"""
        n_atoms = coords.shape[0]
        soap_features = []
        
        for i in range(n_atoms):
            # 计算与其他原子的距离
            distances = torch.norm(coords - coords[i], dim=1)
            mask = (distances < self.r_cut) & (distances > 0)
            
            if mask.sum() > 0:
                neighbor_distances = distances[mask]
                neighbor_types = atom_types[mask]
                
                # 径向基函数
                radial_features = self.radial_basis(neighbor_distances)
                
                # 按原子类型聚合
                type_features = []
                for atom_type in torch.unique(neighbor_types):
                    type_mask = neighbor_types == atom_type
                    if type_mask.sum() > 0:
                        type_radial = radial_features[type_mask].mean(dim=0)
                        type_features.append(type_radial)
                
                if type_features:
                    atom_soap = torch.cat(type_features)
                else:
                    atom_soap = torch.zeros(self.n_max)
            else:
                atom_soap = torch.zeros(self.n_max)
            
            soap_features.append(atom_soap)
        
        return torch.stack(soap_features)


class ACSFDescriptor(nn.Module):
    """ACSF (Atom-Centered Symmetry Functions) 描述符"""
    
    def __init__(self, r_cut: float = 5.0, n_radial: int = 10, n_angular: int = 5):
        super().__init__()
        self.r_cut = r_cut
        self.n_radial = n_radial
        self.n_angular = n_angular
        
        # 径向对称函数参数
        self.eta_radial = torch.linspace(0.1, 2.0, n_radial)
        self.rs_radial = torch.linspace(0.0, r_cut, n_radial)
        
        # 角度对称函数参数
        self.eta_angular = torch.linspace(0.1, 1.0, n_angular)
        self.zeta = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])[:n_angular]
        
    def cutoff_function(self, distances: torch.Tensor) -> torch.Tensor:
        """截断函数"""
        mask = distances <= self.r_cut
        fc = torch.zeros_like(distances)
        fc[mask] = 0.5 * (torch.cos(math.pi * distances[mask] / self.r_cut) + 1)
        return fc
    
    def radial_symmetry_functions(self, distances: torch.Tensor) -> torch.Tensor:
        """径向对称函数"""
        fc = self.cutoff_function(distances)
        
        g2_features = []
        for eta, rs in zip(self.eta_radial, self.rs_radial):
            g2 = torch.exp(-eta * (distances - rs)**2) * fc
            g2_features.append(g2.sum())
        
        return torch.stack(g2_features)
    
    def angular_symmetry_functions(self, coords: torch.Tensor, center_idx: int) -> torch.Tensor:
        """角度对称函数"""
        center = coords[center_idx]
        neighbors = coords[torch.arange(len(coords)) != center_idx]
        
        if len(neighbors) < 2:
            return torch.zeros(self.n_angular)
        
        distances = torch.norm(neighbors - center, dim=1)
        mask = distances <= self.r_cut
        
        if mask.sum() < 2:
            return torch.zeros(self.n_angular)
        
        valid_neighbors = neighbors[mask]
        valid_distances = distances[mask]
        
        g4_features = []
        for eta, zeta in zip(self.eta_angular, self.zeta):
            g4_sum = 0
            for i in range(len(valid_neighbors)):
                for j in range(i+1, len(valid_neighbors)):
                    rij = valid_distances[i]
                    rik = valid_distances[j]
                    
                    # 计算角度
                    vec_ij = valid_neighbors[i] - center
                    vec_ik = valid_neighbors[j] - center
                    cos_angle = torch.dot(vec_ij, vec_ik) / (rij * rik)
                    
                    # 角度对称函数
                    fc_ij = self.cutoff_function(rij)
                    fc_ik = self.cutoff_function(rik)
                    
                    g4_term = (1 + cos_angle)**zeta * torch.exp(-eta * (rij**2 + rik**2)) * fc_ij * fc_ik
                    g4_sum += g4_term
            
            g4_features.append(g4_sum)
        
        return torch.stack(g4_features)
    
    def forward(self, coords: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        """计算ACSF描述符"""
        n_atoms = coords.shape[0]
        acsf_features = []
        
        for i in range(n_atoms):
            # 计算距离
            distances = torch.norm(coords - coords[i], dim=1)
            distances = distances[distances > 0]  # 排除自身
            
            # 径向对称函数
            g2 = self.radial_symmetry_functions(distances)
            
            # 角度对称函数
            g4 = self.angular_symmetry_functions(coords, i)
            
            # 合并特征
            atom_acsf = torch.cat([g2, g4])
            acsf_features.append(atom_acsf)
        
        return torch.stack(acsf_features)


class MolecularFingerprint(nn.Module):
    """分子指纹生成器"""
    
    def __init__(self, fingerprint_size: int = 1024, radius: int = 3):
        super().__init__()
        self.fingerprint_size = fingerprint_size
        self.radius = radius
        
    def atom_environment_hash(self, coords: torch.Tensor, atom_types: torch.Tensor, 
                            center_idx: int, radius: int) -> int:
        """计算原子环境的哈希值"""
        center_type = int(atom_types[center_idx])
        
        # 找到半径内的邻居
        distances = torch.norm(coords - coords[center_idx], dim=1)
        neighbors = torch.where((distances <= radius) & (distances > 0))[0]
        
        # 创建环境描述
        env_description = [center_type]
        for neighbor_idx in neighbors:
            neighbor_type = int(atom_types[neighbor_idx])
            distance = float(distances[neighbor_idx])
            env_description.extend([neighbor_type, int(distance * 10)])  # 量化距离
        
        # 简单哈希
        return hash(tuple(sorted(env_description))) % self.fingerprint_size
    
    def forward(self, coords: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        """生成分子指纹"""
        fingerprint = torch.zeros(self.fingerprint_size)
        
        for i in range(len(coords)):
            for r in range(1, self.radius + 1):
                hash_value = self.atom_environment_hash(coords, atom_types, i, r)
                fingerprint[hash_value] = 1.0
        
        return fingerprint


class EnhancedFeatureExtractor(nn.Module):
    """增强特征提取器 - 整合所有特征"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.geometric_extractor = GeometricFeatureExtractor(config.get('max_atoms', 50))
        self.soap_descriptor = SOAPDescriptor(
            n_max=config.get('soap_n_max', 8),
            l_max=config.get('soap_l_max', 6),
            r_cut=config.get('soap_r_cut', 5.0)
        )
        self.acsf_descriptor = ACSFDescriptor(
            r_cut=config.get('acsf_r_cut', 5.0),
            n_radial=config.get('acsf_n_radial', 10),
            n_angular=config.get('acsf_n_angular', 5)
        )
        self.fingerprint_generator = MolecularFingerprint(
            fingerprint_size=config.get('fingerprint_size', 512),
            radius=config.get('fingerprint_radius', 3)
        )
        
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.get_total_feature_dim(), config.get('hidden_dim', 256)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.get('hidden_dim', 256), config.get('hidden_dim', 256))
        )
    
    def get_total_feature_dim(self) -> int:
        """计算总特征维度"""
        # 这里需要根据实际特征维度计算
        geometric_dim = 6  # bond_features(4) + size_features(3) - 1
        soap_dim = 8  # n_max
        acsf_dim = 15  # n_radial + n_angular
        fingerprint_dim = 512
        
        return geometric_dim + soap_dim + acsf_dim + fingerprint_dim
    
    def forward(self, coords: torch.Tensor, atom_types: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取所有增强特征"""
        # 几何特征
        geometric_features = self.geometric_extractor(coords, atom_types)
        
        # SOAP描述符
        soap_features = self.soap_descriptor(coords, atom_types)
        
        # ACSF描述符
        acsf_features = self.acsf_descriptor(coords, atom_types)
        
        # 分子指纹
        fingerprint = self.fingerprint_generator(coords, atom_types)
        
        # 合并特征 (取平均值用于原子级特征)
        combined_features = torch.cat([
            geometric_features['bond_features'],
            geometric_features['size_features'],
            soap_features.mean(dim=0),  # 原子级特征的平均
            acsf_features.mean(dim=0),  # 原子级特征的平均
            fingerprint
        ])
        
        # 特征融合
        fused_features = self.feature_fusion(combined_features)
        
        return {
            'geometric': geometric_features,
            'soap': soap_features,
            'acsf': acsf_features,
            'fingerprint': fingerprint,
            'fused': fused_features,
            'aligned_coords': geometric_features['aligned_coords']
        }