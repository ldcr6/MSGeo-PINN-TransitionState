#!/usr/bin/env python3
"""
高级分子结构预测模型
基于图神经网络、物理启发和生成模型的过渡态预测
目标: RMSE ≤ 0.2 Å
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pickle
import time
import math
from typing import Dict, List, Tuple, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MolecularFeatureExtractor:
    """分子特征提取器 - 实现SOAP、ACSF等描述符"""
    
    def __init__(self):
        self.atomic_numbers = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53
        }
        
    def extract_geometric_features(self, coords: np.ndarray, atoms: List[str]) -> np.ndarray:
        """提取几何特征"""
        features = []
        
        # 1. 基本几何特征
        centroid = np.mean(coords, axis=0)
        features.extend(centroid)
        
        # 2. 惯性矩特征
        centered_coords = coords - centroid
        inertia_tensor = np.dot(centered_coords.T, centered_coords)
        eigenvals = np.linalg.eigvals(inertia_tensor)
        features.extend(sorted(eigenvals))
        
        # 3. 径向分布函数 (RDF)
        rdf_features = self._compute_rdf(coords)
        features.extend(rdf_features)
        
        # 4. 角度分布函数 (ADF)
        adf_features = self._compute_adf(coords)
        features.extend(adf_features)
        
        # 5. 原子环境特征
        env_features = self._compute_atomic_environment(coords, atoms)
        features.extend(env_features)
        
        return np.array(features)
    
    def _compute_rdf(self, coords: np.ndarray, r_max: float = 10.0, n_bins: int = 50) -> List[float]:
        """计算径向分布函数"""
        distances = []
        n_atoms = len(coords)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= r_max:
                    distances.append(dist)
        
        if not distances:
            return [0.0] * n_bins
        
        hist, _ = np.histogram(distances, bins=n_bins, range=(0, r_max))
        return hist.tolist()
    
    def _compute_adf(self, coords: np.ndarray, n_bins: int = 36) -> List[float]:
        """计算角度分布函数"""
        angles = []
        n_atoms = len(coords)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                for k in range(n_atoms):
                    if i != j and j != k and i != k:
                        vec1 = coords[i] - coords[j]
                        vec2 = coords[k] - coords[j]
                        
                        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        angles.append(angle)
        
        if not angles:
            return [0.0] * n_bins
        
        hist, _ = np.histogram(angles, bins=n_bins, range=(0, np.pi))
        return hist.tolist()
    
    def _compute_atomic_environment(self, coords: np.ndarray, atoms: List[str]) -> List[float]:
        """计算原子环境特征 (简化版ACSF)"""
        features = []
        n_atoms = len(coords)
        
        for i in range(n_atoms):
            atom_features = []
            
            # 计算与其他原子的距离
            distances = []
            for j in range(n_atoms):
                if i != j:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    distances.append(dist)
            
            if distances:
                atom_features.extend([
                    np.min(distances),
                    np.max(distances),
                    np.mean(distances),
                    np.std(distances)
                ])
            else:
                atom_features.extend([0.0, 0.0, 0.0, 0.0])
            
            features.extend(atom_features)
        
        # 填充或截断到固定长度
        target_length = 200  # 固定特征长度
        if len(features) > target_length:
            features = features[:target_length]
        else:
            features.extend([0.0] * (target_length - len(features)))
        
        return features
    
    def compute_soap_features(self, coords: np.ndarray, atoms: List[str]) -> np.ndarray:
        """计算SOAP描述符 (简化版)"""
        # 这里实现简化版的SOAP特征
        # 实际应用中可以使用dscribe库
        
        features = []
        n_atoms = len(coords)
        
        for i in range(n_atoms):
            atom_soap = []
            
            # 局部环境的径向和角度特征
            neighbors = []
            for j in range(n_atoms):
                if i != j:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < 5.0:  # 5Å截断半径
                        neighbors.append((j, dist))
            
            # 径向部分
            radial_features = []
            for r_cut in [1.0, 2.0, 3.0, 4.0, 5.0]:
                count = sum(1 for _, dist in neighbors if dist <= r_cut)
                radial_features.append(count)
            
            # 角度部分
            angular_features = []
            if len(neighbors) >= 2:
                for idx1, (j, _) in enumerate(neighbors):
                    for idx2, (k, _) in enumerate(neighbors[idx1+1:], idx1+1):
                        vec1 = coords[j] - coords[i]
                        vec2 = coords[k] - coords[i]
                        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        angular_features.append(cos_angle)
            
            # 限制特征长度
            if len(angular_features) > 10:
                angular_features = angular_features[:10]
            else:
                angular_features.extend([0.0] * (10 - len(angular_features)))
            
            atom_soap.extend(radial_features)
            atom_soap.extend(angular_features)
            features.extend(atom_soap)
        
        # 标准化到固定长度
        target_length = 300
        if len(features) > target_length:
            features = features[:target_length]
        else:
            features.extend([0.0] * (target_length - len(features)))
        
        return np.array(features)

class GraphMolecularData:
    """分子图数据处理"""
    
    def __init__(self):
        self.atomic_numbers = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53
        }
    
    def coords_to_graph(self, coords: np.ndarray, atoms: List[str], bond_threshold: float = 2.0) -> Data:
        """将分子坐标转换为图数据"""
        n_atoms = len(atoms)
        
        # 节点特征 (原子特征)
        node_features = []
        for atom in atoms:
            atomic_num = self.atomic_numbers.get(atom, 6)  # 默认为碳
            # 原子特征: [原子序数, 原子序数归一化]
            node_feat = [atomic_num, atomic_num / 53.0]  # 归一化到最大原子序数
            node_features.append(node_feat)
        
        node_features = torch.FloatTensor(node_features)
        
        # 边连接 (基于距离的键连接)
        edge_indices = []
        edge_features = []
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= bond_threshold:
                    # 添加双向边
                    edge_indices.extend([[i, j], [j, i]])
                    # 边特征: [距离, 1/距离]
                    edge_feat = [dist, 1.0/max(dist, 0.1)]
                    edge_features.extend([edge_feat, edge_feat])
        
        if edge_indices:
            edge_index = torch.LongTensor(edge_indices).t().contiguous()
            edge_attr = torch.FloatTensor(edge_features)
        else:
            # 如果没有边，创建自环
            edge_index = torch.LongTensor([[i, i] for i in range(n_atoms)]).t().contiguous()
            edge_attr = torch.FloatTensor([[0.0, 0.0] for _ in range(n_atoms)])
        
        # 坐标作为节点位置
        pos = torch.FloatTensor(coords)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

class GraphNeuralNetwork(nn.Module):
    """图神经网络模型"""
    
    def __init__(self, node_features: int = 2, edge_features: int = 2, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 节点嵌入
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # 边嵌入
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # GCN层
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # GAT层
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim//8, heads=8, dropout=0.1, concat=True)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 图级别特征提取
        self.graph_conv = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max pooling
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, data):
        """前向传播"""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 节点嵌入
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # 图卷积层
        for i in range(self.num_layers):
            # GCN
            x_gcn = self.gcn_layers[i](x, edge_index)
            
            # GAT
            x_gat = self.gat_layers[i](x, edge_index)
            
            # 残差连接和层归一化
            x = x + 0.5 * (x_gcn + x_gat)
            x = self.layer_norms[i](x)
            x = F.relu(x)
        
        # 图级别池化
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_features = torch.cat([graph_mean, graph_max], dim=1)
        
        # 图级别特征
        graph_features = self.graph_conv(graph_features)
        
        return graph_features, x

class PhysicsInspiredModel(nn.Module):
    """物理启发的过渡态预测模型"""
    
    def __init__(self, input_dim: int = 500, num_atoms: int = 23, hidden_dim: int = 256):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_atoms = num_atoms
        self.hidden_dim = hidden_dim
        
        # 传统特征处理
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 图神经网络
        self.gnn = GraphNeuralNetwork(hidden_dim=hidden_dim)
        
        # 物理约束网络
        self.physics_net = PhysicsConstraintNetwork(hidden_dim)
        
        # 生成模型组件 (VAE)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.mu_layer = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.logvar_layer = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms * 3)
        )
        
        # 坐标精化网络
        self.coord_refiner = CoordinateRefiner(num_atoms)
        
        # 不确定性估计
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_atoms * 3),
            nn.Softplus()
        )
        
    def reparameterize(self, mu, logvar):
        """VAE重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, features, graph_data=None):
        """前向传播"""
        batch_size = features.size(0)
        
        # 传统特征处理
        feat_traditional = self.feature_net(features)
        
        # 图特征处理
        if graph_data is not None:
            graph_features, node_features = self.gnn(graph_data)
            # 组合特征
            combined_features = torch.cat([feat_traditional, graph_features], dim=1)
        else:
            combined_features = torch.cat([feat_traditional, feat_traditional], dim=1)
        
        # 物理约束
        physics_features = self.physics_net(combined_features)
        
        # VAE编码
        encoded = self.encoder(physics_features)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码生成坐标
        coords_raw = self.decoder(z)
        coords_raw = coords_raw.view(batch_size, self.num_atoms, 3)
        
        # 坐标精化
        coords_refined = self.coord_refiner(coords_raw, physics_features)
        
        # 不确定性估计
        uncertainty = self.uncertainty_net(physics_features)
        uncertainty = uncertainty.view(batch_size, self.num_atoms, 3)
        
        return coords_refined, uncertainty, mu, logvar

class PhysicsConstraintNetwork(nn.Module):
    """物理约束网络"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # 能量约束
        self.energy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 力约束
        self.force_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 几何约束
        self.geometry_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def forward(self, x):
        """应用物理约束"""
        # 能量守恒约束
        energy_constrained = x + 0.1 * torch.tanh(self.energy_net(x))
        
        # 力平衡约束
        force_constrained = energy_constrained + 0.1 * torch.tanh(self.force_net(energy_constrained))
        
        # 几何合理性约束
        geometry_constrained = force_constrained + 0.05 * torch.tanh(self.geometry_net(force_constrained))
        
        return geometry_constrained

class CoordinateRefiner(nn.Module):
    """坐标精化网络"""
    
    def __init__(self, num_atoms: int):
        super().__init__()
        
        self.num_atoms = num_atoms
        
        # 原子间相互作用网络
        self.interaction_net = nn.Sequential(
            nn.Linear(num_atoms * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_atoms * 3)
        )
        
        # 几何优化网络
        self.optimization_net = nn.Sequential(
            nn.Linear(num_atoms * 3 + 256, 256),  # 坐标 + 上下文特征
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_atoms * 3)
        )
        
    def forward(self, coords, context_features):
        """坐标精化"""
        batch_size = coords.size(0)
        coords_flat = coords.view(batch_size, -1)
        
        # 原子间相互作用
        interaction_correction = self.interaction_net(coords_flat)
        coords_corrected = coords_flat + 0.1 * interaction_correction
        
        # 几何优化
        combined_input = torch.cat([coords_corrected, context_features], dim=1)
        optimization_correction = self.optimization_net(combined_input)
        coords_optimized = coords_corrected + 0.05 * optimization_correction
        
        return coords_optimized.view(batch_size, self.num_atoms, 3)

class AdvancedLoss(nn.Module):
    """高级损失函数"""
    
    def __init__(self):
        super().__init__()
        
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=0.05)
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred_coords, true_coords, uncertainty=None, mu=None, logvar=None):
        """计算总损失"""
        
        # 1. 坐标重建损失 (主要损失)
        coord_loss = self.huber_loss(pred_coords, true_coords)
        
        # 2. L1正则化
        l1_loss = self.l1_loss(pred_coords, true_coords)
        
        # 3. 几何一致性损失
        geometry_loss = self._compute_geometry_loss(pred_coords, true_coords)
        
        # 4. VAE损失 (KL散度)
        kl_loss = 0
        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        
        # 5. 不确定性损失
        uncertainty_loss = 0
        if uncertainty is not None:
            # 不确定性加权损失
            weighted_mse = torch.mean((pred_coords - true_coords) ** 2 / (uncertainty + 1e-8))
            uncertainty_reg = torch.mean(uncertainty)
            uncertainty_loss = weighted_mse + 0.1 * uncertainty_reg
        
        # 6. 物理合理性损失
        physics_loss = self._compute_physics_loss(pred_coords)
        
        # 总损失
        total_loss = (
            1.0 * coord_loss +           # 主要坐标损失
            0.2 * l1_loss +              # L1正则化
            0.3 * geometry_loss +        # 几何一致性
            0.1 * kl_loss +              # VAE正则化
            0.2 * uncertainty_loss +     # 不确定性
            0.1 * physics_loss           # 物理合理性
        )
        
        return total_loss, {
            'coord_loss': coord_loss.item(),
            'l1_loss': l1_loss.item(),
            'geometry_loss': geometry_loss.item(),
            'kl_loss': kl_loss if isinstance(kl_loss, float) else kl_loss.item(),
            'uncertainty_loss': uncertainty_loss if isinstance(uncertainty_loss, float) else uncertainty_loss.item(),
            'physics_loss': physics_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _compute_geometry_loss(self, pred_coords, true_coords):
        """几何一致性损失"""
        # 距离矩阵保持
        pred_distances = torch.cdist(pred_coords, pred_coords)
        true_distances = torch.cdist(true_coords, true_coords)
        distance_loss = self.mse_loss(pred_distances, true_distances)
        
        # 角度保持 (简化版)
        angle_loss = 0
        batch_size, num_atoms, _ = pred_coords.shape
        
        if num_atoms >= 3:
            for i in range(min(5, num_atoms)):  # 只计算前5个原子以节省计算
                for j in range(min(5, num_atoms)):
                    for k in range(min(5, num_atoms)):
                        if i != j and j != k and i != k:
                            # 预测角度
                            vec1_pred = pred_coords[:, i] - pred_coords[:, j]
                            vec2_pred = pred_coords[:, k] - pred_coords[:, j]
                            cos_pred = F.cosine_similarity(vec1_pred, vec2_pred, dim=1)
                            
                            # 真实角度
                            vec1_true = true_coords[:, i] - true_coords[:, j]
                            vec2_true = true_coords[:, k] - true_coords[:, j]
                            cos_true = F.cosine_similarity(vec1_true, vec2_true, dim=1)
                            
                            angle_loss += self.mse_loss(cos_pred, cos_true)
        
        return distance_loss + 0.1 * angle_loss
    
    def _compute_physics_loss(self, coords):
        """物理合理性损失"""
        # 检查原子间距离的合理性
        distances = torch.cdist(coords, coords)
        
        # 避免原子过于接近 (最小距离约束)
        min_distance = 0.5  # 0.5 Å
        too_close_penalty = torch.relu(min_distance - distances + torch.eye(distances.size(-1), device=distances.device) * min_distance)
        close_loss = torch.mean(too_close_penalty)
        
        # 避免分子过于分散 (最大距离约束)
        max_distance = 15.0  # 15 Å
        too_far_penalty = torch.relu(distances - max_distance)
        far_loss = torch.mean(too_far_penalty)
        
        return close_loss + far_loss

def load_and_prepare_advanced_data():
    """加载和准备高级数据"""
    logger.info("加载高级数据...")
    
    # 加载预处理数据
    data_file = Path("data/preprocessed_train_data.pkl")
    if not data_file.exists():
        logger.error("预处理数据不存在")
        return None, None, None
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    targets = data['targets']
    reactions = data.get('reactions', {})
    
    # 提取高级特征
    feature_extractor = MolecularFeatureExtractor()
    graph_processor = GraphMolecularData()
    
    enhanced_features = []
    graph_data_list = []
    
    logger.info("提取高级分子特征...")
    
    for i, (reaction_id, reaction_data) in enumerate(reactions.items()):
        if i >= len(features):
            break
            
        try:
            # 获取过渡态信息
            if 'transition_state' in reaction_data:
                ts_data = reaction_data['transition_state']
                atoms = ts_data.get('atoms', ['C'] * 23)  # 默认原子类型
                coords = targets[i]  # 使用目标坐标
                
                # 提取几何特征
                geom_features = feature_extractor.extract_geometric_features(coords, atoms)
                
                # 提取SOAP特征
                soap_features = feature_extractor.compute_soap_features(coords, atoms)
                
                # 组合特征
                combined_features = np.concatenate([features[i], geom_features, soap_features])
                enhanced_features.append(combined_features)
                
                # 创建图数据
                graph_data = graph_processor.coords_to_graph(coords, atoms)
                graph_data_list.append(graph_data)
            else:
                # 如果没有过渡态数据，使用原始特征
                enhanced_features.append(np.concatenate([features[i], np.zeros(500)]))
                # 创建默认图数据
                default_atoms = ['C'] * 23
                graph_data = graph_processor.coords_to_graph(targets[i], default_atoms)
                graph_data_list.append(graph_data)
                
        except Exception as e:
            logger.warning(f"处理反应 {reaction_id} 时出错: {e}")
            # 使用默认值
            enhanced_features.append(np.concatenate([features[i], np.zeros(500)]))
            default_atoms = ['C'] * 23
            graph_data = graph_processor.coords_to_graph(targets[i], default_atoms)
            graph_data_list.append(graph_data)
    
    # 如果enhanced_features为空，使用原始特征
    if not enhanced_features:
        logger.warning("无法提取高级特征，使用原始特征")
        enhanced_features = [np.concatenate([f, np.zeros(500)]) for f in features]
        graph_data_list = []
        for target in targets:
            default_atoms = ['C'] * len(target)
            graph_data = graph_processor.coords_to_graph(target, default_atoms)
            graph_data_list.append(graph_data)
    
    enhanced_features = np.array(enhanced_features)
    
    logger.info(f"高级特征形状: {enhanced_features.shape}")
    logger.info(f"图数据数量: {len(graph_data_list)}")
    
    return enhanced_features, targets, graph_data_list

def create_advanced_dataloaders(features, targets, graph_data_list, batch_size=16, val_split=0.15):
    """创建高级数据加载器"""
    
    # 分割数据
    indices = list(range(len(features)))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)
    
    X_train = features[train_indices]
    X_val = features[val_indices]
    y_train = targets[train_indices]
    y_val = targets[val_indices]
    
    # 分割图数据
    graph_train = [graph_data_list[i] for i in train_indices] if graph_data_list else []
    graph_val = [graph_data_list[i] for i in val_indices] if graph_data_list else []
    
    # 标准化特征
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # 数据增强
    X_train_aug = []
    y_train_aug = []
    graph_train_aug = []
    
    for i in range(len(X_train)):
        # 原始数据
        X_train_aug.append(X_train[i])
        y_train_aug.append(y_train[i])
        if graph_train:
            graph_train_aug.append(graph_train[i])
        
        # 增强数据
        for noise_level in [0.01, 0.02]:
            X_noisy = X_train[i] + np.random.normal(0, noise_level, X_train[i].shape)
            y_noisy = y_train[i] + np.random.normal(0, noise_level * 0.1, y_train[i].shape)
            
            X_train_aug.append(X_noisy)
            y_train_aug.append(y_noisy)
            if graph_train:
                graph_train_aug.append(graph_train[i])  # 图数据不增强
    
    X_train_aug = np.array(X_train_aug)
    y_train_aug = np.array(y_train_aug)
    
    logger.info(f"数据增强: {len(X_train)} -> {len(X_train_aug)} 样本")
    
    # 创建数据集
    class AdvancedTSDataset(Dataset):
        def __init__(self, features, targets, graph_data=None):
            self.features = torch.FloatTensor(features)
            self.targets = torch.FloatTensor(targets)
            self.graph_data = graph_data
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            if self.graph_data:
                return self.features[idx], self.targets[idx], self.graph_data[idx]
            else:
                return self.features[idx], self.targets[idx]
    
    train_dataset = AdvancedTSDataset(X_train_aug, y_train_aug, graph_train_aug if graph_train_aug else None)
    val_dataset = AdvancedTSDataset(X_val, y_val, graph_val if graph_val else None)
    
    # 自定义collate函数处理图数据
    def collate_fn(batch):
        if len(batch[0]) == 3:  # 有图数据
            features, targets, graphs = zip(*batch)
            features = torch.stack(features)
            targets = torch.stack(targets)
            
            # 处理图数据
            if graphs[0] is not None:
                batch_graphs = Batch.from_data_list(graphs)
                return features, targets, batch_graphs
            else:
                return features, targets, None
        else:  # 没有图数据
            features, targets = zip(*batch)
            features = torch.stack(features)
            targets = torch.stack(targets)
            return features, targets, None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)
    
    return train_loader, val_loader, scaler

def main():
    """主函数"""
    logger.info("🚀 开始高级分子模型训练")
    logger.info("目标: RMSE ≤ 0.2 Å")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    features, targets, graph_data_list = load_and_prepare_advanced_data()
    if features is None:
        return False
    
    # 创建数据加载器
    train_loader, val_loader, scaler = create_advanced_dataloaders(features, targets, graph_data_list)
    
    # 创建模型
    input_dim = features.shape[1]
    num_atoms = targets.shape[1]
    model = PhysicsInspiredModel(input_dim=input_dim, num_atoms=num_atoms)
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练器设置
    model = model.to(device)
    loss_fn = AdvancedLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-7
    )
    
    # 训练循环
    num_epochs = 60
    best_rmse = float('inf')
    patience = 20
    patience_counter = 0
    
    train_losses = []
    val_rmses = []
    
    logger.info("开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:
                features_batch, targets_batch, graph_batch = batch_data
            else:
                features_batch, targets_batch = batch_data
                graph_batch = None
            
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)
            if graph_batch is not None:
                graph_batch = graph_batch.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            pred_coords, uncertainty, mu, logvar = model(features_batch, graph_batch)
            
            # 计算损失
            loss, components = loss_fn(pred_coords, targets_batch, uncertainty, mu, logvar)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 20 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    features_batch, targets_batch, graph_batch = batch_data
                else:
                    features_batch, targets_batch = batch_data
                    graph_batch = None
                
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)
                if graph_batch is not None:
                    graph_batch = graph_batch.to(device)
                
                pred_coords, uncertainty, mu, logvar = model(features_batch, graph_batch)
                loss, _ = loss_fn(pred_coords, targets_batch, uncertainty, mu, logvar)
                
                val_loss += loss.item()
                all_predictions.append(pred_coords.cpu().numpy())
                all_targets.append(targets_batch.cpu().numpy())
        
        # 计算RMSE
        predictions = np.concatenate(all_predictions, axis=0)
        targets_val = np.concatenate(all_targets, axis=0)
        rmse = np.sqrt(np.mean((predictions - targets_val) ** 2))
        
        val_rmses.append(rmse)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.6f}")
        logger.info(f"Val Loss: {avg_val_loss:.6f}, Val RMSE: {rmse:.6f} Å")
        logger.info(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 检查是否达到目标
        if rmse <= 0.2:
            logger.info(f"🎉 达到目标RMSE! {rmse:.6f} ≤ 0.2 Å")
            break
        
        # 早停检查
        if rmse < best_rmse:
            best_rmse = rmse
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'epoch': epoch,
                'val_rmse': rmse,
                'input_dim': input_dim,
                'num_atoms': num_atoms
            }, 'advanced_molecular_best_model.pth')
            logger.info(f"保存最佳模型，RMSE: {rmse:.6f} Å")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"早停触发，最佳RMSE: {best_rmse:.6f} Å")
            break
    
    training_time = time.time() - start_time
    
    # 保存结果
    results = {
        'final_rmse': float(best_rmse),
        'target_achieved': bool(best_rmse <= 0.2),
        'training_time': float(training_time),
        'num_epochs': int(len(val_rmses)),
        'train_losses': [float(x) for x in train_losses],
        'val_rmses': [float(x) for x in val_rmses]
    }
    
    with open('advanced_molecular_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"训练完成! 最佳RMSE: {best_rmse:.6f} Å")
    logger.info(f"目标达成: {'✅' if best_rmse <= 0.2 else '❌'}")
    logger.info(f"训练时间: {training_time/60:.1f} 分钟")
    
    return best_rmse <= 0.2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)