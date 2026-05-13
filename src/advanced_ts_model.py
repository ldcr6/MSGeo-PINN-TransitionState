"""
高级过渡态预测模型
融合多种特征工程和深度学习技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SchNet, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj
import numpy as np
import math
from typing import Tuple, Optional


class GeometricFeatureExtractor(nn.Module):
    """几何特征提取器"""
    
    def __init__(self, hidden_dim: int):
        super(GeometricFeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 距离特征编码
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # 角度特征编码
        self.angle_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # 二面角特征编码
        self.dihedral_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
    
    def calculate_distances(self, pos):
        """计算原子间距离"""
        n_atoms = pos.size(0)
        distances = torch.cdist(pos, pos)  # [n_atoms, n_atoms]
        return distances
    
    def calculate_angles(self, pos, edge_index):
        """计算键角"""
        if edge_index.size(1) < 2:
            return torch.zeros(1, device=pos.device)
        
        # 简化的角度计算
        angles = []
        for i in range(min(10, edge_index.size(1) - 1)):  # 限制计算数量
            try:
                atom1 = edge_index[0, i]
                atom2 = edge_index[1, i]
                atom3 = edge_index[0, i + 1] if i + 1 < edge_index.size(1) else edge_index[1, 0]
                
                v1 = pos[atom2] - pos[atom1]
                v2 = pos[atom3] - pos[atom2]
                
                cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
                angle = torch.acos(torch.clamp(cos_angle, -1 + 1e-6, 1 - 1e-6))
                angles.append(angle)
            except:
                continue
        
        if angles:
            return torch.stack(angles).mean().unsqueeze(0)
        else:
            return torch.zeros(1, device=pos.device)
    
    def forward(self, pos, edge_index):
        """提取几何特征"""
        # 距离特征
        distances = self.calculate_distances(pos)
        mean_distance = distances[distances > 0].mean().unsqueeze(0)
        distance_feat = self.distance_encoder(mean_distance.unsqueeze(0))
        
        # 角度特征
        angles = self.calculate_angles(pos, edge_index)
        angle_feat = self.angle_encoder(angles.unsqueeze(0))
        
        # 简化的二面角特征（使用距离方差作为代理）
        distance_var = distances[distances > 0].var().unsqueeze(0)
        dihedral_feat = self.dihedral_encoder(distance_var.unsqueeze(0))
        
        # 合并几何特征
        geometric_features = torch.cat([distance_feat, angle_feat, dihedral_feat], dim=-1)
        
        return geometric_features


class PhysicsInformedGNN(nn.Module):
    """物理启发的图神经网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 4):
        super(PhysicsInformedGNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # SchNet风格的连续滤波器
        self.distance_expansion = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GAT层用于注意力机制
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=False, dropout=0.1))
        
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1))
        
        # 物理约束层
        self.physics_constraint = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # 限制输出范围
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 层归一化
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
    
    def forward(self, x, edge_index, pos, edge_attr=None):
        """前向传播"""
        # 计算边特征（距离）
        row, col = edge_index
        edge_distances = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
        edge_features = self.distance_expansion(edge_distances)
        
        # GAT层处理
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x if i > 0 else None
            
            x = gat(x, edge_index)
            x = norm(x)
            
            # 残差连接
            if residual is not None and residual.shape == x.shape:
                x = x + residual
            
            x = F.gelu(x)
            
            # 物理约束
            x = self.physics_constraint(x)
        
        return x, edge_features


class MultiScaleAttention(nn.Module):
    """多尺度注意力机制"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super(MultiScaleAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 多尺度注意力
        self.local_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        
        # 尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        """多尺度注意力处理"""
        # 局部注意力（短程相互作用）
        local_out, _ = self.local_attention(x, x, x)
        
        # 全局注意力（长程相互作用）
        global_out, _ = self.global_attention(x, x, x)
        
        # 融合多尺度特征
        combined = torch.cat([local_out, global_out], dim=-1)
        fused = self.scale_fusion(combined)
        
        return fused


class AdvancedTransitionStatePredictor(nn.Module):
    """高级过渡态预测器"""
    
    def __init__(self, config: dict):
        super(AdvancedTransitionStatePredictor, self).__init__()
        
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # 几何特征提取器
        self.geometric_extractor = GeometricFeatureExtractor(self.hidden_dim)
        
        # 物理启发的GNN
        self.physics_gnn = PhysicsInformedGNN(
            input_dim=11,  # 原子类型one-hot
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # 多尺度注意力
        self.multi_scale_attention = MultiScaleAttention(self.hidden_dim)
        
        # 反应路径编码器
        self.reaction_path_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 6, self.hidden_dim * 2),  # 反应物+产物的多尺度特征
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
        # 坐标预测网络（更深更宽）
        # 计算输入维度：原子特征(256) + 反应路径(256) + 几何特征(384) = 896
        coord_input_dim = self.hidden_dim + self.hidden_dim + (self.hidden_dim + self.hidden_dim // 2)
        self.coord_predictor = nn.Sequential(
            nn.Linear(coord_input_dim, self.hidden_dim * 2),  # 原子特征+反应路径+几何特征
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.LayerNorm(self.hidden_dim // 4),
            nn.GELU(),
            
            nn.Linear(self.hidden_dim // 4, 3)  # 3D坐标
        )
        
        # 物理约束网络
        self.physics_refiner = nn.Sequential(
            nn.Linear(9, 64),  # 预测坐标+反应物坐标+产物坐标
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 3)  # 坐标修正
        )
        
        # 不确定性估计
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出0-1的不确定性分数
        )
    
    def encode_molecule(self, graph_data):
        """编码分子"""
        x = graph_data.x
        edge_index = graph_data.edge_index
        pos = graph_data.pos if hasattr(graph_data, 'pos') else torch.zeros(x.size(0), 3)
        batch = graph_data.batch if hasattr(graph_data, 'batch') else None
        
        # 物理启发的GNN编码
        node_features, edge_features = self.physics_gnn(x, edge_index, pos)
        
        # 多尺度注意力
        attended_features = self.multi_scale_attention(node_features.unsqueeze(0)).squeeze(0)
        
        # 几何特征提取
        geometric_features = self.geometric_extractor(pos, edge_index)
        
        # 全局池化
        if batch is not None:
            global_mean = global_mean_pool(attended_features, batch)
            global_max = global_max_pool(attended_features, batch)
            global_add = global_add_pool(attended_features, batch)
        else:
            global_mean = torch.mean(attended_features, dim=0, keepdim=True)
            global_max = torch.max(attended_features, dim=0, keepdim=True)[0]
            global_add = torch.sum(attended_features, dim=0, keepdim=True)
        
        # 合并全局特征
        global_features = torch.cat([global_mean, global_max, global_add], dim=-1)
        
        return attended_features, global_features, geometric_features
    
    def forward(self, reactant_batch, product_batch):
        """前向传播"""
        # 编码反应物和产物
        r_node_feat, r_global_feat, r_geom_feat = self.encode_molecule(reactant_batch)
        p_node_feat, p_global_feat, p_geom_feat = self.encode_molecule(product_batch)
        
        # 反应路径编码
        reaction_features = torch.cat([r_global_feat, p_global_feat], dim=-1)
        reaction_context = self.reaction_path_encoder(reaction_features)
        
        # 确保原子数匹配
        min_atoms = min(r_node_feat.size(0), p_node_feat.size(0))
        r_node_feat = r_node_feat[:min_atoms]
        p_node_feat = p_node_feat[:min_atoms]
        
        # 融合原子级特征
        fused_node_features = (r_node_feat + p_node_feat) / 2
        
        # 扩展反应上下文和几何特征
        reaction_context_expanded = reaction_context.expand(min_atoms, -1)
        geometric_context = (r_geom_feat + p_geom_feat) / 2
        geometric_context_expanded = geometric_context.expand(min_atoms, -1)
        
        # 合并所有特征
        combined_features = torch.cat([
            fused_node_features, 
            reaction_context_expanded,
            geometric_context_expanded
        ], dim=-1)
        
        # 预测坐标
        predicted_coords = self.coord_predictor(combined_features)
        
        # 物理约束修正
        reactant_coords = reactant_batch.pos[:min_atoms] if hasattr(reactant_batch, 'pos') else torch.zeros_like(predicted_coords)
        product_coords = product_batch.pos[:min_atoms] if hasattr(product_batch, 'pos') else torch.zeros_like(predicted_coords)
        
        physics_input = torch.cat([predicted_coords, reactant_coords, product_coords], dim=-1)
        physics_correction = self.physics_refiner(physics_input)
        
        # 最终坐标
        final_coords = predicted_coords + 0.05 * physics_correction  # 小幅物理修正
        
        # 不确定性估计
        uncertainty = self.uncertainty_estimator(fused_node_features).mean()
        
        return final_coords, uncertainty


class AdvancedTSPredictionLoss(nn.Module):
    """高级损失函数"""
    
    def __init__(self, coord_weight: float = 1.0, smooth_weight: float = 0.1, 
                 distance_weight: float = 0.2, angle_weight: float = 0.1,
                 physics_weight: float = 0.15, uncertainty_weight: float = 0.05):
        super(AdvancedTSPredictionLoss, self).__init__()
        self.coord_weight = coord_weight
        self.smooth_weight = smooth_weight
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.physics_weight = physics_weight
        self.uncertainty_weight = uncertainty_weight
    
    def coordinate_loss(self, pred_coords, true_coords):
        """坐标损失（组合MSE和Huber）"""
        mse_loss = F.mse_loss(pred_coords, true_coords)
        huber_loss = F.huber_loss(pred_coords, true_coords, delta=0.1)
        return 0.6 * mse_loss + 0.4 * huber_loss
    
    def physics_consistency_loss(self, pred_coords, reactant_coords, product_coords):
        """物理一致性损失"""
        # 能量守恒约束（简化版）
        pred_center = pred_coords.mean(dim=0)
        reactant_center = reactant_coords.mean(dim=0)
        product_center = product_coords.mean(dim=0)
        
        # 过渡态应该在反应路径上
        path_deviation = torch.norm(pred_center - (reactant_center + product_center) / 2)
        
        # 原子间距离的物理合理性
        pred_distances = torch.cdist(pred_coords, pred_coords)
        min_distance = pred_distances[pred_distances > 0].min()
        
        # 防止原子过于接近
        distance_penalty = torch.relu(0.5 - min_distance)  # 最小距离应大于0.5Å
        
        return path_deviation + distance_penalty
    
    def uncertainty_loss(self, uncertainty, coord_error):
        """不确定性损失"""
        # 不确定性应该与坐标误差相关
        target_uncertainty = torch.clamp(coord_error / 2.0, 0, 1)  # 归一化误差
        return F.mse_loss(uncertainty, target_uncertainty.detach())
    
    def forward(self, pred_coords, true_coords, reactant_coords, product_coords, uncertainty=None):
        """计算总损失"""
        coord_loss = self.coordinate_loss(pred_coords, true_coords)
        
        # 路径平滑性
        interpolated = 0.5 * (reactant_coords + product_coords)
        smooth_loss = F.mse_loss(pred_coords, interpolated)
        
        # 距离保持
        pred_dist = torch.cdist(pred_coords, pred_coords)
        true_dist = torch.cdist(true_coords, true_coords)
        distance_loss = F.mse_loss(pred_dist, true_dist)
        
        # 物理一致性
        physics_loss = self.physics_consistency_loss(pred_coords, reactant_coords, product_coords)
        
        # 角度保持（简化）
        angle_loss = torch.tensor(0.0, device=pred_coords.device)
        if pred_coords.size(0) >= 3:
            try:
                pred_angle = self.calculate_angle(pred_coords[:3])
                true_angle = self.calculate_angle(true_coords[:3])
                angle_loss = F.mse_loss(pred_angle, true_angle)
            except:
                pass
        
        # 不确定性损失
        uncertainty_loss = torch.tensor(0.0, device=pred_coords.device)
        if uncertainty is not None:
            coord_error = torch.norm(pred_coords - true_coords, dim=1).mean()
            uncertainty_loss = self.uncertainty_loss(uncertainty, coord_error)
        
        # 总损失
        total_loss = (self.coord_weight * coord_loss + 
                     self.smooth_weight * smooth_loss +
                     self.distance_weight * distance_loss +
                     self.angle_weight * angle_loss +
                     self.physics_weight * physics_loss +
                     self.uncertainty_weight * uncertainty_loss)
        
        return total_loss, {
            'coord_loss': coord_loss.item(),
            'smooth_loss': smooth_loss.item(),
            'distance_loss': distance_loss.item(),
            'angle_loss': angle_loss.item(),
            'physics_loss': physics_loss.item(),
            'uncertainty_loss': uncertainty_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def calculate_angle(self, coords):
        """计算三个原子的角度"""
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[1]
        cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
        return torch.acos(torch.clamp(cos_angle, -1 + 1e-6, 1 - 1e-6))


def test_advanced_model():
    """测试高级模型"""
    config = {
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout': 0.1
    }
    
    model = AdvancedTransitionStatePredictor(config)
    print(f"Advanced model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    n_atoms = 12
    x = torch.randn(n_atoms, 11)  # 原子特征
    edge_index = torch.randint(0, n_atoms, (2, 24))  # 边索引
    pos = torch.randn(n_atoms, 3)  # 坐标
    
    reactant_data = Data(x=x, edge_index=edge_index, pos=pos)
    product_data = Data(x=x, edge_index=edge_index, pos=pos)
    
    # 前向传播
    with torch.no_grad():
        output, uncertainty = model(reactant_data, product_data)
        print(f"Output shape: {output.shape}")
        print(f"Uncertainty: {uncertainty.item():.4f}")
    
    print("Advanced model test passed!")


if __name__ == "__main__":
    test_advanced_model()