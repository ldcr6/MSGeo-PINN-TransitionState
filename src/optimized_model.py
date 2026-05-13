"""
优化的过渡态预测模型
专注于RMSD性能提升
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
import numpy as np


class EnhancedMolecularEncoder(nn.Module):
    """增强的分子编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 4):
        super(EnhancedMolecularEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # GAT层用于更好的注意力机制
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=8, concat=False, dropout=0.1))
        
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=8, concat=False, dropout=0.1))
        
        # 层归一化
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # 残差连接的投影层
        self.residual_proj = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x, edge_index):
        """前向传播"""
        # 初始残差连接
        residual = self.residual_proj(x)
        
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x = gat(x, edge_index)
            x = norm(x)
            
            # 残差连接
            if i == 0:
                x = x + residual
            else:
                x = x + residual
            
            x = F.gelu(x)
            residual = x
        
        return x


class CoordinatePredictor(nn.Module):
    """专门的坐标预测网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super(CoordinatePredictor, self).__init__()
        
        # 多层感知机，专门优化坐标预测
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            
            nn.Linear(hidden_dim // 4, 3)  # 输出3D坐标
        )
        
        # 坐标细化层
        self.refiner = nn.Sequential(
            nn.Linear(6, 32),  # 预测坐标 + 参考坐标
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 3)
        )
    
    def forward(self, features, reference_coords):
        """预测坐标"""
        # 基础预测
        base_coords = self.layers(features)
        
        # 细化预测
        refine_input = torch.cat([base_coords, reference_coords], dim=-1)
        refinement = self.refiner(refine_input)
        
        # 最终坐标（小幅修正）
        final_coords = base_coords + 0.1 * refinement
        
        return final_coords


class OptimizedTransitionStatePredictor(nn.Module):
    """优化的过渡态预测器"""
    
    def __init__(self, config: dict):
        super(OptimizedTransitionStatePredictor, self).__init__()
        
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # 分子编码器
        self.molecular_encoder = EnhancedMolecularEncoder(
            input_dim=11,  # 原子类型one-hot
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # 反应特征融合
        self.reaction_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),  # 反应物+产物的全局特征
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 坐标预测器
        self.coord_predictor = CoordinatePredictor(
            input_dim=self.hidden_dim * 2,  # 原子特征 + 反应上下文
            hidden_dim=self.hidden_dim
        )
    
    def encode_molecule(self, graph_data):
        """编码分子"""
        x = graph_data.x
        edge_index = graph_data.edge_index
        
        # 获取节点特征
        node_features = self.molecular_encoder(x, edge_index)
        
        # 全局特征
        global_mean = torch.mean(node_features, dim=0, keepdim=True)
        global_max = torch.max(node_features, dim=0, keepdim=True)[0]
        global_features = torch.cat([global_mean, global_max], dim=-1)
        
        return node_features, global_features
    
    def forward(self, reactant_batch, product_batch):
        """前向传播"""
        # 编码反应物和产物
        r_node_feat, r_global_feat = self.encode_molecule(reactant_batch)
        p_node_feat, p_global_feat = self.encode_molecule(product_batch)
        
        # 融合反应特征
        reaction_features = torch.cat([r_global_feat, p_global_feat], dim=-1)
        reaction_context = self.reaction_fusion(reaction_features)
        
        # 确保原子数匹配
        min_atoms = min(r_node_feat.size(0), p_node_feat.size(0))
        r_node_feat = r_node_feat[:min_atoms]
        p_node_feat = p_node_feat[:min_atoms]
        
        # 交叉注意力：反应物关注产物
        combined_features = torch.stack([r_node_feat, p_node_feat], dim=1)  # [n_atoms, 2, hidden_dim]
        attended_features, _ = self.cross_attention(
            combined_features, combined_features, combined_features
        )
        attended_features = attended_features.mean(dim=1)  # [n_atoms, hidden_dim]
        
        # 添加反应上下文
        reaction_context_expanded = reaction_context.expand(min_atoms, -1)
        combined_input = torch.cat([attended_features, reaction_context_expanded], dim=-1)
        
        # 获取参考坐标（反应物和产物的平均）
        reactant_coords = reactant_batch.pos[:min_atoms] if hasattr(reactant_batch, 'pos') else torch.zeros(min_atoms, 3)
        product_coords = product_batch.pos[:min_atoms] if hasattr(product_batch, 'pos') else torch.zeros(min_atoms, 3)
        reference_coords = (reactant_coords + product_coords) / 2
        
        # 预测坐标
        predicted_coords = self.coord_predictor(combined_input, reference_coords)
        
        return predicted_coords


class OptimizedTSLoss(nn.Module):
    """优化的损失函数"""
    
    def __init__(self, coord_weight: float = 1.0, smooth_weight: float = 0.2, 
                 distance_weight: float = 0.3):
        super(OptimizedTSLoss, self).__init__()
        self.coord_weight = coord_weight
        self.smooth_weight = smooth_weight
        self.distance_weight = distance_weight
    
    def coordinate_loss(self, pred_coords, true_coords):
        """坐标损失"""
        # 组合MSE和L1损失
        mse_loss = F.mse_loss(pred_coords, true_coords)
        l1_loss = F.l1_loss(pred_coords, true_coords)
        return 0.7 * mse_loss + 0.3 * l1_loss
    
    def distance_preservation_loss(self, pred_coords, true_coords):
        """距离保持损失"""
        pred_dist = torch.cdist(pred_coords, pred_coords)
        true_dist = torch.cdist(true_coords, true_coords)
        
        # 只考虑重要的距离（避免计算所有原子对）
        n_atoms = pred_coords.size(0)
        if n_atoms > 10:
            # 随机采样一些原子对
            indices = torch.randperm(n_atoms)[:10]
            pred_dist = pred_dist[indices][:, indices]
            true_dist = true_dist[indices][:, indices]
        
        return F.mse_loss(pred_dist, true_dist)
    
    def smoothness_loss(self, pred_coords, reactant_coords, product_coords):
        """路径平滑性损失"""
        # 过渡态应该在反应路径上
        interpolated = 0.5 * (reactant_coords + product_coords)
        return F.mse_loss(pred_coords, interpolated)
    
    def forward(self, pred_coords, true_coords, reactant_coords, product_coords):
        """计算总损失"""
        coord_loss = self.coordinate_loss(pred_coords, true_coords)
        smooth_loss = self.smoothness_loss(pred_coords, reactant_coords, product_coords)
        distance_loss = self.distance_preservation_loss(pred_coords, true_coords)
        
        total_loss = (self.coord_weight * coord_loss + 
                     self.smooth_weight * smooth_loss +
                     self.distance_weight * distance_loss)
        
        return total_loss, {
            'coord_loss': coord_loss.item(),
            'smooth_loss': smooth_loss.item(),
            'distance_loss': distance_loss.item(),
            'total_loss': total_loss.item()
        }


def test_optimized_model():
    """测试优化模型"""
    config = {
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout': 0.1
    }
    
    model = OptimizedTransitionStatePredictor(config)
    print(f"Optimized model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    n_atoms = 12
    x = torch.randn(n_atoms, 11)  # 原子特征
    edge_index = torch.randint(0, n_atoms, (2, 24))  # 边索引
    pos = torch.randn(n_atoms, 3)  # 坐标
    
    reactant_data = Data(x=x, edge_index=edge_index, pos=pos)
    product_data = Data(x=x, edge_index=edge_index, pos=pos)
    
    # 前向传播
    with torch.no_grad():
        output = model(reactant_data, product_data)
        print(f"Output shape: {output.shape}")
    
    print("Optimized model test passed!")


if __name__ == "__main__":
    test_optimized_model()