"""
改进的过渡态预测模型
针对RMSD优化的架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Tuple
import math


class ImprovedMolecularGNN(nn.Module):
    """改进的分子图神经网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 4, dropout: float = 0.1):
        super(ImprovedMolecularGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 使用GAT层替代GCN以获得更好的注意力机制
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout))
        
        # 层归一化替代批归一化
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # 残差连接
        self.residual_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.residual_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.residual_layers.append(nn.Identity())
    
    def forward(self, x, edge_index, batch=None):
        """前向传播"""
        residual = x
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            x = conv(x, edge_index)
            x = norm(x)
            
            # 残差连接
            if i > 0:
                x = x + residual
            elif i == 0 and hasattr(self, 'residual_layers'):
                residual = self.residual_layers[0](residual)
                x = x + residual
            
            x = F.gelu(x)  # 使用GELU激活函数
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual = x
        
        return x


class ImprovedTransitionStatePredictor(nn.Module):
    """改进的过渡态结构预测器"""
    
    def __init__(self, config: dict):
        super(ImprovedTransitionStatePredictor, self).__init__()
        
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # 分子编码器
        self.molecular_encoder = ImprovedMolecularGNN(
            input_dim=11,  # 原子类型one-hot编码维度
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # 多尺度特征融合
        self.global_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 6, self.hidden_dim * 2),  # mean+max+add pooling for both molecules
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
        # 改进的注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.hidden_dim, self.dropout)
        
        # 坐标预测网络 - 更深的网络
        self.coord_predictor = nn.Sequential(
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
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_dim // 4, 3)  # 输出3D坐标
        )
        
        # 坐标细化网络
        self.coord_refiner = nn.Sequential(
            nn.Linear(6, 32),  # 输入：预测坐标 + 反应物坐标
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 3)  # 输出坐标修正
        )
    
    def encode_molecule(self, graph_data):
        """编码分子图"""
        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = graph_data.batch if hasattr(graph_data, 'batch') else None
        
        # 获取节点特征
        node_features = self.molecular_encoder(x, edge_index, batch)
        
        # 多尺度全局池化
        if batch is not None:
            global_mean = global_mean_pool(node_features, batch)
            global_max = global_max_pool(node_features, batch)
            global_add = global_add_pool(node_features, batch)
        else:
            global_mean = torch.mean(node_features, dim=0, keepdim=True)
            global_max = torch.max(node_features, dim=0, keepdim=True)[0]
            global_add = torch.sum(node_features, dim=0, keepdim=True)
        
        global_features = torch.cat([global_mean, global_max, global_add], dim=-1)
        
        return node_features, global_features
    
    def forward(self, reactant_batch, product_batch):
        """前向传播"""
        # 编码反应物和产物
        r_node_feat, r_global_feat = self.encode_molecule(reactant_batch)
        p_node_feat, p_global_feat = self.encode_molecule(product_batch)
        
        # 融合全局特征
        combined_global = torch.cat([r_global_feat, p_global_feat], dim=-1)
        reaction_context = self.global_fusion(combined_global)
        
        # 确保原子数匹配
        min_atoms = min(r_node_feat.size(0), p_node_feat.size(0))
        r_node_feat = r_node_feat[:min_atoms]
        p_node_feat = p_node_feat[:min_atoms]
        
        # 添加位置编码
        r_node_feat = self.pos_encoding(r_node_feat.unsqueeze(0)).squeeze(0)
        p_node_feat = self.pos_encoding(p_node_feat.unsqueeze(0)).squeeze(0)
        
        # 交叉注意力：反应物关注产物
        r_attended, _ = self.cross_attention(
            r_node_feat.unsqueeze(0), p_node_feat.unsqueeze(0), p_node_feat.unsqueeze(0)
        )
        r_attended = r_attended.squeeze(0)
        
        # 交叉注意力：产物关注反应物
        p_attended, _ = self.cross_attention(
            p_node_feat.unsqueeze(0), r_node_feat.unsqueeze(0), r_node_feat.unsqueeze(0)
        )
        p_attended = p_attended.squeeze(0)
        
        # 融合注意力特征
        fused_features = (r_attended + p_attended) / 2
        
        # 自注意力
        self_attended, _ = self.self_attention(
            fused_features.unsqueeze(0), fused_features.unsqueeze(0), fused_features.unsqueeze(0)
        )
        self_attended = self_attended.squeeze(0)
        
        # 添加反应上下文
        n_atoms = self_attended.size(0)
        if reaction_context.dim() == 1:
            reaction_context = reaction_context.unsqueeze(0)
        reaction_context_expanded = reaction_context.expand(n_atoms, -1)
        
        # 预测坐标
        combined_features = torch.cat([self_attended, reaction_context_expanded], dim=-1)
        predicted_coords = self.coord_predictor(combined_features)
        
        # 坐标细化（使用反应物坐标作为参考）
        reactant_coords = reactant_batch.pos[:min_atoms] if hasattr(reactant_batch, 'pos') else torch.zeros_like(predicted_coords)
        coord_input = torch.cat([predicted_coords, reactant_coords], dim=-1)
        coord_refinement = self.coord_refiner(coord_input)
        
        # 最终坐标
        final_coords = predicted_coords + 0.1 * coord_refinement  # 小幅修正
        
        return final_coords


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)


class ImprovedTSPredictionLoss(nn.Module):
    """改进的过渡态预测损失函数"""
    
    def __init__(self, coord_weight: float = 1.0, smooth_weight: float = 0.1, 
                 distance_weight: float = 0.2, angle_weight: float = 0.1):
        super(ImprovedTSPredictionLoss, self).__init__()
        self.coord_weight = coord_weight
        self.smooth_weight = smooth_weight
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
    
    def coordinate_loss(self, pred_coords, true_coords):
        """坐标预测损失（MSE + Huber）"""
        mse_loss = F.mse_loss(pred_coords, true_coords)
        huber_loss = F.huber_loss(pred_coords, true_coords, delta=0.1)
        return 0.7 * mse_loss + 0.3 * huber_loss
    
    def distance_preservation_loss(self, pred_coords, true_coords):
        """距离保持损失"""
        # 计算原子间距离
        pred_dist = torch.cdist(pred_coords, pred_coords)
        true_dist = torch.cdist(true_coords, true_coords)
        
        # 只考虑相邻原子的距离（避免计算所有原子对）
        n_atoms = pred_coords.size(0)
        mask = torch.triu(torch.ones(n_atoms, n_atoms), diagonal=1).bool()
        
        pred_dist_masked = pred_dist[mask]
        true_dist_masked = true_dist[mask]
        
        return F.mse_loss(pred_dist_masked, true_dist_masked)
    
    def smoothness_loss(self, pred_coords, reactant_coords, product_coords):
        """路径平滑性损失"""
        # 过渡态应该在反应物和产物之间的合理位置
        interpolated = 0.5 * (reactant_coords + product_coords)
        return F.mse_loss(pred_coords, interpolated)
    
    def angle_preservation_loss(self, pred_coords, true_coords):
        """角度保持损失（简化版）"""
        if pred_coords.size(0) < 3:
            return torch.tensor(0.0, device=pred_coords.device)
        
        # 计算前三个原子的角度
        def calc_angle(coords):
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[1]
            cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
            return torch.acos(torch.clamp(cos_angle, -1 + 1e-6, 1 - 1e-6))
        
        pred_angle = calc_angle(pred_coords)
        true_angle = calc_angle(true_coords)
        
        return F.mse_loss(pred_angle, true_angle)
    
    def forward(self, pred_coords, true_coords, reactant_coords, product_coords):
        """计算总损失"""
        coord_loss = self.coordinate_loss(pred_coords, true_coords)
        smooth_loss = self.smoothness_loss(pred_coords, reactant_coords, product_coords)
        distance_loss = self.distance_preservation_loss(pred_coords, true_coords)
        angle_loss = self.angle_preservation_loss(pred_coords, true_coords)
        
        total_loss = (self.coord_weight * coord_loss + 
                     self.smooth_weight * smooth_loss +
                     self.distance_weight * distance_loss +
                     self.angle_weight * angle_loss)
        
        return total_loss, {
            'coord_loss': coord_loss.item(),
            'smooth_loss': smooth_loss.item(),
            'distance_loss': distance_loss.item(),
            'angle_loss': angle_loss.item(),
            'total_loss': total_loss.item()
        }


def test_improved_model():
    """测试改进的模型"""
    config = {
        'hidden_dim': 128,
        'num_layers': 4,
        'dropout': 0.1
    }
    
    model = ImprovedTransitionStatePredictor(config)
    print(f"Improved model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    n_atoms = 10
    x = torch.randn(n_atoms, 11)  # 节点特征
    edge_index = torch.randint(0, n_atoms, (2, 20))  # 边索引
    pos = torch.randn(n_atoms, 3)  # 坐标
    
    reactant_data = Data(x=x, edge_index=edge_index, pos=pos)
    product_data = Data(x=x, edge_index=edge_index, pos=pos)
    
    # 前向传播
    with torch.no_grad():
        output = model(reactant_data, product_data)
        print(f"Output shape: {output.shape}")
    
    print("Improved model test passed!")


if __name__ == "__main__":
    test_improved_model()