"""
高级过渡态预测模型
包含更复杂的架构和优化技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """位置编码 - 用于处理分子中原子的空间关系"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiScaleGNN(nn.Module):
    """多尺度图神经网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 不同尺度的卷积层
        self.local_convs = nn.ModuleList()  # 局部特征
        self.global_convs = nn.ModuleList()  # 全局特征
        self.attention_convs = nn.ModuleList()  # 注意力特征
        
        # 第一层
        self.local_convs.append(GCNConv(input_dim, hidden_dim))
        self.global_convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
        self.attention_convs.append(TransformerConv(input_dim, hidden_dim, heads=4))
        
        # 后续层
        for _ in range(num_layers - 1):
            self.local_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.global_convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            self.attention_convs.append(TransformerConv(hidden_dim, hidden_dim, heads=4))
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x, edge_index, batch=None):
        """前向传播"""
        for i in range(self.num_layers):
            # 多尺度特征提取
            local_feat = self.local_convs[i](x, edge_index)
            global_feat = self.global_convs[i](x, edge_index)
            attention_feat = self.attention_convs[i](x, edge_index)
            
            # 特征融合
            combined_feat = torch.cat([local_feat, global_feat, attention_feat], dim=-1)
            x = self.feature_fusion(combined_feat)
            
            # 批归一化和激活
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        return x


class EnergyAwarePredictor(nn.Module):
    """能量感知的过渡态预测器"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.use_energy = config.get('use_energy_features', True)
        
        # 多尺度分子编码器
        self.molecular_encoder = MultiScaleGNN(
            input_dim=8,  # 原子类型one-hot编码
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # 能量编码器（如果使用能量特征）
        if self.use_energy:
            self.energy_encoder = nn.Sequential(
                nn.Linear(3, self.hidden_dim // 4),  # 反应物、过渡态、产物能量
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 4, self.hidden_dim // 2)
            )
        
        # 反应路径建模
        self.reaction_path_encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=True
        )
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 坐标预测网络
        coord_input_dim = self.hidden_dim * 2  # 双向LSTM输出
        if self.use_energy:
            coord_input_dim += self.hidden_dim // 2
        
        self.coordinate_predictor = nn.Sequential(
            nn.Linear(coord_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 3)  # 3D坐标
        )
        
        # 不确定性估计
        self.uncertainty_head = nn.Sequential(
            nn.Linear(coord_input_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),  # 每个坐标的不确定性
            nn.Softplus()  # 确保正值
        )
    
    def encode_molecule(self, graph_data):
        """编码分子图"""
        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = graph_data.batch if hasattr(graph_data, 'batch') else None
        
        # 获取节点特征
        node_features = self.molecular_encoder(x, edge_index, batch)
        
        # 全局池化
        if batch is not None:
            global_mean = global_mean_pool(node_features, batch)
            global_max = global_max_pool(node_features, batch)
        else:
            global_mean = torch.mean(node_features, dim=0, keepdim=True)
            global_max = torch.max(node_features, dim=0, keepdim=True)[0]
        
        global_features = torch.cat([global_mean, global_max], dim=-1)
        
        return node_features, global_features
    
    def forward(self, reactant_batch, product_batch, energies=None):
        """
        前向传播
        Args:
            reactant_batch: 反应物图数据
            product_batch: 产物图数据
            energies: 能量信息 [batch_size, 3] (reactant, ts, product)
        """
        # 编码反应物和产物
        r_node_feat, r_global_feat = self.encode_molecule(reactant_batch)
        p_node_feat, p_global_feat = self.encode_molecule(product_batch)
        
        # 构建反应路径序列 [反应物 -> 产物]
        batch_size = r_global_feat.size(0)
        reaction_sequence = torch.stack([r_global_feat, p_global_feat], dim=1)  # [batch, 2, hidden_dim]
        
        # LSTM建模反应路径
        lstm_out, _ = self.reaction_path_encoder(reaction_sequence)  # [batch, 2, hidden_dim*2]
        
        # 使用交叉注意力融合反应物和产物的原子级特征
        # 假设原子数相同且对应
        combined_node_feat = torch.stack([r_node_feat, p_node_feat], dim=1)  # [n_atoms, 2, hidden_dim]
        
        attended_feat, _ = self.cross_attention(
            combined_node_feat, combined_node_feat, combined_node_feat
        )
        attended_feat = attended_feat.mean(dim=1)  # [n_atoms, hidden_dim]
        
        # 获取反应路径的中间状态（过渡态位置）
        ts_context = lstm_out[:, 0, :] + lstm_out[:, 1, :]  # 简单融合
        ts_context = ts_context.unsqueeze(0).expand(attended_feat.size(0), -1)
        
        # 能量特征（如果可用）
        coord_input = torch.cat([attended_feat, ts_context], dim=-1)
        
        if self.use_energy and energies is not None:
            energy_feat = self.energy_encoder(energies)
            energy_feat = energy_feat.unsqueeze(0).expand(attended_feat.size(0), -1)
            coord_input = torch.cat([coord_input, energy_feat], dim=-1)
        
        # 预测坐标和不确定性
        predicted_coords = self.coordinate_predictor(coord_input)
        uncertainty = self.uncertainty_head(coord_input)
        
        return predicted_coords, uncertainty


class AdvancedTSLoss(nn.Module):
    """高级过渡态预测损失函数"""
    
    def __init__(self, coord_weight=1.0, smooth_weight=0.1, energy_weight=0.05, 
                 uncertainty_weight=0.01):
        super().__init__()
        self.coord_weight = coord_weight
        self.smooth_weight = smooth_weight
        self.energy_weight = energy_weight
        self.uncertainty_weight = uncertainty_weight
    
    def coordinate_loss(self, pred_coords, true_coords, uncertainty=None):
        """坐标预测损失（考虑不确定性）"""
        if uncertainty is not None:
            # 不确定性加权损失
            loss = torch.sum((pred_coords - true_coords) ** 2 / (2 * uncertainty ** 2) + 
                           torch.log(uncertainty))
            return loss / pred_coords.numel()
        else:
            return F.mse_loss(pred_coords, true_coords)
    
    def smoothness_loss(self, pred_coords, reactant_coords, product_coords):
        """路径平滑性损失"""
        # 过渡态应该在反应物和产物之间的合理位置
        mid_coords = (reactant_coords + product_coords) / 2
        
        # 距离中点的偏差
        deviation = torch.norm(pred_coords - mid_coords, dim=-1)
        
        # 反应路径的曲率约束
        r_to_ts = pred_coords - reactant_coords
        ts_to_p = product_coords - pred_coords
        
        # 鼓励平滑的路径（减少急转弯）
        curvature = torch.sum(r_to_ts * ts_to_p, dim=-1) / (
            torch.norm(r_to_ts, dim=-1) * torch.norm(ts_to_p, dim=-1) + 1e-8
        )
        curvature_loss = torch.mean((curvature + 1) ** 2)  # 鼓励负相关（平滑路径）
        
        return torch.mean(deviation) + 0.1 * curvature_loss
    
    def energy_consistency_loss(self, pred_coords, true_coords, energies):
        """能量一致性损失"""
        if energies is None:
            return torch.tensor(0.0, device=pred_coords.device)
        
        # 简化的能量一致性检查
        # 实际应用中可以使用更复杂的能量模型
        coord_diff = torch.norm(pred_coords - true_coords, dim=-1)
        energy_penalty = torch.mean(coord_diff * energies[:, 1])  # 使用过渡态能量加权
        
        return energy_penalty
    
    def forward(self, pred_coords, true_coords, reactant_coords, product_coords,
                uncertainty=None, energies=None):
        """计算总损失"""
        coord_loss = self.coordinate_loss(pred_coords, true_coords, uncertainty)
        smooth_loss = self.smoothness_loss(pred_coords, reactant_coords, product_coords)
        energy_loss = self.energy_consistency_loss(pred_coords, true_coords, energies)
        
        # 不确定性正则化
        uncertainty_reg = 0.0
        if uncertainty is not None:
            uncertainty_reg = torch.mean(uncertainty)  # 防止不确定性过大
        
        total_loss = (self.coord_weight * coord_loss + 
                     self.smooth_weight * smooth_loss +
                     self.energy_weight * energy_loss +
                     self.uncertainty_weight * uncertainty_reg)
        
        return total_loss, {
            'coord_loss': coord_loss.item(),
            'smooth_loss': smooth_loss.item(),
            'energy_loss': energy_loss.item(),
            'uncertainty_reg': uncertainty_reg if isinstance(uncertainty_reg, float) else uncertainty_reg.item(),
            'total_loss': total_loss.item()
        }


def test_advanced_model():
    """测试高级模型"""
    config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.1,
        'use_energy_features': True
    }
    
    model = EnergyAwarePredictor(config)
    print(f"Advanced model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    batch_size = 2
    n_atoms = 10
    
    # 模拟图数据
    x = torch.randn(n_atoms * batch_size, 8)
    edge_index = torch.randint(0, n_atoms * batch_size, (2, 20))
    batch = torch.repeat_interleave(torch.arange(batch_size), n_atoms)
    
    reactant_data = Data(x=x, edge_index=edge_index, batch=batch)
    product_data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # 能量数据
    energies = torch.randn(batch_size, 3)  # [reactant, ts, product] energies
    
    # 前向传播
    with torch.no_grad():
        coords, uncertainty = model(reactant_data, product_data, energies)
        print(f"Output coords shape: {coords.shape}")
        print(f"Output uncertainty shape: {uncertainty.shape}")
    
    print("Advanced model test passed!")


if __name__ == "__main__":
    test_advanced_model()