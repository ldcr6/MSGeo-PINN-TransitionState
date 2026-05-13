"""
过渡态预测模型
基于图神经网络的分子结构预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Tuple


class MolecularGNN(nn.Module):
    """分子图神经网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super(MolecularGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 批归一化
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, batch=None):
        """前向传播"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class TransitionStatePredictor(nn.Module):
    """过渡态结构预测器"""
    
    def __init__(self, config: dict):
        super(TransitionStatePredictor, self).__init__()
        
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # 分子编码器（用于反应物和产物）
        self.molecular_encoder = MolecularGNN(
            input_dim=11,  # 原子类型one-hot编码维度 (H,C,N,O,F,Si,P,S,Cl,Br,I)
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # 反应特征融合层
        self.reaction_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),  # 反应物+产物的全局特征
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # 坐标预测头
        self.coordinate_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3)  # 输出3D坐标
        )
        
        # 注意力机制用于原子级特征融合
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
    
    def encode_molecule(self, graph_data):
        """编码分子图"""
        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = graph_data.batch if hasattr(graph_data, 'batch') else None
        
        # 获取节点特征
        node_features = self.molecular_encoder(x, edge_index, batch)
        
        # 全局池化获取分子级特征
        if batch is not None:
            global_mean = global_mean_pool(node_features, batch)
            global_max = global_max_pool(node_features, batch)
        else:
            global_mean = torch.mean(node_features, dim=0, keepdim=True)
            global_max = torch.max(node_features, dim=0, keepdim=True)[0]
        
        global_features = torch.cat([global_mean, global_max], dim=-1)
        
        return node_features, global_features
    
    def forward(self, reactant_batch, product_batch):
        """
        前向传播
        Args:
            reactant_batch: 反应物图数据批次
            product_batch: 产物图数据批次
        Returns:
            predicted_ts_coords: 预测的过渡态坐标
        """
        # 编码反应物和产物
        r_node_feat, r_global_feat = self.encode_molecule(reactant_batch)
        p_node_feat, p_global_feat = self.encode_molecule(product_batch)
        
        # 融合反应特征
        reaction_features = torch.cat([r_global_feat, p_global_feat], dim=-1)
        reaction_context = self.reaction_fusion(reaction_features)
        
        # 使用注意力机制融合原子级特征
        # 假设反应物和产物原子数相同且对应
        combined_node_feat = torch.stack([r_node_feat, p_node_feat], dim=1)  # [n_atoms, 2, hidden_dim]
        
        # 应用注意力
        attended_feat, _ = self.attention(
            combined_node_feat, combined_node_feat, combined_node_feat
        )
        attended_feat = attended_feat.mean(dim=1)  # [n_atoms, hidden_dim]
        
        # 为每个原子添加反应上下文
        n_atoms = attended_feat.size(0)
        # 确保reaction_context是2D张量 [batch_size, hidden_dim]
        if reaction_context.dim() == 1:
            reaction_context = reaction_context.unsqueeze(0)
        reaction_context_expanded = reaction_context.expand(n_atoms, -1)
        
        # 预测坐标
        atom_context = torch.cat([attended_feat, reaction_context_expanded], dim=-1)
        predicted_coords = self.coordinate_predictor(atom_context)
        
        return predicted_coords


class TSPredictionLoss(nn.Module):
    """过渡态预测损失函数"""
    
    def __init__(self, coord_weight: float = 1.0, smooth_weight: float = 0.1):
        super(TSPredictionLoss, self).__init__()
        self.coord_weight = coord_weight
        self.smooth_weight = smooth_weight
    
    def coordinate_loss(self, pred_coords, true_coords):
        """坐标预测损失（RMSE）"""
        return F.mse_loss(pred_coords, true_coords)
    
    def smoothness_loss(self, pred_coords, reactant_coords, product_coords):
        """路径平滑性损失"""
        # 过渡态应该在反应物和产物之间
        mid_coords = (reactant_coords + product_coords) / 2
        return F.mse_loss(pred_coords, mid_coords)
    
    def forward(self, pred_coords, true_coords, reactant_coords, product_coords):
        """计算总损失"""
        coord_loss = self.coordinate_loss(pred_coords, true_coords)
        smooth_loss = self.smoothness_loss(pred_coords, reactant_coords, product_coords)
        
        total_loss = (self.coord_weight * coord_loss + 
                     self.smooth_weight * smooth_loss)
        
        return total_loss, {
            'coord_loss': coord_loss.item(),
            'smooth_loss': smooth_loss.item(),
            'total_loss': total_loss.item()
        }


def test_model():
    """测试模型"""
    config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.1
    }
    
    model = TransitionStatePredictor(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    batch_size = 2
    n_atoms = 10
    
    # 模拟图数据
    x = torch.randn(n_atoms * batch_size, 8)  # 节点特征
    edge_index = torch.randint(0, n_atoms * batch_size, (2, 20))  # 边索引
    batch = torch.repeat_interleave(torch.arange(batch_size), n_atoms)
    
    reactant_data = Data(x=x, edge_index=edge_index, batch=batch)
    product_data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # 前向传播
    with torch.no_grad():
        output = model(reactant_data, product_data)
        print(f"Output shape: {output.shape}")
    
    print("Model test passed!")


if __name__ == "__main__":
    test_model()