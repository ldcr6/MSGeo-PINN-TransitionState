#!/usr/bin/env python3
"""
终极优化模型
整合所有先进技术：增强特征、生成模型、物理约束、知识蒸馏等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

from enhanced_features import EnhancedFeatureExtractor
from generative_models import TransitionStateVAE, TransitionStateDiffusion
from data_augmentation import AdaptiveAugmentation


class PhysicsInformedLayer(nn.Module):
    """物理启发层"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 能量预测分支
        self.energy_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 力预测分支
        self.force_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # x, y, z 方向的力
        )
        
        # 物理约束权重
        self.constraint_weights = nn.Parameter(torch.ones(3))
        
    def energy_conservation_loss(self, predicted_energy: torch.Tensor, 
                                reactant_energy: torch.Tensor, 
                                product_energy: torch.Tensor) -> torch.Tensor:
        """能量守恒约束"""
        # 过渡态能量应该高于反应物和产物
        energy_barrier_r = F.relu(reactant_energy - predicted_energy + 0.5)  # 至少0.5 eV的能垒
        energy_barrier_p = F.relu(product_energy - predicted_energy + 0.5)
        
        return energy_barrier_r + energy_barrier_p
    
    def force_consistency_loss(self, predicted_forces: torch.Tensor, 
                              coords: torch.Tensor) -> torch.Tensor:
        """力的一致性约束"""
        # 力的总和应该接近零（动量守恒）
        total_force = predicted_forces.sum(dim=0)
        momentum_conservation = torch.norm(total_force)
        
        # 力的方向应该合理（指向能量降低的方向）
        force_magnitude = torch.norm(predicted_forces, dim=1)
        force_smoothness = torch.var(force_magnitude)
        
        return momentum_conservation + 0.1 * force_smoothness
    
    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 预测能量和力
        energy = self.energy_predictor(features.mean(dim=0))  # 分子级别的能量
        forces = self.force_predictor(features)  # 原子级别的力
        
        return {
            'energy': energy,
            'forces': forces,
            'constraint_weights': self.constraint_weights
        }


class AttentionFusion(nn.Module):
    """注意力融合模块"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 交叉注意力（反应物-产物）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(100, hidden_dim))
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, reactant_features: torch.Tensor, 
                product_features: torch.Tensor) -> torch.Tensor:
        """融合反应物和产物特征"""
        batch_size, n_atoms, hidden_dim = reactant_features.shape
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:n_atoms].unsqueeze(0).expand(batch_size, -1, -1)
        reactant_features = reactant_features + pos_enc
        product_features = product_features + pos_enc
        
        # 自注意力
        r_attn, _ = self.multihead_attn(reactant_features, reactant_features, reactant_features)
        p_attn, _ = self.multihead_attn(product_features, product_features, product_features)
        
        # 残差连接和层归一化
        reactant_features = self.layer_norm1(reactant_features + r_attn)
        product_features = self.layer_norm1(product_features + p_attn)
        
        # 交叉注意力
        rp_cross, _ = self.cross_attn(reactant_features, product_features, product_features)
        pr_cross, _ = self.cross_attn(product_features, reactant_features, reactant_features)
        
        # 融合特征
        fused_features = (reactant_features + rp_cross + product_features + pr_cross) / 2
        
        # 前馈网络
        ffn_output = self.ffn(fused_features)
        fused_features = self.layer_norm2(fused_features + ffn_output)
        
        return fused_features


class UncertaintyQuantification(nn.Module):
    """不确定性量化模块"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # 认知不确定性（模型不确定性）
        self.epistemic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # x, y, z 方向的不确定性
            nn.Softplus()  # 确保正值
        )
        
        # 偶然不确定性（数据不确定性）
        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Softplus()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测不确定性"""
        epistemic_uncertainty = self.epistemic_head(features)
        aleatoric_uncertainty = self.aleatoric_head(features)
        
        # 总不确定性
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'epistemic': epistemic_uncertainty,
            'aleatoric': aleatoric_uncertainty,
            'total': total_uncertainty
        }


class KnowledgeDistillation(nn.Module):
    """知识蒸馏模块"""
    
    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        
        # 特征对齐网络
        self.feature_adapter = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.ReLU(),
            nn.Linear(teacher_dim, teacher_dim)
        )
        
        # 注意力转移
        self.attention_transfer = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim // 2),
            nn.ReLU(),
            nn.Linear(teacher_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.temperature = 4.0  # 蒸馏温度
        
    def distillation_loss(self, student_logits: torch.Tensor, 
                         teacher_logits: torch.Tensor, 
                         target: torch.Tensor, 
                         alpha: float = 0.7) -> torch.Tensor:
        """知识蒸馏损失"""
        # 软目标损失
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 硬目标损失
        hard_loss = F.mse_loss(student_logits, target)
        
        return alpha * soft_loss + (1 - alpha) * hard_loss
    
    def forward(self, student_features: torch.Tensor, 
                teacher_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """特征蒸馏"""
        # 对齐学生特征到教师维度
        aligned_features = self.feature_adapter(student_features)
        
        # 注意力转移
        attention_weights = self.attention_transfer(teacher_features)
        
        # 特征匹配损失
        feature_loss = F.mse_loss(aligned_features, teacher_features)
        
        # 注意力匹配损失
        student_attention = self.attention_transfer(aligned_features)
        attention_loss = F.mse_loss(student_attention, attention_weights)
        
        return {
            'feature_loss': feature_loss,
            'attention_loss': attention_loss,
            'aligned_features': aligned_features
        }


class UltimateTransitionStatePredictor(nn.Module):
    """终极过渡态预测模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 512)
        self.max_atoms = config.get('max_atoms', 50)
        
        # 增强特征提取器
        self.feature_extractor = EnhancedFeatureExtractor(config)
        
        # 图神经网络层
        self.gnn_layers = nn.ModuleList([
            GATConv(self.hidden_dim, self.hidden_dim, heads=8, concat=False, dropout=0.1),
            GCNConv(self.hidden_dim, self.hidden_dim),
            GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False, dropout=0.1),
        ])
        
        # 注意力融合
        self.attention_fusion = AttentionFusion(self.hidden_dim, num_heads=8)
        
        # 物理启发层
        self.physics_layer = PhysicsInformedLayer(self.hidden_dim)
        
        # 不确定性量化
        self.uncertainty_module = UncertaintyQuantification(self.hidden_dim)
        
        # 生成模型
        if config.get('use_vae', True):
            self.vae = TransitionStateVAE(config)
        
        if config.get('use_diffusion', True):
            self.diffusion = TransitionStateDiffusion(config)
        
        # 坐标预测头
        self.coord_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3)  # x, y, z 坐标
        )
        
        # 多尺度预测
        self.multiscale_predictors = nn.ModuleList([
            nn.Linear(self.hidden_dim, 3) for _ in range(3)  # 不同尺度的预测器
        ])
        
        # 自适应权重
        self.adaptive_weights = nn.Parameter(torch.ones(5))  # 5个损失项的权重
        
    def create_molecular_graph(self, coords: torch.Tensor, atom_types: torch.Tensor, 
                              cutoff: float = 5.0) -> Data:
        """创建分子图"""
        n_atoms = coords.shape[0]
        
        # 计算距离矩阵
        distances = torch.cdist(coords, coords)
        
        # 创建边（距离小于cutoff的原子对）
        edge_indices = torch.where((distances < cutoff) & (distances > 0))
        edge_index = torch.stack(edge_indices, dim=0)
        
        # 边特征（距离）
        edge_attr = distances[edge_indices].unsqueeze(-1)
        
        # 节点特征（原子类型 + 坐标）
        node_features = torch.cat([
            F.one_hot(atom_types.long(), num_classes=20).float(),  # 原子类型编码
            coords  # 坐标
        ], dim=1)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=coords)
    
    def graph_neural_network(self, graph: Data) -> torch.Tensor:
        """图神经网络处理"""
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        
        # 初始特征投影
        x = nn.Linear(x.shape[1], self.hidden_dim).to(x.device)(x)
        
        # GNN层
        for i, gnn_layer in enumerate(self.gnn_layers):
            if isinstance(gnn_layer, GATConv):
                x = gnn_layer(x, edge_index)
            else:  # GCNConv
                x = gnn_layer(x, edge_index)
            
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        return x
    
    def forward(self, reactant_coords: torch.Tensor, product_coords: torch.Tensor,
                reactant_types: torch.Tensor, product_types: torch.Tensor,
                mode: str = 'prediction') -> Dict[str, torch.Tensor]:
        """前向传播"""
        
        # 1. 增强特征提取
        r_enhanced = self.feature_extractor(reactant_coords, reactant_types)
        p_enhanced = self.feature_extractor(product_coords, product_types)
        
        # 2. 图神经网络处理
        r_graph = self.create_molecular_graph(reactant_coords, reactant_types)
        p_graph = self.create_molecular_graph(product_coords, product_types)
        
        r_graph_features = self.graph_neural_network(r_graph)
        p_graph_features = self.graph_neural_network(p_graph)
        
        # 3. 注意力融合
        fused_features = self.attention_fusion(
            r_graph_features.unsqueeze(0), 
            p_graph_features.unsqueeze(0)
        ).squeeze(0)
        
        # 4. 物理约束
        physics_output = self.physics_layer(fused_features, reactant_coords)
        
        # 5. 不确定性量化
        uncertainty_output = self.uncertainty_module(fused_features)
        
        # 6. 坐标预测
        predicted_coords = self.coord_predictor(fused_features)
        
        # 7. 多尺度预测
        multiscale_predictions = []
        for predictor in self.multiscale_predictors:
            scale_pred = predictor(fused_features)
            multiscale_predictions.append(scale_pred)
        
        # 8. 生成模型（如果启用）
        generative_outputs = {}
        if hasattr(self, 'vae') and mode == 'training':
            vae_output = self.vae(predicted_coords.unsqueeze(0), 
                                 reactant_coords.unsqueeze(0), 
                                 product_coords.unsqueeze(0))
            generative_outputs['vae'] = vae_output
        
        if hasattr(self, 'diffusion') and mode == 'training':
            diffusion_output = self.diffusion(predicted_coords.unsqueeze(0),
                                            reactant_coords.unsqueeze(0),
                                            product_coords.unsqueeze(0))
            generative_outputs['diffusion'] = diffusion_output
        
        return {
            'predicted_coords': predicted_coords,
            'multiscale_predictions': multiscale_predictions,
            'physics_output': physics_output,
            'uncertainty': uncertainty_output,
            'enhanced_features': {
                'reactant': r_enhanced,
                'product': p_enhanced
            },
            'fused_features': fused_features,
            'generative_outputs': generative_outputs
        }
    
    def generate_transition_state(self, reactant_coords: torch.Tensor, 
                                product_coords: torch.Tensor,
                                reactant_types: torch.Tensor, 
                                product_types: torch.Tensor,
                                method: str = 'deterministic') -> torch.Tensor:
        """生成过渡态坐标"""
        
        if method == 'deterministic':
            # 确定性预测
            output = self.forward(reactant_coords, product_coords, 
                                reactant_types, product_types, mode='inference')
            return output['predicted_coords']
        
        elif method == 'vae' and hasattr(self, 'vae'):
            # VAE生成
            generated = self.vae.generate(
                reactant_coords.unsqueeze(0), 
                product_coords.unsqueeze(0),
                num_samples=1
            )
            return generated.squeeze(0).squeeze(0)
        
        elif method == 'diffusion' and hasattr(self, 'diffusion'):
            # 扩散模型生成
            generated = self.diffusion.sample(
                reactant_coords.unsqueeze(0),
                product_coords.unsqueeze(0)
            )
            return generated.squeeze(0)
        
        else:
            # 默认确定性预测
            output = self.forward(reactant_coords, product_coords, 
                                reactant_types, product_types, mode='inference')
            return output['predicted_coords']


class UltimateLoss(nn.Module):
    """终极损失函数"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.coord_weight = config.get('coord_weight', 1.0)
        self.physics_weight = config.get('physics_weight', 0.2)
        self.uncertainty_weight = config.get('uncertainty_weight', 0.1)
        self.multiscale_weight = config.get('multiscale_weight', 0.3)
        self.generative_weight = config.get('generative_weight', 0.5)
        
    def coordinate_loss(self, predicted: torch.Tensor, target: torch.Tensor,
                       uncertainty: torch.Tensor) -> torch.Tensor:
        """不确定性加权的坐标损失"""
        # 基础MSE损失
        mse_loss = F.mse_loss(predicted, target, reduction='none')
        
        # 不确定性加权
        weighted_loss = mse_loss / (uncertainty['total'] + 1e-8) + torch.log(uncertainty['total'] + 1e-8)
        
        return weighted_loss.mean()
    
    def physics_loss(self, physics_output: Dict, predicted_coords: torch.Tensor,
                    reactant_coords: torch.Tensor, product_coords: torch.Tensor) -> torch.Tensor:
        """物理约束损失"""
        # 能量守恒
        energy_loss = physics_output.get('energy_conservation_loss', 0)
        
        # 力的一致性
        force_loss = physics_output.get('force_consistency_loss', 0)
        
        # 几何约束
        geometry_loss = self.geometry_constraint_loss(predicted_coords, reactant_coords, product_coords)
        
        return energy_loss + force_loss + geometry_loss
    
    def geometry_constraint_loss(self, predicted: torch.Tensor, 
                               reactant: torch.Tensor, product: torch.Tensor) -> torch.Tensor:
        """几何约束损失"""
        # 过渡态应该在反应物和产物之间
        r_distances = torch.norm(predicted - reactant, dim=1)
        p_distances = torch.norm(predicted - product, dim=1)
        
        # 鼓励过渡态在反应路径上
        path_loss = F.relu(r_distances + p_distances - torch.norm(reactant - product, dim=1))
        
        return path_loss.mean()
    
    def multiscale_loss(self, multiscale_predictions: List[torch.Tensor], 
                       target: torch.Tensor) -> torch.Tensor:
        """多尺度预测损失"""
        total_loss = 0
        for pred in multiscale_predictions:
            total_loss += F.mse_loss(pred, target)
        
        return total_loss / len(multiscale_predictions)
    
    def forward(self, model_output: Dict, target_coords: torch.Tensor,
                reactant_coords: torch.Tensor, product_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算总损失"""
        
        losses = {}
        
        # 1. 坐标损失
        coord_loss = self.coordinate_loss(
            model_output['predicted_coords'], 
            target_coords,
            model_output['uncertainty']
        )
        losses['coordinate'] = coord_loss
        
        # 2. 物理损失
        physics_loss = self.physics_loss(
            model_output['physics_output'],
            model_output['predicted_coords'],
            reactant_coords,
            product_coords
        )
        losses['physics'] = physics_loss
        
        # 3. 多尺度损失
        multiscale_loss = self.multiscale_loss(
            model_output['multiscale_predictions'],
            target_coords
        )
        losses['multiscale'] = multiscale_loss
        
        # 4. 不确定性正则化
        uncertainty_reg = model_output['uncertainty']['total'].mean()
        losses['uncertainty'] = uncertainty_reg
        
        # 5. 生成模型损失
        generative_loss = 0
        if 'generative_outputs' in model_output:
            for gen_type, gen_output in model_output['generative_outputs'].items():
                if gen_type == 'vae':
                    vae_losses = self.vae_loss(gen_output, target_coords)
                    generative_loss += vae_losses['total_loss']
                elif gen_type == 'diffusion':
                    diff_loss = F.mse_loss(gen_output['predicted_noise'], gen_output['target_noise'])
                    generative_loss += diff_loss
        
        losses['generative'] = generative_loss
        
        # 总损失
        total_loss = (
            self.coord_weight * coord_loss +
            self.physics_weight * physics_loss +
            self.multiscale_weight * multiscale_loss +
            self.uncertainty_weight * uncertainty_reg +
            self.generative_weight * generative_loss
        )
        
        losses['total'] = total_loss
        
        return losses
    
    def vae_loss(self, vae_output: Dict, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """VAE损失"""
        reconstructed = vae_output['reconstructed']
        mu = vae_output['mu']
        logvar = vae_output['logvar']
        
        recon_loss = F.mse_loss(reconstructed, target.unsqueeze(0))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        
        return {
            'total_loss': recon_loss + 0.001 * kl_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }