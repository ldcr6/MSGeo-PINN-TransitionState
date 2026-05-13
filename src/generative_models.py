#!/usr/bin/env python3
"""
生成模型模块
包含VAE、扩散模型等用于连续坐标生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class VariationalEncoder(nn.Module):
    """VAE编码器"""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 均值和方差网络
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码输入到潜在空间"""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class VariationalDecoder(nn.Module):
    """VAE解码器"""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码到坐标"""
        return self.decoder(z)


class TransitionStateVAE(nn.Module):
    """过渡态坐标生成的VAE模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.max_atoms = config.get('max_atoms', 50)
        self.coord_dim = self.max_atoms * 3  # x, y, z坐标
        self.latent_dim = config.get('latent_dim', 128)
        self.hidden_dims = config.get('hidden_dims', [512, 256, 128])
        
        # 编码器和解码器
        self.encoder = VariationalEncoder(
            input_dim=self.coord_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        )
        
        self.decoder = VariationalDecoder(
            latent_dim=self.latent_dim,
            output_dim=self.coord_dim,
            hidden_dims=self.hidden_dims
        )
        
        # 条件编码器（用于反应物和产物信息）
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.coord_dim * 2, 256),  # 反应物 + 产物
            nn.ReLU(),
            nn.Linear(256, self.latent_dim)
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, ts_coords: torch.Tensor, r_coords: torch.Tensor, 
                p_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size = ts_coords.shape[0]
        
        # 展平坐标
        ts_flat = ts_coords.view(batch_size, -1)
        r_flat = r_coords.view(batch_size, -1)
        p_flat = p_coords.view(batch_size, -1)
        
        # 编码过渡态
        mu, logvar = self.encoder(ts_flat)
        z = self.reparameterize(mu, logvar)
        
        # 条件信息
        condition = torch.cat([r_flat, p_flat], dim=1)
        condition_z = self.condition_encoder(condition)
        
        # 结合条件信息
        combined_z = z + condition_z
        
        # 解码
        reconstructed = self.decoder(combined_z)
        reconstructed_coords = reconstructed.view(batch_size, self.max_atoms, 3)
        
        return {
            'reconstructed': reconstructed_coords,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'condition_z': condition_z
        }
    
    def generate(self, r_coords: torch.Tensor, p_coords: torch.Tensor, 
                 num_samples: int = 1) -> torch.Tensor:
        """生成过渡态坐标"""
        batch_size = r_coords.shape[0]
        
        # 条件信息
        r_flat = r_coords.view(batch_size, -1)
        p_flat = p_coords.view(batch_size, -1)
        condition = torch.cat([r_flat, p_flat], dim=1)
        condition_z = self.condition_encoder(condition)
        
        generated_samples = []
        for _ in range(num_samples):
            # 从先验分布采样
            z = torch.randn(batch_size, self.latent_dim, device=r_coords.device)
            
            # 结合条件信息
            combined_z = z + condition_z
            
            # 解码
            generated = self.decoder(combined_z)
            generated_coords = generated.view(batch_size, self.max_atoms, 3)
            generated_samples.append(generated_coords)
        
        return torch.stack(generated_samples, dim=1)  # [batch, num_samples, atoms, 3]


class DiffusionScheduler:
    """扩散模型调度器"""
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, 
                 beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
        # 线性调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算扩散过程需要的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 逆过程系数
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """添加噪声到原始数据"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class DiffusionUNet(nn.Module):
    """扩散模型的U-Net架构"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.max_atoms = config.get('max_atoms', 50)
        self.coord_dim = 3
        self.hidden_dim = config.get('hidden_dim', 256)
        self.time_embed_dim = config.get('time_embed_dim', 128)
        
        # 时间嵌入
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.ReLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # 条件嵌入（反应物和产物）
        self.condition_embedding = nn.Sequential(
            nn.Linear(self.max_atoms * self.coord_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # U-Net编码器
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.coord_dim + self.time_embed_dim + self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ),
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
            ),
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4)
            )
        ])
        
        # 中间层
        self.middle_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4)
        )
        
        # U-Net解码器
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 8, self.hidden_dim * 2),  # 跳跃连接
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
            ),
            nn.Sequential(
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ),
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.coord_dim)
            )
        ])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, r_coords: torch.Tensor, 
                p_coords: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, num_atoms, _ = x.shape
        
        # 时间嵌入
        t_normalized = t.float() / 1000.0  # 归一化时间步
        time_embed = self.time_embedding(t_normalized.unsqueeze(-1))  # [batch, time_embed_dim]
        time_embed = time_embed.unsqueeze(1).expand(-1, num_atoms, -1)  # [batch, atoms, time_embed_dim]
        
        # 条件嵌入
        condition_input = torch.cat([
            r_coords.view(batch_size, -1),
            p_coords.view(batch_size, -1)
        ], dim=1)
        condition_embed = self.condition_embedding(condition_input)  # [batch, hidden_dim]
        condition_embed = condition_embed.unsqueeze(1).expand(-1, num_atoms, -1)  # [batch, atoms, hidden_dim]
        
        # 合并输入
        h = torch.cat([x, time_embed, condition_embed], dim=-1)  # [batch, atoms, coord_dim + time_embed_dim + hidden_dim]
        
        # 编码器（保存跳跃连接）
        skip_connections = []
        for encoder in self.encoder_layers:
            h = encoder(h)
            skip_connections.append(h)
        
        # 中间层
        h = self.middle_layer(h)
        
        # 解码器（使用跳跃连接）
        for i, decoder in enumerate(self.decoder_layers):
            if i < len(skip_connections):
                h = torch.cat([h, skip_connections[-(i+1)]], dim=-1)
            h = decoder(h)
        
        return h


class TransitionStateDiffusion(nn.Module):
    """过渡态坐标生成的扩散模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.scheduler = DiffusionScheduler(
            num_timesteps=config.get('num_timesteps', 1000),
            beta_start=config.get('beta_start', 1e-4),
            beta_end=config.get('beta_end', 0.02)
        )
        
        self.unet = DiffusionUNet(config)
        
    def forward(self, ts_coords: torch.Tensor, r_coords: torch.Tensor, 
                p_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """训练时的前向传播"""
        batch_size = ts_coords.shape[0]
        device = ts_coords.device
        
        # 随机采样时间步
        t = self.scheduler.sample_timesteps(batch_size, device)
        
        # 生成噪声
        noise = torch.randn_like(ts_coords)
        
        # 添加噪声
        noisy_coords = self.scheduler.add_noise(ts_coords, t, noise)
        
        # 预测噪声
        predicted_noise = self.unet(noisy_coords, t, r_coords, p_coords)
        
        return {
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            'noisy_coords': noisy_coords,
            'timesteps': t
        }
    
    @torch.no_grad()
    def sample(self, r_coords: torch.Tensor, p_coords: torch.Tensor, 
               num_inference_steps: int = 50) -> torch.Tensor:
        """生成过渡态坐标"""
        batch_size = r_coords.shape[0]
        device = r_coords.device
        
        # 从纯噪声开始
        x = torch.randn(batch_size, self.unet.max_atoms, 3, device=device)
        
        # 逆扩散过程
        timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        for t in timesteps:
            t_batch = t.expand(batch_size)
            
            # 预测噪声
            predicted_noise = self.unet(x, t_batch, r_coords, p_coords)
            
            # 去噪
            alpha_t = self.scheduler.alphas[t]
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t]
            beta_t = self.scheduler.betas[t]
            
            # 计算去噪后的坐标
            x = (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # 添加噪声（除了最后一步）
            if t > 0:
                noise = torch.randn_like(x)
                variance = self.scheduler.posterior_variance[t]
                x = x + torch.sqrt(variance) * noise
        
        return x


class GenerativeModelLoss(nn.Module):
    """生成模型损失函数"""
    
    def __init__(self, vae_weight: float = 1.0, diffusion_weight: float = 1.0):
        super().__init__()
        self.vae_weight = vae_weight
        self.diffusion_weight = diffusion_weight
        
    def vae_loss(self, vae_output: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """VAE损失"""
        reconstructed = vae_output['reconstructed']
        mu = vae_output['mu']
        logvar = vae_output['logvar']
        
        # 重构损失
        recon_loss = F.mse_loss(reconstructed, target, reduction='mean')
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        
        total_loss = recon_loss + 0.001 * kl_loss  # KL权重
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def diffusion_loss(self, diffusion_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """扩散模型损失"""
        predicted_noise = diffusion_output['predicted_noise']
        target_noise = diffusion_output['target_noise']
        
        return F.mse_loss(predicted_noise, target_noise, reduction='mean')
    
    def forward(self, vae_output: Optional[Dict] = None, diffusion_output: Optional[Dict] = None,
                target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """总损失"""
        losses = {}
        total_loss = 0
        
        if vae_output is not None and target is not None:
            vae_losses = self.vae_loss(vae_output, target)
            losses.update({f'vae_{k}': v for k, v in vae_losses.items()})
            total_loss += self.vae_weight * vae_losses['total_loss']
        
        if diffusion_output is not None:
            diff_loss = self.diffusion_loss(diffusion_output)
            losses['diffusion_loss'] = diff_loss
            total_loss += self.diffusion_weight * diff_loss
        
        losses['total_loss'] = total_loss
        return losses