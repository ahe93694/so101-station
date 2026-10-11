"""
Loupe-Diffusion: Sparse Attention for Action Dimensions in Diffusion Policy

Adapted from: "The Loupe: A Plug-and-Play Attention Module for 
               Amplifying Discriminative Features in Vision Transformers"
               arXiv:2508.16663

Core idea: Generate sparse attention weights over action dimensions,
           guiding the denoising process to focus on the most important
           action dimensions at each timestep.

Original Loupe: 2D spatial attention (H x W) for images
Loupe-Diffusion: 1D action-dimension attention (D_action) for robot actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class LoupeActionAttention(nn.Module):
    """
    Loupe注意力模块适配版：用于Diffusion Policy的动作维度注意力
    
    核心思想：
    - 在去噪网络的中间层生成动作维度的稀疏注意力权重
    - L1稀疏性确保只有关键动作维度被关注
    - 引导去噪过程聚焦于最重要的动作维度
    
    Architecture:
        features -> Linear(D->H) -> ReLU -> LayerNorm -> Linear(H->D_action) -> Sigmoid
        
    Loss:
        L_total = L_diffusion + λ * ||attention_map||_1
    """
    
    def __init__(
        self,
        feature_dim: int,        # 输入特征维度
        action_dim: int,         # 动作维度（输出注意力维度）
        hidden_dim: int = 128,   # 中间层维度
        sparsity_lambda: float = 0.05,  # L1稀疏性权重
    ):
        super().__init__()
        self.action_dim = action_dim
        self.sparsity_lambda = sparsity_lambda
        
        # 两层网络：Linear -> ReLU -> Linear -> Sigmoid
        # 类似原Loupe的 Conv3x3 + ReLU + Conv1x1 + Sigmoid
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),  # 输出 [0,1] 范围的注意力权重
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, D] 或 [B, T, D] 的去噪网络中间特征
            
        Returns:
            attention_map: [B, action_dim] 或 [B, T, action_dim] 的注意力权重
            sparsity_loss: 标量，L1稀疏性损失
        """
        original_shape = features.shape
        
        # 处理不同输入维度
        if len(original_shape) == 3:
            B, T, D = original_shape
            features = features.reshape(B * T, D)
        elif len(original_shape) == 2:
            B, D = original_shape
            T = 1
        else:
            raise ValueError(f"Expected 2D or 3D input, got {len(original_shape)}D")
            
        # 生成注意力图
        attention_map = self.attention_net(features)  # [B*T, action_dim]
        
        # 计算L1稀疏性损失
        sparsity_loss = torch.mean(torch.abs(attention_map)) * self.sparsity_lambda
        
        # 恢复维度
        if len(original_shape) == 3:
            attention_map = attention_map.reshape(B, T, self.action_dim)
            
        return attention_map, sparsity_loss


class LoupeDiffusionWrapper(nn.Module):
    """
    Wrapper that integrates Loupe attention into Diffusion Policy.
    
    Usage:
        wrapper = LoupeDiffusionWrapper(feature_dim=256, action_dim=7)
        
        # During training
        pred_actions = model(noisy_actions, timesteps, cond)
        enhanced_actions, sparsity_loss = wrapper(pred_actions, features)
        total_loss = diffusion_loss + sparsity_loss
    """
    
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        sparsity_lambda: float = 0.05,
        num_layers: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        
        # Multiple Loupe modules for different layers
        self.loupe_modules = nn.ModuleList([
            LoupeActionAttention(
                feature_dim=feature_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                sparsity_lambda=sparsity_lambda,
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        actions: torch.Tensor,
        features: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Loupe attention to enhance actions.
        
        Args:
            actions: [B, T, D_action] predicted actions
            features: [B, T, D_feat] intermediate features from denoising network
            layer_idx: which Loupe layer to use
            
        Returns:
            enhanced_actions: [B, T, D_action] attention-weighted actions
            attention_map: [B, T, D_action] attention weights
            sparsity_loss: scalar L1 loss
        """
        loupe = self.loupe_modules[layer_idx % len(self.loupe_modules)]
        
        # Get attention weights
        attention_map, sparsity_loss = loupe(features)
        
        # Apply attention to actions (gate mechanism)
        enhanced_actions = actions * attention_map
        
        return enhanced_actions, attention_map, sparsity_loss


def create_loupe_config():
    """Returns configuration dict for Loupe integration."""
    return {
        'use_loupe_attention': False,
        'loupe_sparsity_lambda': 0.05,
        'loupe_hidden_dim': 128,
        'num_loupe_layers': 2,
    }


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Loupe-Diffusion Module Test")
    print("=" * 60)
    
    # Test parameters
    B, T = 4, 16
    feature_dim = 256
    action_dim = 7
    
    # Create module
    loupe = LoupeActionAttention(
        feature_dim=feature_dim,
        action_dim=action_dim,
        hidden_dim=128,
        sparsity_lambda=0.05,
    )
    
    # Test with 3D input
    features = torch.randn(B, T, feature_dim)
    attn_map, sp_loss = loupe(features)
    
    print(f"Input shape:  {features.shape}")
    print(f"Output shape: {attn_map.shape}")
    print(f"Sparsity loss: {sp_loss.item():.4f}")
    print(f"Attention range: [{attn_map.min():.3f}, {attn_map.max():.3f}]")
    print(f"Attention mean: {attn_map.mean():.3f}")
    
    # Test wrapper
    wrapper = LoupeDiffusionWrapper(
        feature_dim=feature_dim,
        action_dim=action_dim,
        num_layers=2,
    )
    
    actions = torch.randn(B, T, action_dim)
    enhanced, attn, loss = wrapper(actions, features)
    
    print(f"\nWrapper test:")
    print(f"Actions shape: {actions.shape}")
    print(f"Enhanced shape: {enhanced.shape}")
    print(f"Total sparsity loss: {loss.item():.4f}")
    
    # Test gradient flow
    loss.backward()
    print(f"\n✓ Gradient flow works!")
    print("=" * 60)
