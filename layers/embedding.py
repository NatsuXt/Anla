import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer
from Anla.utils.complex_ops import complex_kaiming_normal_

class ComplexEmbedding(ComplexLayer):
    """
    [Anla AGI Core] Manifold Embedding Layer
    支持：
    1. 融合自适应优化器 (Fused Adaptive Optimizer)
    2. 全局能量稳态 (Global Energy Homeostasis)
    3. 双向纠缠接口 (Bidirectional Entanglement Interface)

    v2 变更:
        全局能量稳态: 原版基于 batch 活跃子集的 RMS → 基于全量 vocab 的 RMS
        原版问题: 不同 batch 采样到的子集不同, RMS 估计每步不同,
                  引入依赖采样顺序的随机模长抖动。且只有活跃 token 被缩放,
                  冷门 token 保持初始化模长, 产生模长双峰分布。
        修正: 基于全量 embedding 的 RMS 计算缩放因子, 使缩放基准稳定;
              缩放仍只施加于活跃 token (保持原版行为, 不改变非活跃 token)。
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 1. 复数流形初始化
        weight_real = torch.empty(num_embeddings, embedding_dim)
        weight_imag = torch.empty(num_embeddings, embedding_dim)
        complex_kaiming_normal_(weight_real, weight_imag)
        
        # 初始化归一化：保证起点在单位球体附近
        raw_w = torch.complex(weight_real, weight_imag)
        init_w = raw_w / (raw_w.abs() + 1e-9)
        self.weight = nn.Parameter(init_w)
        
        # 2. 能量缓冲
        self.register_buffer('weight_energy', torch.full((num_embeddings, embedding_dim), 1e-3, dtype=torch.float32))
        
        self.input_cache = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.input_cache = input_ids if self.training else None
        with torch.no_grad():
            return nn.functional.embedding(input_ids, self.weight)

    def manual_backward(self, grad_output: torch.Tensor, lr: float, weight_decay: float = 0.0):
        """Path A: 针对 forward 输入的默认反向传播"""
        if self.input_cache is None:
            return
        self._apply_update(grad_output, self.input_cache, lr, weight_decay)
        self.input_cache = None

    def manual_backward_explicit(self, grad: torch.Tensor, indices: torch.Tensor, lr: float, weight_decay: float = 0.0):
        """Path B: [双向纠缠接口]"""
        self._apply_update(grad, indices, lr, weight_decay)

    def _apply_update(self, grad: torch.Tensor, indices: torch.Tensor, lr: float, weight_decay: float):
        """
        核心动力学更新逻辑 — 与原版完全一致, 仅修改稳态 RMS 的计算基准。
        """
        # 1. 维度展平
        grad_flat = grad.reshape(-1, self.embedding_dim)
        ids_flat = indices.reshape(-1)
        
        # 2. 梯度聚合 (与原版一致: grad_sum)
        unique_ids, inverse_indices = torch.unique(ids_flat, return_inverse=True)
        
        grad_sum = torch.zeros(len(unique_ids), self.embedding_dim, 
                             dtype=grad_flat.dtype, device=grad_flat.device)
        grad_sum.index_add_(0, inverse_indices, grad_flat)
        
        with torch.no_grad():
            current_weights = self.weight.data[unique_ids]
            current_energy = self.weight_energy[unique_ids]
            
            # --- Step 1: 融合优化器 (与原版完全一致) ---
            curr_grad_sq = grad_sum.abs().pow(2)
            
            beta = 0.90
            eps = 1e-5
            
            current_energy.mul_(beta).add_(curr_grad_sq, alpha=1-beta)
            denom = current_energy.sqrt().add_(eps)
            adaptive_step = grad_sum / denom * lr
            
            if weight_decay > 0:
                current_weights.mul_(1.0 - weight_decay)
            
            current_weights.sub_(adaptive_step)
            
            # --- Step 2: 全局能量稳态 ---
            # [v2] 使用全量 vocab 的 RMS 替代 batch 活跃子集的 RMS
            # 这样缩放因子不再依赖当前 batch 的随机采样
            all_moduli_sq = self.weight.data.abs().pow(2)
            full_rms = torch.sqrt(all_moduli_sq.mean() + 1e-9)
            
            target_energy = 1.0
            scaling_factor = target_energy / (full_rms + 1e-9)
            
            # 缩放仍只施加于活跃 token (与原版行为一致)
            current_weights.mul_(scaling_factor)
            
            # --- Step 3: 状态回写 ---
            self.weight.data.index_put_((unique_ids,), current_weights)
            self.weight_energy.index_put_((unique_ids,), current_energy)
