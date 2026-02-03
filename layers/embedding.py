import torch
import torch.nn as nn
import numpy as np
from Anla.core.base_layer import ComplexLayer

class ComplexEmbedding(ComplexLayer):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim, dtype=torch.complex64))
        self.reset_parameters()
        
    def reset_parameters(self):
        with torch.no_grad():
            phase = torch.empty_like(self.weight.real).uniform_(-np.pi, np.pi)
            scale = np.sqrt(1 / self.embedding_dim)
            magnitude = torch.sqrt(torch.empty_like(self.weight.real).exponential_(1.0)) * scale
            self.weight.data = torch.polar(magnitude, phase)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.input_cache = input_ids.detach().clone()
        return self.weight[input_ids]

    def manual_backward(self, grad_output: torch.Tensor, learning_rate: float, weight_decay: float = 1e-4) -> torch.Tensor:
        indices = self.input_cache
        grad_flat = grad_output.view(-1, self.embedding_dim)
        indices_flat = indices.view(-1)
        
        if indices_flat.device != self.weight.device:
            indices_flat = indices_flat.to(self.weight.device)
            
        with torch.no_grad():
            # 1. 稀疏权重衰减
            active_weights = self.weight.index_select(0, indices_flat)
            decay_delta = - active_weights * weight_decay
            
            # 2. [新增] 稀疏梯度裁剪
            # 计算每个样本产生的梯度模长
            grad_norm = torch.abs(grad_flat)
            # 限制模长
            clip_scale = torch.clamp(5.0 / (grad_norm + 1e-6), max=1.0)
            clipped_grad = grad_flat * clip_scale
            
            # 3. 应用更新
            # 实部虚部分开 add
            total_delta = clipped_grad + decay_delta
            
            self.weight.real.index_add_(0, indices_flat, learning_rate * total_delta.real)
            self.weight.imag.index_add_(0, indices_flat, learning_rate * total_delta.imag)
            
        self.clear_cache()
        return None
