import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer

class ComplexRMSNorm(ComplexLayer):
    """
    复数 RMS Normalization (Root Mean Square Layer Normalization)
    [修正版]: 支持 3D 序列输入的反向传播
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        # 可学习的缩放参数 (实数)
        self.scale = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.input_cache = z.detach().clone()
        
        # 1. 计算 RMS
        norm_z = torch.abs(z)
        rms = torch.sqrt(torch.mean(norm_z**2, dim=-1, keepdim=True) + self.eps)
        
        # 2. 归一化
        z_normalized = z / rms
        
        # 3. 缩放
        output = z_normalized * self.scale
        
        self.output_cache = (z_normalized, rms)
        return output

    def manual_backward(self, grad_output: torch.Tensor, learning_rate: float, **kwargs) -> torch.Tensor:
        """
        [修正]: 正确处理任意维度输入的参数更新
        """
        z = self.input_cache
        z_normalized, rms = self.output_cache
        
        # --- 1. Scale 的梯度 ---
        # grad_scale_per_element = Re(grad_output * conj(z_normalized))
        # 形状: (Batch, Seq, Dim)
        grad_scale_per_element = torch.real(grad_output * torch.conj(z_normalized))
        
        # [Fix]: 展平除最后一维外的所有维度: (Total_Samples, Dim)
        # 这样无论是 2D (Batch, Dim) 还是 3D (Batch, Seq, Dim) 都能统一处理
        grad_scale_flat = grad_scale_per_element.view(-1, self.normalized_shape)
        
        total_samples = grad_scale_flat.shape[0]
        grad_scale = torch.sum(grad_scale_flat, dim=0) / total_samples
        
        # 更新 scale
        with torch.no_grad():
            self.scale.data += learning_rate * grad_scale
            
        # --- 2. 输入 z 的梯度 ---
        # 简化版高效实现 (参考 PyTorch RMSNorm 梯度结构)
        # dx = g * (dy - z_norm * real(dy * conj(z_norm)) / dim) / rms
        
        grad_y = grad_output * self.scale
        
        dot_prod = torch.real(torch.sum(grad_y * torch.conj(z_normalized), dim=-1, keepdim=True))
        
        dim = self.normalized_shape
        numerator = grad_y - z_normalized * (dot_prod / dim)
        grad_input = numerator / rms
        
        self.clear_cache()
        return grad_input
