import torch
import torch.nn as nn
import numpy as np
from Anla.core.base_layer import ComplexLayer

class ComplexLinear(ComplexLayer):
    """
    全复数线性层 (Fully Complex Linear Layer)
    支持 2D (Batch, Dim) 和 3D (Batch, Seq, Dim) 输入。
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重初始化
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.complex64))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        scale = np.sqrt(1 / (self.in_features + self.out_features))
        with torch.no_grad():
            nn.init.uniform_(self.weight.real, -scale, scale)
            nn.init.uniform_(self.weight.imag, -scale, scale)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)

    def forward(self, input_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_z: (Batch, ..., in_features)
        """
        self.input_cache = input_z.detach().clone()
        
        # PyTorch 的 linear 实现是 input @ weight.T
        # 它自动支持多维输入 (Batch, Seq, In) -> (Batch, Seq, Out)
        output = torch.matmul(input_z, self.weight.t())
        
        if self.bias is not None:
            output += self.bias
            
        return output

    def manual_backward(self, grad_output: torch.Tensor, learning_rate: float, weight_decay: float = 1e-4) -> torch.Tensor:
        """
        [修正版] 支持任意维度输入的反向传播 (2D or 3D)
        """
        if self.input_cache is None:
            raise RuntimeError("Forward pass must be called before manual_backward.")
            
        X = self.input_cache        # (Batch, ..., In)
        delta_Y = grad_output       # (Batch, ..., Out)
        
        # --- 1. Error Backpropagation (梯度回传) ---
        # Formula: grad_in = grad_out @ W*
        # PyTorch matmul 自动处理最后两维，所以 3D 输入也能直接乘
        # (..., Out) @ (Out, In) -> (..., In)
        grad_input = torch.matmul(delta_Y, self.weight.conj())
        
        # --- 2. Weight Gradient Calculation (dW) ---
        # Formula: dW = delta_Y^T @ X*
        # 但我们需要处理 Batch 和 Sequence 维度。
        # 最简单的方法：将前导维度展平 (Flatten batch & seq)
        
        # 将 X 和 delta_Y 展平为 2D: (N_samples, Features)
        X_flat = X.reshape(-1, self.in_features)
        delta_flat = delta_Y.reshape(-1, self.out_features)
        
        # 计算有效样本数 (Batch * Seq)
        total_samples = X_flat.shape[0]
        
        # 现在可以使用 .t() 了
        # (Out, N) @ (N, In) -> (Out, In)
        dW = torch.matmul(delta_flat.t(), X_flat.conj()) / total_samples
        
        # Bias Gradient
        if self.bias is not None:
            # 对所有样本维度求和
            dB = torch.sum(delta_flat, dim=0) / total_samples
        else:
            dB = None
            
        # --- 3. Update Parameters ---
        # 使用基类的 _apply_gradient (如果基类已更新)
        # 如果尚未更新基类，这里直接写更新逻辑
        if hasattr(self, '_apply_gradient'):
            self._apply_gradient(self.weight, dW, learning_rate, decay=weight_decay, clip_norm=5.0)
            if self.bias is not None:
                self._apply_gradient(self.bias, dB, learning_rate, decay=0.0, clip_norm=5.0)
        else:
            # Fallback (如果你没有更新 base_layer.py)
            with torch.no_grad():
                # Decay
                self.weight.data.mul_(1.0 - weight_decay)
                # Update
                self.weight.data.add_(dW, alpha=learning_rate)
                if self.bias is not None:
                    self.bias.data.add_(dB, alpha=learning_rate)
                
        self.clear_cache()
        return grad_input
