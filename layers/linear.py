import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer
from Anla.utils.complex_ops import complex_kaiming_normal_

class ComplexLinear(ComplexLayer):
    """
    [GPU Ready + High Viscosity]
    Fix: Increase rest mass and responsiveness to prevent warmup shock.
    """
    def __init__(self, in_features, out_features, bias=True, mode='descent'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        
        weight_real = torch.empty(out_features, in_features)
        weight_imag = torch.empty(out_features, in_features)
        complex_kaiming_normal_(weight_real, weight_imag)
        self.weight = nn.Parameter(torch.complex(weight_real, weight_imag))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)
            
        # [FIX 1] 增加静止质量：1e-5 -> 1e-3
        # 提供更强的初始阻尼
        self.register_buffer('weight_energy', torch.full((out_features, in_features), 1e-3, dtype=torch.float32))
        
        if bias:
            self.register_buffer('bias_energy', torch.full((out_features,), 1e-3, dtype=torch.float32))
        else:
            self.bias_energy = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_cache = x if self.training else None
        with torch.no_grad():
            return nn.functional.linear(x, self.weight, self.bias)

    def manual_backward(self, grad_output: torch.Tensor, learning_rate: float, weight_decay: float = 0.0) -> torch.Tensor:
        x = self.input_cache
        grad_input = grad_output @ self.weight.conj()
        
        if x.dim() > 2:
            x_flat = x.reshape(-1, x.shape[-1])
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        else:
            x_flat = x
            grad_flat = grad_output
            
        d_weight = grad_flat.mT @ x_flat.conj()
        
        with torch.no_grad():
            scale = 1.0 / x_flat.shape[0]
            
            # [FIX 2] 降低惯性：0.99 -> 0.90
            # 让阻力更快地随梯度增大而增大，防止滞后爆炸
            beta = 0.90 
            eps = 1e-5
            
            curr_grad_mag_sq = (d_weight * scale).abs().pow(2)
            self.weight_energy.mul_(beta).add_(curr_grad_mag_sq, alpha=1-beta)
            
            denom = self.weight_energy.sqrt().add_(eps)
            adaptive_step = (d_weight * scale) / denom * learning_rate
            
            if self.bias is not None:
                d_bias = grad_flat.sum(dim=0)
                curr_bias_mag_sq = (d_bias * scale).abs().pow(2)
                self.bias_energy.mul_(beta).add_(curr_bias_mag_sq, alpha=1-beta)
                denom_b = self.bias_energy.sqrt().add_(eps)
                adaptive_step_b = (d_bias * scale) / denom_b * learning_rate
            
            if weight_decay > 0:
                self.weight.data.mul_(1.0 - weight_decay)
                
            if self.mode == 'hebbian':
                self.weight.data.add_(adaptive_step)
                if self.bias is not None: self.bias.data.add_(adaptive_step_b)
            elif self.mode == 'descent':
                self.weight.data.sub_(adaptive_step)
                if self.bias is not None: self.bias.data.sub_(adaptive_step_b)

        return grad_input
