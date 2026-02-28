"""

复数 RMS Normalization
=========================================================================

公式 (前向):
    z_out = (z / rms(z)) · scale
    
    其中 rms(z) = sqrt( mean(|z|²) + ε ),  scale 为可学习实数缩放因子

物理含义:
    将信号的模长约束在单位球附近, 解耦"能量"与"相位信息"。
    没有它, Hebbian 网络会因能量正反馈瞬间发散。
    scale 参数类似探测器增益, 允许各层自适应调节信号强度。

修正说明:
    原版 scale 更新使用 `scale += lr * grad`, 这是梯度上升。
    物理上, 当 grad > 0 意味着"增大 scale 会增大 loss", 
    应该减小 scale, 即使用 `scale -= lr * grad` (梯度下降)。
    
    此修正与 activation.py 中的参数更新方向修正一致,
    确保整条反向传播链上所有可学习参数的更新方向统一为梯度下降。
"""

import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer


class ComplexRMSNorm(ComplexLayer):
    """
    复数 RMS Normalization (Root Mean Square Layer Normalization)
    
    支持 2D (Batch, Dim) 和 3D (Batch, Seq, Dim) 输入。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        # 可学习的缩放参数 (实数, 初始化为 1.0)
        self.scale = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.input_cache = z.detach().clone()

        # 1. 计算 RMS (Root Mean Square of magnitudes)
        norm_z = torch.abs(z)
        rms = torch.sqrt(torch.mean(norm_z ** 2, dim=-1, keepdim=True) + self.eps)

        # 2. 归一化: 保持相位不变, 将模长约束到 ~1
        z_normalized = z / rms

        # 3. 缩放: 乘以可学习增益
        output = z_normalized * self.scale

        # 缓存中间结果供反向使用
        self.output_cache = (z_normalized, rms)
        return output

    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        """
        手动反向传播, 支持任意维度输入。
        
        G = grad_output = dL/d(output)*
        返回 dL/dz* (传给前层)
        """
        z = self.input_cache
        z_normalized, rms = self.output_cache

        # ----- 1. scale 的梯度 -----
        #
        # output = z_norm · scale
        # d(output)/d(scale) = z_norm   (scale 是实数, 所以没有共轭歧义)
        #
        # dL/d(scale) = 2·Re( G · conj(d(output)/d(scale)) )
        #             = 2·Re( G · conj(z_norm) )
        #
        # 实现中计算 Re(G · conj(z_norm)), 常数 2 由学习率吸收。
        #
        grad_scale_per_element = torch.real(grad_output * torch.conj(z_normalized))

        # 展平: 无论 2D 还是 3D, 统一为 (TotalSamples, Dim)
        grad_scale_flat = grad_scale_per_element.view(-1, self.normalized_shape)
        total_samples = grad_scale_flat.shape[0]
        grad_scale = torch.sum(grad_scale_flat, dim=0) / total_samples

        with torch.no_grad():
            # ★ 梯度下降 (修正: 原版使用 += 是梯度上升)
            self.scale.data -= learning_rate * grad_scale

        # ----- 2. 输入 z 的梯度 -----
        #
        # 采用高效 RMSNorm Jacobian 结构:
        #   dL/dz* = scale · [ G_y - z_norm · Re( sum_j G_y_j · conj(z_norm_j) ) / dim ] / rms
        #
        # 其中 G_y = G · scale  (穿过 scale 层的梯度)
        #
        grad_y = grad_output * self.scale

        dot_prod = torch.real(
            torch.sum(grad_y * torch.conj(z_normalized), dim=-1, keepdim=True)
        )

        dim = self.normalized_shape
        numerator = grad_y - z_normalized * (dot_prod / dim)
        grad_input = numerator / rms

        self.clear_cache()
        return grad_input

