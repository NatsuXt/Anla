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

修正说明 (相对于原版):

Fix 1 — scale 梯度的常数因子 2:
    原版注释 "常数 2 由学习率吸收"。
    但 ComplexLinear 通过 Wirtinger 梯度 dL/dW* = G^T @ X* 
    获得的是完整梯度 (无缺失因子, 因为 Y 关于 W 全纯)。
    而 scale 是实参数, 其完整梯度为 dL/dp = 2·Re(G · conj(df/dp))。
    原版只计算了 Re(...)，导致 scale 的有效学习率是 ComplexLinear
    权重的 1/2，引入层间学习率不均匀。
    → 修正: 显式乘以 2。

    注: 输入梯度 dL/dz* 的公式不受此影响 (已经是完整的 Wirtinger 表达式)。
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
        """
        前向传播: z → z/rms · scale

        Args:
            z: 复数输入张量, shape (..., normalized_shape)

        Returns:
            归一化后的复数张量, shape 同输入
        """
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

        推导 — scale 梯度:
            output = z_norm · scale    (scale 是实参数)
            dL/d(scale) = 2·Re( G · conj(z_norm) )

        推导 — 输入梯度:
            z_norm_d = z_d / rms
            rms² = (1/D) Σ_d |z_d|² + ε

            通过 Wirtinger 微积分:
            ∂z_norm_d/∂z_j* = -z_norm_d · z_norm_j / (2D · rms)
            ∂z_norm_d/∂z_j  = δ_{dj}/rms - z_norm_d · conj(z_norm_j) / (2D · rms)

            dL/dz_j* = Σ_d [ conj(G_y_d) · ∂z_norm_d/∂z_j* + G_y_d · conj(∂z_norm_d/∂z_j) ]
            
            其中 G_y = G · scale (穿过 scale 层的梯度)

            展开后化简:
            dL/dz_j* = [ G_y_j - z_norm_j · Re(Σ_d G_y_d · conj(z_norm_d)) / D ] / rms

            注: Wirtinger 的 1/2 因子与 2·Re(...) 的 2 抵消,
                输入梯度公式中没有多余的常数因子。
        """
        z = self.input_cache
        z_normalized, rms = self.output_cache

        # ----- 1. scale 的梯度 -----
        #
        # output = z_norm · scale   (scale 是实数)
        # d(output)/d(scale) = z_norm
        #
        # dL/d(scale) = 2·Re( G · conj(z_norm) )
        #
        # [Fix 1] 显式乘以 2，不再让学习率静默吸收
        #
        grad_scale_per_element = 2.0 * torch.real(grad_output * torch.conj(z_normalized))

        # 展平: 无论 2D 还是 3D, 统一为 (TotalSamples, Dim)
        grad_scale_flat = grad_scale_per_element.view(-1, self.normalized_shape)
        total_samples = grad_scale_flat.shape[0]
        grad_scale = torch.sum(grad_scale_flat, dim=0) / total_samples

        with torch.no_grad():
            # 梯度下降: scale -= η · (dL/d_scale)
            self.scale.data -= learning_rate * grad_scale

        # ----- 2. 输入 z 的梯度 -----
        #
        # 高效 RMSNorm Jacobian 结构:
        #   dL/dz* = [ G_y - z_norm · Re(Σ_j G_y_j · conj(z_norm_j)) / D ] / rms
        #
        # 其中 G_y = G · scale (穿过 scale 层后的梯度)
        #
        # 注: 此公式已是完整的 Wirtinger 表达式, 无缺失因子。
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
