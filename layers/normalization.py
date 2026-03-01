"""

复数 RMS Normalization
=========================================================================

v2 变更:
    scale 参数的裸 SGD → AdaptiveParamState (β=0.90, eps=1e-5, init=1e-3)
    与 ComplexLinear 使用完全一致的优化器参数。

    前向和输入梯度数学完全不变。
"""

import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer, AdaptiveParamState


class ComplexRMSNorm(ComplexLayer):
    """
    复数 RMS Normalization

    v2: scale 参数配备 AdaptiveParamState, 替代裸 SGD。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.scale = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32))

        # [v2] 自适应优化器 (参数与 ComplexLinear 完全一致)
        self._scale_optim = AdaptiveParamState(shape=(normalized_shape,))

    def _ensure_optim_device(self):
        self._scale_optim.to(self.scale.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.input_cache = z.detach().clone()

        norm_z = torch.abs(z)
        rms = torch.sqrt(torch.mean(norm_z ** 2, dim=-1, keepdim=True) + self.eps)
        z_normalized = z / rms
        output = z_normalized * self.scale

        self.output_cache = (z_normalized, rms)
        return output

    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        self._ensure_optim_device()

        z = self.input_cache
        z_normalized, rms = self.output_cache

        # scale 梯度
        grad_scale_per_element = 2.0 * torch.real(grad_output * torch.conj(z_normalized))
        grad_scale_flat = grad_scale_per_element.view(-1, self.normalized_shape)
        total_samples = grad_scale_flat.shape[0]
        grad_scale = torch.sum(grad_scale_flat, dim=0) / total_samples

        # [v2] 使用 AdaptiveParamState 替代裸 SGD
        # 原版: self.scale.data -= learning_rate * grad_scale
        self._scale_optim.step(self.scale, grad_scale, learning_rate)

        # 输入梯度 (完全不变)
        grad_y = grad_output * self.scale
        dot_prod = torch.real(
            torch.sum(grad_y * torch.conj(z_normalized), dim=-1, keepdim=True)
        )
        dim = self.normalized_shape
        numerator = grad_y - z_normalized * (dot_prod / dim)
        grad_input = numerator / rms

        self.clear_cache()
        return grad_input
