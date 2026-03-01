"""
保存位置: Anla/core/base_layer.py

修正版 v2 — 新增 AdaptiveParamState 工具类
=========================================================================

唯一变更: 新增 AdaptiveParamState 类, 供 PhaseTwist 和 ComplexRMSNorm 调用,
         使其获得与 ComplexLinear 完全一致的自适应优化器行为。

参数完全复用 ComplexLinear 的原版设定:
    β = 0.90,  eps = 1e-5,  init_energy = 1e-3
    无梯度裁剪 (原版没有, 我们也不加)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AdaptiveParamState:
    """
    为实数参数 (如 PhaseTwist 的 γ/β/φ, RMSNorm 的 scale) 提供
    与 ComplexLinear 完全一致的 RMSProp 风格自适应步长。

    参数完全复用 ComplexLinear 的原版值:
        β = 0.90   (有效窗口 ~10 步, 与原版一致)
        eps = 1e-5 (与原版一致)
        init_energy = 1e-3 (与原版一致)
    """

    def __init__(self, shape: tuple, device: torch.device = None,
                 beta: float = 0.90, eps: float = 1e-5,
                 init_energy: float = 1e-3):
        self.beta = beta
        self.eps = eps
        self.energy = torch.full(shape, init_energy,
                                 dtype=torch.float32, device=device)

    def to(self, device: torch.device):
        self.energy = self.energy.to(device)
        return self

    def step(self, param: nn.Parameter, raw_grad: torch.Tensor,
             lr: float, weight_decay: float = 0.0):
        """
        执行一步自适应梯度下降, 逻辑与 ComplexLinear 完全一致:
            energy = β * energy + (1-β) * |grad|²
            step = grad / (√energy + eps) * lr
            param -= step
        """
        with torch.no_grad():
            grad_mag_sq = raw_grad.pow(2) if not raw_grad.is_complex() else raw_grad.abs().pow(2)
            self.energy.mul_(self.beta).add_(grad_mag_sq, alpha=1.0 - self.beta)
            denom = self.energy.sqrt().add_(self.eps)
            adaptive_step = raw_grad / denom * lr

            if weight_decay > 0:
                param.data.mul_(1.0 - weight_decay)

            param.data.sub_(adaptive_step)


class ComplexLayer(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.input_cache = None
        self.output_cache = None

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        pass

    def clear_cache(self):
        self.input_cache = None
        self.output_cache = None
