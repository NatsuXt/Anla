import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class ComplexLayer(nn.Module, ABC):
    """
    Anla 架构的核心基类。
    """
    def __init__(self):
        super().__init__()
        self.input_cache = None
        self.output_cache = None

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def manual_backward(self, grad_output: torch.Tensor, learning_rate: float, **kwargs) -> torch.Tensor:
        pass

    def clear_cache(self):
        self.input_cache = None
        self.output_cache = None

    def _apply_gradient(self, param: nn.Parameter, grad: torch.Tensor, learning_rate: float, decay: float = 0.0, clip_norm: float = 1.0):
        """
        [新增]: 统一的参数更新原子操作。
        包含: Weight Decay -> Gradient Clipping -> Update
        
        Args:
            param: 要更新的参数 (W or B)
            grad: 计算出的原始梯度
            learning_rate: 学习率
            decay: 权重衰减系数
            clip_norm: 梯度裁剪阈值 (模长上限)
        """
        if grad is None:
            return

        with torch.no_grad():
            # 1. 梯度裁剪 (Gradient Clipping)
            # 防止单次更新步长过大导致相位翻转
            # 计算梯度的模长
            grad_norm = torch.abs(grad)
            # 如果是矩阵，通常计算整体范数，或者按元素裁剪。
            # 这里为了物理上的“限制更新速率”，我们采用按元素裁剪 (Element-wise Clipping)
            # 确保没有任何一个权重的改变量超过 clip_norm
            scale = torch.clamp(clip_norm / (grad_norm + 1e-6), max=1.0)
            clipped_grad = grad * scale
            
            # 2. 权重衰减 (Weight Decay / Damping)
            if decay > 0:
                param.data.mul_(1.0 - decay)
            
            # 3. 执行更新 (Hebbian: W += lr * grad)
            param.data.add_(clipped_grad, alpha=learning_rate)
