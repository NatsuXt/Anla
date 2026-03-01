"""

双向耦合激活函数 (Bidirectional AM-PM / PM-AM Coupling Activation)
=========================================================================

v2 变更:
    γ, β, φ 的裸 SGD → AdaptiveParamState (β=0.90, eps=1e-5, init=1e-3)
    与 ComplexLinear 使用完全一致的优化器参数, 消除层间学习率鸿沟。

    前向传播和梯度数学完全不变。
"""

import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer, AdaptiveParamState


# ========== 统一的数值常量 ==========
EPS = 1e-7
EPS_SAFE = 1e-4


class PhaseTwist(ComplexLayer):
    """
    双向耦合复数激活函数。

    v2: γ, β, φ 各自配备独立的 AdaptiveParamState, 替代裸 SGD。
    """

    def __init__(self, channels: int,
                 init_gamma: float = 0.01,
                 init_beta: float = 0.01,
                 init_phi: float = 0.0):
        super().__init__()
        self.channels = channels

        self.gamma = nn.Parameter(
            torch.full((channels,), init_gamma, dtype=torch.float32))
        self.beta = nn.Parameter(
            torch.full((channels,), init_beta, dtype=torch.float32))
        self.phi = nn.Parameter(
            torch.full((channels,), init_phi, dtype=torch.float32))

        # [v2] 自适应优化器 (参数与 ComplexLinear 完全一致)
        self._gamma_optim = AdaptiveParamState(shape=(channels,))
        self._beta_optim = AdaptiveParamState(shape=(channels,))
        self._phi_optim = AdaptiveParamState(shape=(channels,))

    def _ensure_optim_device(self):
        device = self.gamma.device
        self._gamma_optim.to(device)
        self._beta_optim.to(device)
        self._phi_optim.to(device)

    def _broadcast(self, param: torch.Tensor, ndim: int) -> torch.Tensor:
        shape = [1] * (ndim - 1) + [self.channels]
        return param.view(*shape)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.input_cache = z.detach().clone() if self.training else None

        r = torch.abs(z) + EPS
        theta = torch.angle(z)

        gamma = self._broadcast(self.gamma, z.dim())
        beta = self._broadcast(self.beta, z.dim())
        phi = self._broadcast(self.phi, z.dim())

        cos_diff = torch.cos(theta - phi)
        m = r * (1.0 + beta * cos_diff)
        r_out = torch.tanh(m)
        theta_out = theta + gamma * r

        output = torch.polar(r_out, theta_out)
        return output

    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        z = self.input_cache
        if z is None:
            return grad_output

        self._ensure_optim_device()

        r = torch.abs(z) + EPS
        theta = torch.angle(z)

        gamma = self._broadcast(self.gamma, z.dim())
        beta = self._broadcast(self.beta, z.dim())
        phi = self._broadcast(self.phi, z.dim())

        cos_diff = torch.cos(theta - phi)
        sin_diff = torch.sin(theta - phi)
        m = r * (1.0 + beta * cos_diff)
        tanh_m = torch.tanh(m)
        sech2_m = 1.0 - tanh_m ** 2
        theta_out = theta + gamma * r
        e_i_tout = torch.polar(torch.ones_like(theta_out), theta_out)
        f = torch.polar(tanh_m, theta_out)

        # Step 1: 极坐标偏导数
        df_dr = sech2_m * (1.0 + beta * cos_diff) * e_i_tout + 1j * gamma * f
        df_dtheta = -sech2_m * r * beta * sin_diff * e_i_tout + 1j * f

        # Step 2: Wirtinger 变换
        z_hat = z / r
        z_hat_c = z_hat.conj()
        safe_inv_r = 1.0 / torch.clamp(r, min=EPS_SAFE)

        df_dz = df_dr * (0.5 * z_hat_c) + df_dtheta * (-0.5j * safe_inv_r * z_hat_c)
        df_dz_conj = df_dr * (0.5 * z_hat) + df_dtheta * (0.5j * safe_inv_r * z_hat)

        # Step 3: 输入梯度
        grad_input = (
            torch.conj(grad_output) * df_dz_conj
            + grad_output * torch.conj(df_dz)
        )

        # Step 4: 参数梯度
        df_dgamma = 1j * f * r
        d_gamma_elem = 2.0 * torch.real(grad_output * torch.conj(df_dgamma))

        df_dbeta = sech2_m * r * cos_diff * e_i_tout
        d_beta_elem = 2.0 * torch.real(grad_output * torch.conj(df_dbeta))

        df_dphi = sech2_m * r * beta * sin_diff * e_i_tout
        d_phi_elem = 2.0 * torch.real(grad_output * torch.conj(df_dphi))

        # Step 5: 聚合 & 更新
        total_samples = z.numel() // z.shape[-1]

        d_gamma = d_gamma_elem.reshape(-1, self.channels).sum(dim=0) / total_samples
        d_beta = d_beta_elem.reshape(-1, self.channels).sum(dim=0) / total_samples
        d_phi = d_phi_elem.reshape(-1, self.channels).sum(dim=0) / total_samples

        # [v2] 使用 AdaptiveParamState 替代裸 SGD
        # 原版: self.gamma.data -= learning_rate * d_gamma
        self._gamma_optim.step(self.gamma, d_gamma, learning_rate)
        self._beta_optim.step(self.beta, d_beta, learning_rate)
        self._phi_optim.step(self.phi, d_phi, learning_rate)

        self.clear_cache()
        return grad_input
