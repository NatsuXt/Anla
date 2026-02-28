"""

双向耦合激活函数 (Bidirectional AM-PM / PM-AM Coupling Activation)
=========================================================================

公式:
    z_out = tanh( r · (1 + β·cos(θ - φ)) ) · e^{i(θ + γ·r)}

其中:
    r = |z|,  θ = angle(z)
    γ  — PM→AM 耦合系数:  输入模长如何扭曲输出相位   [可学习, per-channel]
    β  — AM←PM 耦合强度:  输入相位如何调制输出模长   [可学习, per-channel]
    φ  — AM←PM 参考相位:  调制的相位基准            [可学习, per-channel]

物理解读:
    · tanh(·) 提供模长饱和（能量有界，防止共振灾难）
    · β·cos(θ-φ) 使模长响应依赖于信号相位（AM←PM: 相位影响强度）
      — 当 θ 与 φ 对齐时增强, 正交时衰减, 实现方向选择性
    · γ·r 使输出相位依赖于输入强度（PM→AM: 强度影响相位）
      — 强信号获得更大的相位旋转, 实现非线性频率混合
    · 双向耦合赋予网络处理非线性干涉与拓扑折叠的完整能力

反向传播: 严格 Wirtinger 微积分
=========================================================================

本实现修正了原版 PhaseTwist 中的两处数学错误:

错误 1 — 输入梯度的共轭方向:
    原版:  grad_input = conj(df/dz)·G + conj(df/dz*)·conj(G)
    正确:  grad_input = conj(df/dz)·G + (df/dz*)·conj(G)
    
    推导依据 (Wirtinger 链式法则):
        dL/dz* = (dL/df)·(df/dz*) + (dL/df*)·(df*/dz*)
        其中 dL/df = conj(G),  df*/dz* = conj(df/dz)
        => dL/dz* = conj(G)·(df/dz*) + G·conj(df/dz)

错误 2 — 参数更新方向:
    原版:  param += lr * Re(G · conj(df/dp))   (梯度上升!)
    正确:  param -= lr * Re(G · conj(df/dp))   (梯度下降)
    
    推导: dL/dp = 2·Re(G · conj(df/dp)),  下降方向为 p -= η·(dL/dp)
"""

import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer


class PhaseTwist(ComplexLayer):
    """
    双向耦合复数激活函数。

    接口与原版 PhaseTwist 完全兼容:
        forward(z) -> z_out
        manual_backward(grad_output, learning_rate, **kwargs) -> grad_input

    新增参数:
        beta  — AM←PM 耦合强度 (init=0.01, 足够小以保证训练初期稳定)
        phi   — AM←PM 参考相位 (init=0.0)
    """

    def __init__(self, channels: int,
                 init_gamma: float = 0.01,
                 init_beta: float = 0.01,
                 init_phi: float = 0.0):
        super().__init__()
        self.channels = channels

        # PM→AM: 输入模长 → 输出相位扭曲
        self.gamma = nn.Parameter(
            torch.full((channels,), init_gamma, dtype=torch.float32))

        # AM←PM: 输入相位 → 输出模长调制
        self.beta = nn.Parameter(
            torch.full((channels,), init_beta, dtype=torch.float32))

        # AM←PM 参考相位基准
        self.phi = nn.Parameter(
            torch.full((channels,), init_phi, dtype=torch.float32))

    # ------------------------------------------------------------------
    #  辅助: 将 (channels,) 参数广播到 z 的形状
    #  支持 2D (Batch, Ch) 和 3D (Batch, Seq, Ch)
    # ------------------------------------------------------------------
    def _broadcast(self, param: torch.Tensor, ndim: int) -> torch.Tensor:
        """将 (channels,) 的参数 reshape 为 (1,...,1, channels)，维度数 = ndim。"""
        shape = [1] * (ndim - 1) + [self.channels]
        return param.view(*shape)

    # ==================================================================
    #  前向传播
    # ==================================================================
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 缓存输入供反向使用（训练态）
        self.input_cache = z.detach().clone() if self.training else None

        eps = 1e-8
        r = torch.abs(z) + eps           # 模长  (Batch, ..., Ch)
        theta = torch.angle(z)           # 相位

        # 广播参数
        gamma = self._broadcast(self.gamma, z.dim())
        beta = self._broadcast(self.beta, z.dim())
        phi = self._broadcast(self.phi, z.dim())

        # --- 双向耦合 ---
        # AM←PM 支路: 相位差 cos(θ-φ) 调制有效模长
        cos_diff = torch.cos(theta - phi)
        m = r * (1.0 + beta * cos_diff)  # 调制后的有效模长
        r_out = torch.tanh(m)            # 饱和压缩 → 输出模长

        # PM→AM 支路: 输入模长扭曲输出相位
        theta_out = theta + gamma * r    # 输出相位

        # 合成输出
        output = torch.polar(r_out, theta_out)
        return output

    # ==================================================================
    #  手动反向传播 (Wirtinger Calculus)
    # ==================================================================
    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        """
        接收上游传来的复数误差向量 G = dL/df*,
        计算并返回 dL/dz* (传给前层),
        同时原地更新 gamma, beta, phi (梯度下降).
        """
        z = self.input_cache
        if z is None:
            # 非训练态, 直接透传
            return grad_output

        eps = 1e-9
        r = torch.abs(z) + eps
        theta = torch.angle(z)

        # 广播参数
        gamma = self._broadcast(self.gamma, z.dim())
        beta = self._broadcast(self.beta, z.dim())
        phi = self._broadcast(self.phi, z.dim())

        # ----- 重算前向中间量 (避免额外缓存, 节省显存) -----
        cos_diff = torch.cos(theta - phi)
        sin_diff = torch.sin(theta - phi)
        m = r * (1.0 + beta * cos_diff)
        tanh_m = torch.tanh(m)
        sech2_m = 1.0 - tanh_m ** 2       # sech²(m)
        theta_out = theta + gamma * r
        e_i_tout = torch.polar(torch.ones_like(theta_out), theta_out)   # e^{iθ_out}
        f = torch.polar(tanh_m, theta_out)  # 前向输出 (重算)

        # =============================================================
        #  Step 1: 对极坐标 (r, θ) 的偏导数
        # =============================================================
        #
        #  df/dr = sech²(m) · (1 + β·cos(θ-φ)) · e^{iθ_out}  +  iγ · f
        #
        #  推导:
        #    f = tanh(m) · e^{iθ_out}
        #    dm/dr = 1 + β·cos(θ-φ)
        #    dθ_out/dr = γ
        #    df/dr = [d(tanh(m))/dr] · e^{iθ_out}  +  tanh(m) · [d(e^{iθ_out})/dr]
        #          = sech²(m)·(dm/dr)·e^{iθ_out}   +  tanh(m)·iγ·e^{iθ_out}
        #          = sech²(m)·(1+β·cos(θ-φ))·e^{iθ_out} + iγ·f
        #
        df_dr = sech2_m * (1.0 + beta * cos_diff) * e_i_tout + 1j * gamma * f

        #  df/dθ = sech²(m) · (-r·β·sin(θ-φ)) · e^{iθ_out}  +  i · f
        #
        #  推导:
        #    dm/dθ = r · β · (-sin(θ-φ))
        #    dθ_out/dθ = 1
        #    df/dθ = sech²(m)·(dm/dθ)·e^{iθ_out}  +  tanh(m)·i·e^{iθ_out}
        #          = -sech²(m)·r·β·sin(θ-φ)·e^{iθ_out} + i·f
        #
        df_dtheta = -sech2_m * r * beta * sin_diff * e_i_tout + 1j * f

        # =============================================================
        #  Step 2: 极坐标 → Wirtinger 坐标变换
        # =============================================================
        #
        #  标准 Wirtinger 变换因子:
        #    dr/dz     = (1/2) · e^{-iθ}      dr/dz*    = (1/2) · e^{iθ}
        #    dθ/dz     = -i/(2r) · e^{-iθ}    dθ/dz*    = i/(2r) · e^{iθ}
        #
        z_hat = z / r       # e^{iθ}  (归一化方向)
        z_hat_c = z_hat.conj()  # e^{-iθ}

        #  df/dz  = df/dr · dr/dz  + df/dθ · dθ/dz
        #         = df/dr · (1/2)·e^{-iθ}  +  df/dθ · (-i/(2r))·e^{-iθ}
        df_dz = df_dr * (0.5 * z_hat_c) + df_dtheta * (-0.5j / r * z_hat_c)

        #  df/dz* = df/dr · dr/dz* + df/dθ · dθ/dz*
        #         = df/dr · (1/2)·e^{iθ}  +  df/dθ · (i/(2r))·e^{iθ}
        df_dz_conj = df_dr * (0.5 * z_hat) + df_dtheta * (0.5j / r * z_hat)

        # =============================================================
        #  Step 3: 输入梯度 (修正后的 Wirtinger 链式法则)
        # =============================================================
        #
        #  dL/dz* = (dL/df) · (df/dz*)  +  (dL/df*) · (df*/dz*)
        #
        #  令 G ≡ grad_output = dL/df*,  则 dL/df = conj(G)
        #  又 df*/dz* = conj(df/dz)
        #
        #  => dL/dz* = conj(G) · (df/dz*)  +  G · conj(df/dz)
        #
        grad_input = (
            torch.conj(grad_output) * df_dz_conj
            + grad_output * torch.conj(df_dz)
        )

        # =============================================================
        #  Step 4: 参数梯度
        # =============================================================
        #
        #  对实参数 p ∈ {γ, β, φ}:
        #    dL/dp = (dL/df)·(df/dp) + (dL/df*)·(df*/dp)
        #          = conj(G)·(df/dp) + G·conj(df/dp)
        #          = 2·Re( G · conj(df/dp) )
        #
        #  实现中计算 Re(G · conj(df/dp))，常数 2 由学习率吸收。
        #

        # (a) γ 梯度:  df/dγ = i · f · r
        #     (增大 γ → 输出相位多旋转 → f 在复平面上转动)
        df_dgamma = 1j * f * r
        d_gamma_elem = torch.real(grad_output * torch.conj(df_dgamma))

        # (b) β 梯度:  df/dβ = sech²(m) · r · cos(θ-φ) · e^{iθ_out}
        #     (增大 β → 相位-模长耦合更强 → 模长响应随相位振荡更剧烈)
        df_dbeta = sech2_m * r * cos_diff * e_i_tout
        d_beta_elem = torch.real(grad_output * torch.conj(df_dbeta))

        # (c) φ 梯度:  df/dφ = sech²(m) · r · β · sin(θ-φ) · e^{iθ_out}
        #     (增大 φ → 参考相位旋转 → 改变哪些方向被增强/衰减)
        df_dphi = sech2_m * r * beta * sin_diff * e_i_tout
        d_phi_elem = torch.real(grad_output * torch.conj(df_dphi))

        # =============================================================
        #  Step 5: 聚合 & 更新参数
        # =============================================================
        # 展平 batch / seq 维度，对 channels 维聚合取均值
        total_samples = z.numel() // z.shape[-1]

        d_gamma = d_gamma_elem.reshape(-1, self.channels).sum(dim=0) / total_samples
        d_beta = d_beta_elem.reshape(-1, self.channels).sum(dim=0) / total_samples
        d_phi = d_phi_elem.reshape(-1, self.channels).sum(dim=0) / total_samples

        with torch.no_grad():
            # ★ 梯度下降 (原版使用 += 是梯度上升，此处修正为 -=)
            self.gamma.data -= learning_rate * d_gamma
            self.beta.data -= learning_rate * d_beta
            self.phi.data -= learning_rate * d_phi

        self.clear_cache()
        return grad_input

