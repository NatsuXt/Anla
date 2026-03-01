"""
保存位置: Anla/wirtinger_gradcheck.py

Wirtinger 有限差分梯度验证器
=========================================================================

目的:
    对 Anla 每一层的 manual_backward 实现进行数值验证。
    用有限差分逼近真实的 Wirtinger 梯度 dL/dz*，与 manual_backward
    返回的 grad_input 做对比。

核心公式:
    定义标量损失 L(z) = Re( sum( conj(G) · f(z) ) )
    其中 G 是随机上游梯度，f 是层的前向函数。

    则 dL/dz_k* = (1/2) * [
        (L(z + ε·e_k) - L(z - ε·e_k)) / (2ε)
      + i·(L(z + iε·e_k) - L(z - iε·e_k)) / (2ε)
    ]

    这应该等于 manual_backward(G) 在第 k 个元素的值。

检测层 (按嫌疑排序):
    1. MagPhaseSoftmax — 头号嫌疑, 非严格推导
    2. HolographicAttention — 完整注意力链路
    3. PhaseTwist — 新版双向耦合
    4. ComplexRMSNorm — 刚修正符号
    5. ComplexLinear — 基础组件
    6. ComplexRotaryEmbedding — 应该透明

用法:
    python -m Anla.wirtinger_gradcheck
    python -m Anla.wirtinger_gradcheck --layer MagPhaseSoftmax
    python -m Anla.wirtinger_gradcheck --layer all --eps 1e-5 --n-samples 20
"""

import argparse
import copy
import sys
import os
import time
from typing import Callable, Dict, List, Optional, Tuple, Any
from collections import OrderedDict

import torch
import torch.nn as nn

# ==========================================================================
#  核心: Wirtinger 有限差分
# ==========================================================================

def wirtinger_fd_gradient(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    upstream_grad: torch.Tensor,
    eps: float = 1e-5,
    sample_indices: Optional[List[Tuple[int, ...]]] = None,
) -> torch.Tensor:
    """
    用有限差分计算 dL/dz*，其中 L(z) = Re(sum(conj(G) * f(z)))。

    Parameters
    ----------
    forward_fn : callable
        z -> f(z)，纯函数（无副作用）
    z : torch.Tensor (complex)
        输入张量
    upstream_grad : torch.Tensor (complex)
        上游梯度 G = dL/df*
    eps : float
        有限差分步长
    sample_indices : list of tuples, optional
        只在这些位置计算有限差分（加速大张量）
        如果 None，计算所有位置

    Returns
    -------
    grad_fd : torch.Tensor (complex)
        有限差分逼近的 dL/dz*
    """
    z_flat = z.detach().clone().reshape(-1)
    n = z_flat.shape[0]

    # 如果没有指定采样位置，全部计算
    if sample_indices is None:
        indices = list(range(n))
    else:
        indices = [_multi_to_flat(idx, z.shape) for idx in sample_indices]

    grad_fd_flat = torch.zeros(n, dtype=z.dtype, device=z.device)

    G = upstream_grad.detach()

    def scalar_loss(z_perturbed_flat):
        z_p = z_perturbed_flat.reshape(z.shape)
        f_val = forward_fn(z_p)
        # L = 2 * Re(sum(conj(G) * f))
        # 因子 2 使得 dL/dz* 与 Wirtinger 链式法则约定一致:
        #   dL/dz* = conj(G)·(df/dz*) + G·conj(df/dz)
        # 不含 2 时, Re(sum(conj(G)*f)) 的 Wirtinger 导数只有上式的 1/2
        return 2.0 * torch.real(torch.sum(torch.conj(G) * f_val)).item()

    for k in indices:
        # 实方向扰动: (L(z+εe_k) - L(z-εe_k)) / (2ε)
        z_plus = z_flat.clone()
        z_plus[k] = z_plus[k] + eps
        z_minus = z_flat.clone()
        z_minus[k] = z_minus[k] - eps
        dL_real = (scalar_loss(z_plus) - scalar_loss(z_minus)) / (2.0 * eps)

        # 虚方向扰动: (L(z+iεe_k) - L(z-iεe_k)) / (2ε)
        z_plus_i = z_flat.clone()
        z_plus_i[k] = z_plus_i[k] + 1j * eps
        z_minus_i = z_flat.clone()
        z_minus_i[k] = z_minus_i[k] - 1j * eps
        dL_imag = (scalar_loss(z_plus_i) - scalar_loss(z_minus_i)) / (2.0 * eps)

        # dL/dz* = (1/2)(dL_real + i * dL_imag)
        grad_fd_flat[k] = 0.5 * (dL_real + 1j * dL_imag)

    return grad_fd_flat.reshape(z.shape)


def _multi_to_flat(multi_idx, shape):
    """多维索引转一维索引"""
    flat = 0
    for i, s in zip(multi_idx, shape):
        flat = flat * s + i
    return flat


def random_sample_indices(shape, n_samples=20, rng=None):
    """随机选取 n_samples 个位置"""
    if rng is None:
        rng = torch.Generator()
    total = 1
    for s in shape:
        total *= s
    n_samples = min(n_samples, total)
    flat_indices = torch.randperm(total, generator=rng)[:n_samples]
    multi_indices = []
    for fi in flat_indices:
        fi = fi.item()
        idx = []
        for s in reversed(shape):
            idx.append(fi % s)
            fi //= s
        idx.reverse()
        multi_indices.append(tuple(idx))
    return multi_indices


# ==========================================================================
#  比较指标
# ==========================================================================

def compare_gradients(
    grad_manual: torch.Tensor,
    grad_fd: torch.Tensor,
    sample_indices: Optional[List[Tuple[int, ...]]] = None,
) -> Dict[str, float]:
    """比较手动梯度与有限差分梯度"""
    if sample_indices is not None:
        # 只比较采样位置
        m_vals = torch.stack([grad_manual[idx] for idx in sample_indices])
        f_vals = torch.stack([grad_fd[idx] for idx in sample_indices])
    else:
        m_vals = grad_manual.reshape(-1)
        f_vals = grad_fd.reshape(-1)

    # 余弦相似度 (把复数展开为实虚拼接的实向量)
    m_real = torch.cat([m_vals.real, m_vals.imag])
    f_real = torch.cat([f_vals.real, f_vals.imag])

    dot = torch.dot(m_real, f_real)
    norm_m = m_real.norm()
    norm_f = f_real.norm()
    cosine = (dot / (norm_m * norm_f + 1e-30)).item()

    # 相对误差
    diff = (m_vals - f_vals).abs()
    rel_err = (diff / (f_vals.abs() + 1e-30)).mean().item()
    max_rel_err = (diff / (f_vals.abs() + 1e-30)).max().item()

    # 绝对误差
    abs_err = diff.mean().item()
    max_abs_err = diff.max().item()

    # 幅值比
    mag_ratio = (norm_m / (norm_f + 1e-30)).item()

    return {
        "cosine_similarity": cosine,
        "mean_relative_error": rel_err,
        "max_relative_error": max_rel_err,
        "mean_absolute_error": abs_err,
        "max_absolute_error": max_abs_err,
        "magnitude_ratio": mag_ratio,
        "manual_norm": norm_m.item(),
        "fd_norm": norm_f.item(),
        "n_compared": len(m_vals),
    }


# ==========================================================================
#  参数保存/恢复 (处理 manual_backward 的副作用)
# ==========================================================================

def save_layer_state(layer: nn.Module) -> Dict[str, torch.Tensor]:
    """保存层的全部参数和 buffer 状态"""
    state = {}
    for name, param in layer.named_parameters():
        state[f"param_{name}"] = param.data.clone()
    for name, buf in layer.named_buffers():
        state[f"buffer_{name}"] = buf.clone()
    return state


def restore_layer_state(layer: nn.Module, state: Dict[str, torch.Tensor]):
    """恢复层的状态"""
    for name, param in layer.named_parameters():
        key = f"param_{name}"
        if key in state:
            param.data.copy_(state[key])
    for name, buf in layer.named_buffers():
        key = f"buffer_{name}"
        if key in state:
            buf.copy_(state[key])


# ==========================================================================
#  逐层测试函数
# ==========================================================================

def test_mag_phase_softmax(eps=1e-5, n_samples=30, seed=42):
    """
    测试 MagPhaseSoftmax 的 manual_backward。
    头号嫌疑：非严格推导的近似反传。
    """
    from Anla.layers.holographic_attention import MagPhaseSoftmax

    torch.manual_seed(seed)
    layer = MagPhaseSoftmax(dim=-1)
    layer.train()

    # 输入: 注意力分数矩阵 (Batch=2, Heads=2, Seq_Q=4, Seq_K=4)
    B, H, SQ, SK = 2, 2, 4, 4
    scale = 1.0 / (16 ** 0.5)  # 模拟 1/sqrt(head_dim)

    z = torch.randn(B, H, SQ, SK, dtype=torch.cfloat) * 0.5 + 0.3
    G = torch.randn(B, H, SQ, SK, dtype=torch.cfloat) * 0.1

    # --- 手动反传 ---
    state = save_layer_state(layer)
    layer.forward(z, scale=scale)
    grad_manual = layer.manual_backward(G, learning_rate=0.001)
    restore_layer_state(layer, state)

    # --- 有限差分 ---
    sample_idx = random_sample_indices(z.shape, n_samples, torch.Generator().manual_seed(seed))

    def fwd_fn(z_in):
        # 需要重新创建层实例以避免缓存污染
        layer_clean = MagPhaseSoftmax(dim=-1)
        layer_clean.eval()
        return layer_clean.forward(z_in, scale=scale)

    grad_fd = wirtinger_fd_gradient(fwd_fn, z, G, eps=eps, sample_indices=sample_idx)

    return compare_gradients(grad_manual, grad_fd, sample_idx)


def test_phase_twist(eps=1e-5, n_samples=30, seed=42):
    """
    测试 PhaseTwist (双向耦合激活函数) 的 manual_backward。
    """
    from Anla.layers.activation import PhaseTwist

    torch.manual_seed(seed)
    channels = 16
    layer = PhaseTwist(channels, init_gamma=0.05, init_beta=0.05, init_phi=0.1)
    layer.train()

    B, S = 2, 8
    z = torch.randn(B, S, channels, dtype=torch.cfloat) * 0.5 + 0.3
    G = torch.randn(B, S, channels, dtype=torch.cfloat) * 0.1

    # --- 手动反传 ---
    state = save_layer_state(layer)
    layer.forward(z)
    grad_manual = layer.manual_backward(G, learning_rate=0.001)
    restore_layer_state(layer, state)

    # --- 有限差分 ---
    sample_idx = random_sample_indices(z.shape, n_samples, torch.Generator().manual_seed(seed))

    # 固定参数快照
    gamma_snap = layer.gamma.data.clone()
    beta_snap = layer.beta.data.clone()
    phi_snap = layer.phi.data.clone()

    def fwd_fn(z_in):
        layer_eval = PhaseTwist(channels)
        layer_eval.gamma.data.copy_(gamma_snap)
        layer_eval.beta.data.copy_(beta_snap)
        layer_eval.phi.data.copy_(phi_snap)
        layer_eval.eval()
        return layer_eval.forward(z_in)

    grad_fd = wirtinger_fd_gradient(fwd_fn, z, G, eps=eps, sample_indices=sample_idx)

    return compare_gradients(grad_manual, grad_fd, sample_idx)


def test_complex_rms_norm(eps=1e-5, n_samples=30, seed=42):
    """
    测试 ComplexRMSNorm 的 manual_backward。
    """
    from Anla.layers.normalization import ComplexRMSNorm

    torch.manual_seed(seed)
    dim = 16
    layer = ComplexRMSNorm(dim)
    layer.train()

    B, S = 2, 8
    z = torch.randn(B, S, dim, dtype=torch.cfloat) * 0.5 + 0.3
    G = torch.randn(B, S, dim, dtype=torch.cfloat) * 0.1

    state = save_layer_state(layer)
    layer.forward(z)
    grad_manual = layer.manual_backward(G, learning_rate=0.001)
    restore_layer_state(layer, state)

    sample_idx = random_sample_indices(z.shape, n_samples, torch.Generator().manual_seed(seed))

    scale_snap = layer.scale.data.clone()

    def fwd_fn(z_in):
        lyr = ComplexRMSNorm(dim)
        lyr.scale.data.copy_(scale_snap)
        lyr.eval()
        return lyr.forward(z_in)

    grad_fd = wirtinger_fd_gradient(fwd_fn, z, G, eps=eps, sample_indices=sample_idx)

    return compare_gradients(grad_manual, grad_fd, sample_idx)


def test_complex_linear(eps=1e-5, n_samples=30, seed=42):
    """
    测试 ComplexLinear 的 manual_backward (输入梯度部分)。
    """
    from Anla.layers.linear import ComplexLinear

    torch.manual_seed(seed)
    in_feat, out_feat = 16, 16
    layer = ComplexLinear(in_feat, out_feat, bias=True, mode='descent')
    layer.train()

    B, S = 2, 8
    z = torch.randn(B, S, in_feat, dtype=torch.cfloat) * 0.3
    G = torch.randn(B, S, out_feat, dtype=torch.cfloat) * 0.1

    state = save_layer_state(layer)
    layer.forward(z)
    grad_manual = layer.manual_backward(G, learning_rate=0.001, weight_decay=0.0)
    restore_layer_state(layer, state)

    sample_idx = random_sample_indices(z.shape, n_samples, torch.Generator().manual_seed(seed))

    w_snap = layer.weight.data.clone()
    b_snap = layer.bias.data.clone() if layer.bias is not None else None

    def fwd_fn(z_in):
        lyr = ComplexLinear(in_feat, out_feat, bias=True, mode='descent')
        lyr.weight.data.copy_(w_snap)
        if b_snap is not None:
            lyr.bias.data.copy_(b_snap)
        lyr.eval()
        return lyr.forward(z_in)

    grad_fd = wirtinger_fd_gradient(fwd_fn, z, G, eps=eps, sample_indices=sample_idx)

    return compare_gradients(grad_manual, grad_fd, sample_idx)


def test_complex_rotary(eps=1e-5, n_samples=30, seed=42):
    """
    测试 ComplexRotaryEmbedding 的 manual_backward。
    """
    from Anla.layers.positional import ComplexRotaryEmbedding

    torch.manual_seed(seed)
    dim = 16
    layer = ComplexRotaryEmbedding(dim, max_seq_len=64)

    B, S = 2, 8
    z = torch.randn(B, S, dim, dtype=torch.cfloat) * 0.5
    G = torch.randn(B, S, dim, dtype=torch.cfloat) * 0.1

    layer.forward(z)  # rotary has no training state to worry about
    grad_manual = layer.manual_backward(G)

    sample_idx = random_sample_indices(z.shape, n_samples, torch.Generator().manual_seed(seed))

    def fwd_fn(z_in):
        lyr = ComplexRotaryEmbedding(dim, max_seq_len=64)
        lyr.rotary_emb = layer.rotary_emb.clone()
        return lyr.forward(z_in)

    grad_fd = wirtinger_fd_gradient(fwd_fn, z, G, eps=eps, sample_indices=sample_idx)

    return compare_gradients(grad_manual, grad_fd, sample_idx)


def test_holographic_attention(eps=1e-5, n_samples=20, seed=42):
    """
    测试 HolographicAttention 的端到端 manual_backward。
    注意: 这包含了 MagPhaseSoftmax + 4 个 ComplexLinear 的组合。
    """
    from Anla.layers.holographic_attention import HolographicAttention

    torch.manual_seed(seed)
    d_model = 16
    num_heads = 2
    layer = HolographicAttention(d_model, num_heads=num_heads)
    layer.train()

    B, S = 2, 6
    z = torch.randn(B, S, d_model, dtype=torch.cfloat) * 0.3
    G = torch.randn(B, S, d_model, dtype=torch.cfloat) * 0.1

    state = save_layer_state(layer)
    layer.forward(z)
    grad_manual = layer.manual_backward(G, learning_rate=0.001, weight_decay=0.0)
    restore_layer_state(layer, state)

    sample_idx = random_sample_indices(z.shape, n_samples, torch.Generator().manual_seed(seed))

    # 需要深拷贝整个层（含所有子层权重）
    def make_clean_layer():
        lyr = HolographicAttention(d_model, num_heads=num_heads)
        lyr.load_state_dict(
            {k: v.clone() for k, v in
             zip(layer.state_dict().keys(),
                 [state.get(f"param_{k}", state.get(f"buffer_{k}", v))
                  for k, v in layer.state_dict().items()])}
        )
        # 直接用保存的 state_dict 更简洁
        return lyr

    # 保存完整 state_dict
    full_sd = {k: v.clone() for k, v in layer.state_dict().items()}
    # 恢复到检查点状态
    restore_layer_state(layer, state)

    def fwd_fn(z_in):
        lyr = HolographicAttention(d_model, num_heads=num_heads)
        lyr.load_state_dict({k: v.clone() for k, v in full_sd.items()})
        lyr.eval()
        return lyr.forward(z_in)

    grad_fd = wirtinger_fd_gradient(fwd_fn, z, G, eps=eps, sample_indices=sample_idx)

    return compare_gradients(grad_manual, grad_fd, sample_idx)


def test_transformer_block(eps=1e-5, n_samples=15, seed=42):
    """
    测试 ComplexTransformerBlock 的端到端 manual_backward。
    最全面的测试：包含注意力 + FFN + Norm + Residual。
    """
    from Anla.layers.transformer_block import ComplexTransformerBlock

    torch.manual_seed(seed)
    d_model = 16
    num_heads = 2
    layer = ComplexTransformerBlock(d_model, num_heads=num_heads, ff_mult=2)
    layer.train()

    B, S = 2, 6
    z = torch.randn(B, S, d_model, dtype=torch.cfloat) * 0.3
    G = torch.randn(B, S, d_model, dtype=torch.cfloat) * 0.1

    state = save_layer_state(layer)
    layer.forward(z)
    grad_manual = layer.manual_backward(G, lr=0.001, wd=0.0)
    restore_layer_state(layer, state)

    sample_idx = random_sample_indices(z.shape, n_samples, torch.Generator().manual_seed(seed))

    full_sd = {k: v.clone() for k, v in layer.state_dict().items()}
    restore_layer_state(layer, state)

    def fwd_fn(z_in):
        lyr = ComplexTransformerBlock(d_model, num_heads=num_heads, ff_mult=2)
        lyr.load_state_dict({k: v.clone() for k, v in full_sd.items()})
        lyr.eval()
        return lyr.forward(z_in)

    grad_fd = wirtinger_fd_gradient(fwd_fn, z, G, eps=eps, sample_indices=sample_idx)

    return compare_gradients(grad_manual, grad_fd, sample_idx)


# ==========================================================================
#  参数梯度验证 (额外: 检查 dL/d(param) 是否正确)
# ==========================================================================

def test_param_gradients_phase_twist(eps=1e-5, n_samples=10, seed=42):
    """
    验证 PhaseTwist 的参数梯度 (gamma, beta, phi)。
    通过比较参数更新前后的变化量与有限差分。
    """
    from Anla.layers.activation import PhaseTwist

    torch.manual_seed(seed)
    channels = 8
    layer = PhaseTwist(channels, init_gamma=0.05, init_beta=0.05, init_phi=0.1)
    layer.train()

    B, S = 2, 4
    z = torch.randn(B, S, channels, dtype=torch.cfloat) * 0.5 + 0.3
    G = torch.randn(B, S, channels, dtype=torch.cfloat) * 0.1

    results = {}

    for param_name in ['gamma', 'beta', 'phi']:
        param = getattr(layer, param_name)
        param_snap = param.data.clone()

        # 有限差分: 对参数的每个元素扰动
        grad_fd = torch.zeros_like(param)

        for k in range(min(n_samples, param.numel())):
            def scalar_loss_for_param(p_val):
                p_clone = param_snap.clone()
                p_clone[k] = p_val
                # 创建干净层
                lyr = PhaseTwist(channels)
                lyr.gamma.data.copy_(layer.gamma.data.clone() if param_name != 'gamma' else p_clone)
                lyr.beta.data.copy_(layer.beta.data.clone() if param_name != 'beta' else p_clone)
                lyr.phi.data.copy_(layer.phi.data.clone() if param_name != 'phi' else p_clone)
                lyr.eval()
                f_val = lyr.forward(z)
                return torch.real(torch.sum(torch.conj(G) * f_val)).item()

            # 实参数只需要实方向差分
            L_plus = scalar_loss_for_param(param_snap[k] + eps)
            L_minus = scalar_loss_for_param(param_snap[k] - eps)
            grad_fd[k] = (L_plus - L_minus) / (2.0 * eps)

        # 手动反传得到的参数梯度 (通过观察参数变化量)
        state = save_layer_state(layer)
        lr_test = 1.0  # 用 lr=1 使得 param_new = param_old - grad
        layer.forward(z)
        layer.manual_backward(G, learning_rate=lr_test)
        param_after = getattr(layer, param_name).data.clone()
        restore_layer_state(layer, state)

        # param_new = param_old - lr * grad  =>  grad = (param_old - param_new) / lr
        grad_manual_param = (param_snap - param_after) / lr_test

        # 比较
        n_cmp = min(n_samples, param.numel())
        m_vals = grad_manual_param[:n_cmp]
        f_vals = grad_fd[:n_cmp]

        cos_sim = torch.dot(m_vals, f_vals) / (m_vals.norm() * f_vals.norm() + 1e-30)
        rel_err = ((m_vals - f_vals).abs() / (f_vals.abs() + 1e-30)).mean()

        results[param_name] = {
            "cosine_similarity": cos_sim.item(),
            "mean_relative_error": rel_err.item(),
            "manual_norm": m_vals.norm().item(),
            "fd_norm": f_vals.norm().item(),
        }

    return results


# ==========================================================================
#  Softmax Jacobian 单独验证
# ==========================================================================

def test_softmax_jacobian_only(eps=1e-6, n_samples=20, seed=42):
    """
    单独验证: softmax(|z|*scale) 对 |z| 的反传是否正确。
    剥离相位部分，只看 MagPhaseSoftmax 的模长通路。
    """
    torch.manual_seed(seed)
    B, H, SQ, SK = 1, 1, 4, 4
    scale = 0.25

    mag = torch.rand(B, H, SQ, SK) * 2.0 + 0.1  # 正实数

    # forward: y = softmax(mag * scale)
    def fwd_mag(m):
        return torch.softmax(m * scale, dim=-1)

    y = fwd_mag(mag)
    G_real = torch.randn_like(mag) * 0.1

    # 手动 softmax backward
    tmp = y * G_real
    sum_tmp = tmp.sum(dim=-1, keepdim=True)
    grad_manual_mag = (tmp - y * sum_tmp) * scale

    # 有限差分
    grad_fd_mag = torch.zeros_like(mag)
    flat_mag = mag.reshape(-1)
    n = flat_mag.shape[0]
    indices = list(range(min(n_samples, n)))

    for k in indices:
        m_plus = flat_mag.clone()
        m_plus[k] += eps
        m_minus = flat_mag.clone()
        m_minus[k] -= eps

        L_plus = torch.sum(G_real * fwd_mag(m_plus.reshape(mag.shape))).item()
        L_minus = torch.sum(G_real * fwd_mag(m_minus.reshape(mag.shape))).item()
        grad_fd_mag.reshape(-1)[k] = (L_plus - L_minus) / (2.0 * eps)

    # 比较
    m_v = grad_manual_mag.reshape(-1)[:len(indices)]
    f_v = grad_fd_mag.reshape(-1)[:len(indices)]
    cos = torch.dot(m_v, f_v) / (m_v.norm() * f_v.norm() + 1e-30)

    return {
        "softmax_jacobian_cosine": cos.item(),
        "softmax_jacobian_rel_err": ((m_v - f_v).abs() / (f_v.abs() + 1e-30)).mean().item(),
    }


# ==========================================================================
#  主报告
# ==========================================================================

def format_result(name: str, result: Dict[str, Any], threshold: float = 0.95):
    """格式化单个测试结果"""
    cos = result.get("cosine_similarity", -999)
    rel = result.get("mean_relative_error", 999)

    if cos > threshold:
        verdict = "✅ PASS"
    elif cos > 0.8:
        verdict = "⚠️  WARN"
    elif cos > 0.5:
        verdict = "❌ FAIL"
    else:
        verdict = "💀 CRITICAL"

    lines = [
        f"\n{'='*70}",
        f"  {name}",
        f"{'='*70}",
        f"  Verdict:              {verdict}",
        f"  Cosine Similarity:    {cos:.6f}",
        f"  Mean Relative Error:  {rel:.6f}",
    ]

    if "max_relative_error" in result:
        lines.append(f"  Max Relative Error:   {result['max_relative_error']:.6f}")
    if "magnitude_ratio" in result:
        lines.append(f"  Magnitude Ratio:      {result['magnitude_ratio']:.6f}")
    if "manual_norm" in result:
        lines.append(f"  Manual Grad Norm:     {result['manual_norm']:.6e}")
    if "fd_norm" in result:
        lines.append(f"  FD Grad Norm:         {result['fd_norm']:.6e}")
    if "n_compared" in result:
        lines.append(f"  Points Compared:      {result['n_compared']}")

    return "\n".join(lines)


def run_all_tests(eps=1e-5, n_samples=20, seed=42, layers=None):
    """运行所有梯度检查"""

    all_tests = OrderedDict([
        ("1. MagPhaseSoftmax (HEAD SUSPECT)", test_mag_phase_softmax),
        ("2. HolographicAttention (Full)", test_holographic_attention),
        ("3. PhaseTwist (Bidirectional)", test_phase_twist),
        ("4. ComplexRMSNorm (Corrected)", test_complex_rms_norm),
        ("5. ComplexLinear", test_complex_linear),
        ("6. ComplexRotaryEmbedding", test_complex_rotary),
        ("7. ComplexTransformerBlock (E2E)", test_transformer_block),
    ])

    auxiliary_tests = OrderedDict([
        ("AUX: Softmax Jacobian Isolation", test_softmax_jacobian_only),
        ("AUX: PhaseTwist Param Gradients", test_param_gradients_phase_twist),
    ])

    if layers is not None and layers != ['all']:
        # 按名称过滤
        filtered = OrderedDict()
        for k, v in list(all_tests.items()) + list(auxiliary_tests.items()):
            for l in layers:
                if l.lower() in k.lower():
                    filtered[k] = v
                    break
        all_tests = filtered
        auxiliary_tests = OrderedDict()

    print("=" * 70)
    print("  Anla Wirtinger Gradient Checker")
    print("  有限差分 vs manual_backward 数值对比")
    print("=" * 70)
    print(f"  eps = {eps}")
    print(f"  n_samples = {n_samples}")
    print(f"  seed = {seed}")
    print(f"  dtype = torch.cfloat (complex64)")
    print()

    results = {}
    summary = []

    for name, test_fn in all_tests.items():
        print(f"\n  Running: {name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = test_fn(eps=eps, n_samples=n_samples, seed=seed)
            dt = time.time() - t0
            print(f"({dt:.1f}s)")
            print(format_result(name, result))
            results[name] = result
            cos = result.get("cosine_similarity", -999)
            summary.append((name, cos))
        except Exception as e:
            dt = time.time() - t0
            print(f"ERROR ({dt:.1f}s)")
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()
            summary.append((name, -999))

    for name, test_fn in auxiliary_tests.items():
        print(f"\n  Running: {name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = test_fn(eps=eps, n_samples=n_samples, seed=seed)
            dt = time.time() - t0
            print(f"({dt:.1f}s)")

            if isinstance(result, dict) and all(isinstance(v, dict) for v in result.values()):
                # 嵌套结果 (如 param gradients)
                for sub_name, sub_result in result.items():
                    full_name = f"{name} → {sub_name}"
                    print(format_result(full_name, sub_result))
                    cos = sub_result.get("cosine_similarity", -999)
                    summary.append((full_name, cos))
            else:
                print(format_result(name, result))
                cos = result.get("cosine_similarity",
                                 result.get("softmax_jacobian_cosine", -999))
                summary.append((name, cos))
        except Exception as e:
            dt = time.time() - t0
            print(f"ERROR ({dt:.1f}s)")
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()

    # ========== 总结 ==========
    print("\n")
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Layer':<50} {'Cosine':>8}  {'Status':>10}")
    print(f"  {'-'*50} {'-'*8}  {'-'*10}")

    n_pass = 0
    n_fail = 0
    for name, cos in summary:
        if cos > 0.95:
            status = "✅ PASS"
            n_pass += 1
        elif cos > 0.8:
            status = "⚠️  WARN"
            n_fail += 1
        elif cos > -900:
            status = "❌ FAIL"
            n_fail += 1
        else:
            status = "💥 ERROR"
            n_fail += 1
        print(f"  {name:<50} {cos:>8.4f}  {status:>10}")

    print()
    print(f"  Total: {n_pass} passed, {n_fail} failed/warned")
    print()

    if n_fail > 0:
        print("  ⚠️  DIAGNOSIS: 存在梯度不一致。")
        print("  cosine < 0.95 的层的 manual_backward 实现有数学错误。")
        print("  建议: 对 FAIL/WARN 层重新推导 Wirtinger 导数。")
    else:
        print("  ✅ 所有层通过梯度验证。")
        print("  如果拓扑仍未涌现，问题不在梯度数学，而在训练范式/架构深度。")

    return results


# ==========================================================================
#  CLI 入口
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Anla Wirtinger 有限差分梯度验证"
    )
    parser.add_argument(
        "--layer", nargs="+", default=["all"],
        help="要测试的层 (默认: all). 可选: MagPhase, Holographic, PhaseTwist, "
             "RMSNorm, Linear, Rotary, Transformer, Softmax, Param"
    )
    parser.add_argument("--eps", type=float, default=1e-5, help="有限差分步长")
    parser.add_argument("--n-samples", type=int, default=20, help="每层采样点数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    layers = args.layer if args.layer != ["all"] else None
    run_all_tests(eps=args.eps, n_samples=args.n_samples, seed=args.seed, layers=layers)


if __name__ == "__main__":
    main()
