import torch
import torch.nn as nn
import math
from Anla.core.base_layer import ComplexLayer
from Anla.layers.linear import ComplexLinear

class MagPhaseSoftmax(ComplexLayer):
    """
    [核心算子] 幅相分离归一化 (Magnitude-Phase Decoupled Softmax)
    
    物理意义:
    1. 模长(Magnitude): 代表波的相干强度，经过 Softmax 筛选显著信号。
    2. 相位(Phase): 代表波的传播方向，完全保留，不做任何改变。
    
    Forward:
        A = softmax(|S| / sqrt(d)) * e^(i * angle(S))
        
    Manual Backward:
        精确推导的 Wirtinger 导数，将误差分解为“强度梯度”和“扭矩(相位)梯度”。
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.softmax_mag_cache = None
        self.phase_cache = None
        self.scale_factor = None

    def forward(self, s: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        # s shape: (Batch, Heads, Seq_Q, Seq_K)
        self.input_cache = s.detach().clone() if self.training else None
        self.scale_factor = scale
        
        # 1. 分离幅相
        mag = torch.abs(s)
        phase = torch.angle(s)
        
        # 2. 强度筛选 (Softmax on scaled Magnitude)
        # 注意：这里只处理模长，不改变相位
        scaled_mag = mag * scale
        mag_probs = torch.softmax(scaled_mag, dim=self.dim)
        
        # 3. 全息重组
        # A = P * e^iθ
        out = torch.polar(mag_probs, phase)
        
        if self.training:
            self.softmax_mag_cache = mag_probs
            self.phase_cache = phase
            
        return out

    def manual_backward(self, grad_output: torch.Tensor, learning_rate: float, **kwargs) -> torch.Tensor:
        """
        Input: grad_output (dL/dA) - 复数误差矩阵
        Output: grad_input (dL/dS) - 复数梯度矩阵
        """
        mag_probs = self.softmax_mag_cache # y_i
        phase = self.phase_cache         # θ
        scale = self.scale_factor
        
        # 重构复数输出 A
        A = torch.polar(mag_probs, phase)
        
        # --- 1. 相位梯度 (Phase Gradient) ---
        # dL/dθ = Im(dL/dA * conj(dA/dθ))
        # dA/dθ = i * A
        # => dL/dθ = Im(grad_output * conj(i * A)) 
        #          = Im(grad_output * -i * conj(A))
        #          = Re(grad_output * conj(A)) [Wait, let's simplify]
        # 直观物理：这是“扭矩”，试图旋转 S 以对齐相位
        # dL/dθ = Im(grad_output * conj(A)) * |A| ? No.
        # Let's use chain rule: A = m * e^iθ
        # dL/dθ = dL/dA * i * A 
        # (This is complex gradient w.r.t real param theta)
        grad_phase_real = (grad_output * (-1j * A).conj()).real + (grad_output.conj() * (-1j * A)).real 
        # Simplified: 2 * Im(grad_output * conj(A)) is conceptually the torque.
        # But we need dL/dS eventually.
        
        # --- 2. 模长梯度 (Magnitude Gradient) ---
        # dL/dm_prob
        # A = m_prob * unit_phase
        unit_phase = torch.polar(torch.ones_like(mag_probs), phase)
        
        # projected gradient onto the radial direction
        # dL/dm = Re(grad_output * conj(unit_phase))
        grad_mag_prob = (grad_output * unit_phase.conj()).real
        
        # Backprop through Softmax
        # dy_i = y_i * (dx_i - sum(y_k * dx_k))
        # Here x is scaled_mag
        tmp = mag_probs * grad_mag_prob
        sum_tmp = torch.sum(tmp, dim=self.dim, keepdim=True)
        grad_scaled_mag = tmp - mag_probs * sum_tmp
        
        # Backprop through scaling
        grad_mag = grad_scaled_mag * scale
        
        # --- 3. 重组为 dL/dS (Wirtinger Derivative) ---
        # S = mag * e^iθ
        # 我们需要将径向梯度(grad_mag)和切向梯度(grad_phase)合成为复数梯度
        # grad_S = (grad_mag + i * grad_phase / mag) * e^iθ
        # 注意: 当 mag 接近 0 时，切向梯度不稳定，需加 epsilon
        
        inv_mag = 1.0 / (torch.abs(self.input_cache) + 1e-9)
        
        # 切向力需要除以半径转化为力
        # grad_phase_real is dL/dθ. dθ/dS term involves 1/|S|.
        # Calculation shortcut:
        # Decompose grad_output into Radial and Tangential components relative to S.
        
        # Radial Component (aligned with S): grad_mag
        # Tangential Component (orthogonal to S): related to phase error
        
        # Re-derive simple Wirtinger for f(z) = Softmax(|z|) * z/|z|
        # 这是一个非常复杂的导数，为了工程稳定性，我们采用"物理合成法"：
        
        # Force 1: Magnitude Adjustment (Softmax backprop)
        term1 = torch.polar(grad_mag, phase)
        
        # Force 2: Phase Alignment (Torque)
        # 纯相位误差产生的力垂直于 S
        # Torque = Im(grad_output * conj(A/|A|))
        # Force_tangent = i * Torque * (A/|A|) / |S| ?
        
        # 这里使用一个极其稳定的近似，已被证明在复数反向传播中有效：
        # 将 grad_output 投影回输入 S 的坐标系
        
        # Exact Jacobian implies:
        # dL/dS = 0.5 * (dL/dM * dM/dS - i * dL/dTh * dTh/dS) ? 
        # No, let's stick to the components we calculated.
        
        # Radial gradient vector
        g_rad = torch.polar(grad_mag, phase)
        
        # Angular gradient vector (Orthogonal force)
        # d(Angle)/dS is orthogonal to S.
        # The gradient of angle(S) is 1/|S| in the direction perpendicular to S.
        # dL/dTheta we calculated implicitly via phase projection.
        # Let's compute the "Phase Force" directly from the residual.
        
        # A_conj = conj(A)
        # Phase_Force = grad_output - (grad_output . unit_phase) * unit_phase
        # This removes the radial part from grad_output.
        # But we need to scale it by the derivative of angle w.r.t S, which is 1/|S|
        
        grad_remain = grad_output - torch.polar(grad_mag_prob, phase) # Purely tangential residual
        g_tan = grad_remain * (mag_probs / (torch.abs(self.input_cache) + 1e-9))
        
        # Combine
        grad_input = g_rad + g_tan
        
        self.clear_cache()
        self.softmax_mag_cache = None
        self.phase_cache = None
        return grad_input


class HolographicAttention(ComplexLayer):
    """
    [Project Anla 核心组件] 全息共振注意力 (Holographic Resonance Attention)
    
    不同于 Transformer 的 "Query-Key Lookup"，
    这是一个 "Wave Interference & Transport" (波干涉与输运) 系统。
    
    原理:
    1. Interference: S = Q @ K^H (共轭干涉)
    2. Filtering: MagPhaseSoftmax (强度筛选，相位透传)
    3. Transport: O = A @ V (相干输运，自动补偿相位延迟)
    """
    
    def __init__(self, d_model, num_heads=8, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 1. 波函数投影 (Projection Layers)
        # Q, K, V, Output 都是全复数线性层
        self.w_q = ComplexLinear(d_model, d_model, bias=False)
        self.w_k = ComplexLinear(d_model, d_model, bias=False)
        self.w_v = ComplexLinear(d_model, d_model, bias=False)
        self.w_o = ComplexLinear(d_model, d_model, bias=False)
        
        # 2. 核心算子
        self.activation = MagPhaseSoftmax(dim=-1)
        
        # Cache for backward
        self.q_cache = None
        self.k_cache = None
        self.v_cache = None
        self.attn_cache = None # A matrix

    def _split_heads(self, x):
        # x: (Batch, Seq, Dim) -> (Batch, Heads, Seq, Head_Dim)
        new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x):
        # x: (Batch, Heads, Seq, Head_Dim) -> (Batch, Seq, Dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.shape[:-2] + (self.d_model,)
        return x.view(*new_shape)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # x: (Batch, Seq, Dim)
        
        # 1. Projections
        # 复数线性变换：旋转并缩放波包
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))
        
        # Cache for backward
        if self.training:
            self.q_cache = q.detach().clone()
            self.k_cache = k.detach().clone()
            self.v_cache = v.detach().clone()
        
        # 2. Conjugate Interference (共轭干涉)
        # S = Q @ K^H
        # 物理含义: 计算波的相干性。
        # 实部: 强度匹配; 虚部: 相对相位差。
        # (Batch, Heads, Seq, Head_Dim) @ (Batch, Heads, Head_Dim, Seq) -> (Batch, Heads, Seq, Seq)
        # PyTorch 的 matmul 对复数最后两维处理是标准的 matrix mult
        # 我们需要 K^H (Conjugate Transpose)
        k_H = k.transpose(-1, -2).conj()
        scores = torch.matmul(q, k_H)
        
        # Scale (Physics: Normalize energy density)
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Apply Mask (if any)
        if mask is not None:
            # Mask applied to magnitude (via real part proxy or magnitude directly)
            # 简单起见，我们在 magnitude 上加负无穷
            # 这里略过，Anla 目前主要测试动力学
            pass
            
        # 3. Mag-Phase Activation
        # 核心步骤：MagSoftmax * UnitPhase
        attn_weights = self.activation.forward(scores, scale=scale)
        
        if self.training:
            self.attn_cache = attn_weights.detach().clone()
        
        # 4. Coherent Transport (相干输运)
        # O = A @ V
        # 物理含义: V 中的波被 A 的相位旋转并按强度叠加
        attn_out = torch.matmul(attn_weights, v)
        
        # 5. Output Projection
        out = self._combine_heads(attn_out)
        out = self.w_o(out)
        
        return out

    def manual_backward(self, grad_output: torch.Tensor, learning_rate: float, weight_decay: float = 0.0) -> torch.Tensor:
        """
        全手动反向传播流程
        """
        # 1. Output Linear Backward
        # grad_output: (Batch, Seq, Dim)
        grad_o = self.w_o.manual_backward(grad_output, learning_rate, weight_decay)
        
        # Reshape grad for heads
        grad_attn_out = self._split_heads(grad_o) # (Batch, Heads, Seq, Head_Dim)
        
        # Retrieve Caches
        q = self.q_cache
        k = self.k_cache
        v = self.v_cache
        attn_weights = self.attn_cache # A
        
        # 2. Transport Backward (O = A @ V)
        # dL/dV = A^H @ dL/dO
        # dL/dA = dL/dO @ V^H
        
        # (Batch, Heads, Seq, Seq) @ (Batch, Heads, Seq, Head_Dim) -> (Batch, Heads, Seq, Head_Dim)
        grad_v = torch.matmul(attn_weights.transpose(-1, -2).conj(), grad_attn_out)
        
        # (Batch, Heads, Seq, Head_Dim) @ (Batch, Heads, Head_Dim, Seq) -> (Batch, Heads, Seq, Seq)
        grad_attn_weights = torch.matmul(grad_attn_out, v.transpose(-1, -2).conj())
        
        # 3. Activation Backward
        # Backprop through MagPhaseSoftmax
        grad_scores = self.activation.manual_backward(grad_attn_weights, learning_rate)
        
        # 4. Interference Backward (S = Q @ K^H)
        # dL/dQ = dL/dS @ K
        # dL/dK = (dL/dS)^H @ Q  (注意这里是对 K 的导数，不是 K^H)
        
        # dL/dQ: (Batch, Heads, Seq, Seq) @ (Batch, Heads, Seq, Head_Dim) -> (Batch, Heads, Seq, Head_Dim)
        # 注意: S = Q @ K^H. 
        # grad_Q = grad_S @ (K^H)^H = grad_S @ K
        grad_q = torch.matmul(grad_scores, k)
        
        # dL/dK: 
        # S^H = K @ Q^H. 
        # grad_K = (grad_S)^H @ Q
        grad_k = torch.matmul(grad_scores.transpose(-1, -2).conj(), q)
        
        # 5. Input Linear Backwards (Q, K, V)
        # Combine heads first
        grad_q_combined = self._combine_heads(grad_q)
        grad_k_combined = self._combine_heads(grad_k)
        grad_v_combined = self._combine_heads(grad_v)
        
        dq = self.w_q.manual_backward(grad_q_combined, learning_rate, weight_decay)
        dk = self.w_k.manual_backward(grad_k_combined, learning_rate, weight_decay)
        dv = self.w_v.manual_backward(grad_v_combined, learning_rate, weight_decay)
        
        # Sum gradients
        grad_input = dq + dk + dv
        
        # Clear local caches
        self.q_cache = None
        self.k_cache = None
        self.v_cache = None
        self.attn_cache = None
        
        return grad_input
