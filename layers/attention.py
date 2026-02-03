import torch
import torch.nn as nn
import numpy as np
from Anla.core.base_layer import ComplexLayer
from Anla.layers.linear import ComplexLinear

class ComplexAttention(ComplexLayer):
    """
    全原生复数注意力机制 (Complex-Valued Attention)
    
    核心公式:
    Score = Re(Q @ K^H) / sqrt(d_k)
    Attn = Softmax(Score)
    Output = Attn @ V
    """
    
    def __init__(self, d_model: int, d_k: int = None):
        super().__init__()
        self.d_model = d_model
        # 暂时默认 Single Head，即 d_k = d_model
        self.d_k = d_k if d_k is not None else d_model
        
        # Q, K, V Projections
        # 注意: 即使是 Linear，我们也用 ComplexLinear
        self.w_q = ComplexLinear(d_model, self.d_k, bias=False)
        self.w_k = ComplexLinear(d_model, self.d_k, bias=False)
        self.w_v = ComplexLinear(d_model, self.d_k, bias=False)
        self.w_o = ComplexLinear(self.d_k, d_model, bias=False)
        
        self.scale = 1.0 / np.sqrt(self.d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Seq_Len, Dim)
        """
        self.input_cache = x.detach().clone()
        
        # 1. Projections
        q = self.w_q.forward(x)
        k = self.w_k.forward(x)
        v = self.w_v.forward(x)
        
        # 2. Score Calculation (Hermitian Product)
        # k^H: transpose last two dims and conjugate
        k_t = k.transpose(-2, -1).conj() 
        score_complex = torch.matmul(q, k_t)
        
        # 3. Softmax on Real Part
        # 复数内积实部 = |q||k|cos(theta)，代表相位对齐程度
        score_real = score_complex.real * self.scale
        attn_weights = torch.softmax(score_real, dim=-1)
        
        # 4. Weighted Sum
        # Weights (Real) * V (Complex)
        context = torch.matmul(attn_weights.type(torch.complex64), v)
        
        # 5. Output Projection
        output = self.w_o.forward(context)
        
        # Cache for backward
        self.attn_cache = (q, k, v, score_complex, attn_weights)
        
        return output

    def manual_backward(self, grad_output: torch.Tensor, learning_rate: float, weight_decay: float = 1e-4) -> torch.Tensor:
        """
        手动反向传播。
        """
        x = self.input_cache
        q, k, v, score_complex, attn_weights = self.attn_cache
        
        # 1. Output Projection Backward
        grad_context = self.w_o.manual_backward(grad_output, learning_rate, weight_decay)
        
        # 2. V Backward
        # grad_v = weights.T @ grad_context
        grad_v = torch.matmul(attn_weights.transpose(-2, -1).type(torch.complex64), grad_context)
        
        # 3. Weights Backward (Real)
        # grad_weights = Re(grad_context @ v^H)
        grad_weights = torch.matmul(grad_context, v.transpose(-2, -1).conj()).real
        
        # 4. Softmax Backward
        # dS = s * (dg - sum(s*dg)) * scale
        sum_grad = torch.sum(grad_weights * attn_weights, dim=-1, keepdim=True)
        grad_score_real = attn_weights * (grad_weights - sum_grad)
        grad_score_real = grad_score_real * self.scale
        
        # 5. Q & K Backward
        # dL/dScore_complex = grad_score_real (Imag part is 0)
        grad_score_complex = grad_score_real.type(torch.complex64)
        
        # dL/dQ = dL/dScore @ K
        grad_q = torch.matmul(grad_score_complex, k)
        
        # dL/dK = (dL/dScore.T @ Q)^H = (Q^H @ dL/dScore)^H (Trick for Real Score)
        # Correct derivation for Hermitian form: dL/dK = (grad_score^T @ Q)
        # Verify: Score = Q @ K^H. dScore = Q @ dK^H. 
        # Tr(G^T @ Q @ dK^H) = Tr((G^T @ Q)^H @ dK)^H ...
        # Standard result: dL/dK = (G^T @ Q) for real G.
        grad_k = torch.matmul(grad_score_complex.transpose(-2, -1), q)
        
        # 6. Projections Backward
        # 注意: 这里的 input x 是共享的。
        # 我们需要分别计算三个 W 的 input gradient，然后相加。
        # w_q.manual_backward 会更新 w_q 并返回对 x 的梯度。
        
        # 重要修正: w_q, w_k, w_v 的 input_cache 可能已经被覆盖? 
        # 不，它们各自持有 x 的一份 clone (在 forward 中)。
        # 所以这里的调用是安全的。
        
        grad_x_q = self.w_q.manual_backward(grad_q, learning_rate, weight_decay)
        grad_x_k = self.w_k.manual_backward(grad_k, learning_rate, weight_decay)
        grad_x_v = self.w_v.manual_backward(grad_v, learning_rate, weight_decay)
        
        grad_x = grad_x_q + grad_x_k + grad_x_v
        
        self.clear_cache()
        self.attn_cache = None
        
        return grad_x
