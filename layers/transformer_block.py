import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer
from Anla.layers.attention import ComplexAttention
from Anla.layers.linear import ComplexLinear
from Anla.layers.activation import PhaseTwist
from Anla.layers.normalization import ComplexRMSNorm

class ComplexTransformerBlock(ComplexLayer):
    """
    Standard Transformer Block (Pre-Norm architecture)
    
    Structure:
    x -> Norm -> Attention -> Add(x) -> Norm -> FFN -> Add
    
    FFN: Linear -> Act -> Linear
    """
    
    def __init__(self, d_model: int, ff_dim: int = None):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * d_model
            
        # Sub-layer 1: Attention
        self.norm1 = ComplexRMSNorm(d_model)
        self.attn = ComplexAttention(d_model)
        
        # Sub-layer 2: Feed Forward
        self.norm2 = ComplexRMSNorm(d_model)
        self.ff1 = ComplexLinear(d_model, ff_dim)
        self.act = PhaseTwist(ff_dim, init_gamma=0.01) # V1 Linear Twist
        self.ff2 = ComplexLinear(ff_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_cache = x.detach().clone()
        
        # --- 1. Attention Block (Pre-Norm + Residual) ---
        # x_norm = Norm(x)
        x1 = self.norm1.forward(x)
        # attn_out = Attn(x_norm)
        attn_out = self.attn.forward(x1)
        # res1 = x + attn_out
        res1 = x + attn_out
        
        # --- 2. FFN Block (Pre-Norm + Residual) ---
        # x_norm2 = Norm(res1)
        x2 = self.norm2.forward(res1)
        # ffn_out = Linear2(Act(Linear1(x_norm2)))
        ff_h = self.ff1.forward(x2)
        ff_a = self.act.forward(ff_h)
        ff_out = self.ff2.forward(ff_a)
        # res2 = res1 + ffn_out
        output = res1 + ff_out
        
        self.block_cache = (res1, ff_out) # Cache residuals for backward
        return output

    def manual_backward(self, grad_output: torch.Tensor, learning_rate: float, weight_decay: float = 1e-4) -> torch.Tensor:
        """
        Transformer Block Backward
        关键是处理残差连接 (Add Gate) 的梯度分流。
        y = x + f(x) -> dy/dx = 1 + df/dx
        """
        x = self.input_cache
        res1, ff_out = self.block_cache
        
        # --- 2. FFN Backward ---
        # grad_output flow into (res1) and (ff_out)
        
        # Branch B: FFN Path
        grad_ff2 = self.ff2.manual_backward(grad_output, learning_rate, weight_decay)
        grad_act = self.act.manual_backward(grad_ff2, learning_rate)
        grad_ff1 = self.ff1.manual_backward(grad_act, learning_rate, weight_decay)
        grad_n2 = self.norm2.manual_backward(grad_ff1, learning_rate)
        
        # Branch A: Residual Path from Block 2
        # grad_res1 = grad_output (Skip) + grad_n2 (from FFN path)
        grad_res1 = grad_output + grad_n2
        
        # --- 1. Attention Backward ---
        # grad_res1 flows into (x) and (attn_out)
        
        # Branch D: Attn Path
        grad_attn = self.attn.manual_backward(grad_res1, learning_rate, weight_decay)
        grad_n1 = self.norm1.manual_backward(grad_attn, learning_rate)
        
        # Branch C: Residual Path from Block 1
        # grad_x = grad_res1 (Skip) + grad_n1 (from Attn path)
        grad_input = grad_res1 + grad_n1
        
        self.clear_cache()
        self.block_cache = None
        
        return grad_input
