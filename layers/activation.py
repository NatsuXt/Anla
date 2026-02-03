import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer

class PhaseTwist(ComplexLayer):
    """
    PhaseTwist V1 (Restored & 3D Supported): 线性动力学激活
    """
    
    def __init__(self, channels: int, init_gamma: float = 0.01):
        super().__init__()
        # 初始化 gamma
        self.gamma = nn.Parameter(torch.full((channels,), init_gamma, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.input_cache = z.detach().clone()
        r = torch.abs(z) + 1e-8
        theta = torch.angle(z)
        
        # 1. 幅度非线性
        r_out = torch.tanh(r)
        
        # 2. 线性相位扭曲
        twist = self.gamma.view(1, -1) * r
        theta_out = theta + twist
        
        output = torch.polar(r_out, theta_out)
        self.output_cache = output
        return output

    def manual_backward(self, grad_output: torch.Tensor, learning_rate: float, **kwargs) -> torch.Tensor:
        """
        [修正版] 支持 3D 输入 (Batch, Seq, Dim) 的梯度聚合
        """
        z = self.input_cache
        
        r = torch.abs(z) + 1e-9
        theta = torch.angle(z)
        gamma = self.gamma.view(1, -1)
        
        # 中间变量
        tanh_r = torch.tanh(r)
        sech2_r = 1 - tanh_r**2
        
        # 导数推导 (V1)
        u = theta + gamma * r
        e_iu = torch.polar(torch.ones_like(u), u)
        
        df_dr = (sech2_r + 1j * tanh_r * gamma) * e_iu
        df_dtheta = 1j * tanh_r * e_iu
        
        # 坐标变换导数
        z_norm = z / r
        e_minus_itheta = torch.conj(z_norm)
        e_itheta = z_norm
        
        dr_dz = 0.5 * e_minus_itheta
        dr_dz_conj = 0.5 * e_itheta
        dtheta_dz = -0.5j / r * e_minus_itheta
        dtheta_dz_conj = 0.5j / r * e_itheta
        
        # 组装 Wirtinger 导数
        df_dz = df_dr * dr_dz + df_dtheta * dtheta_dz
        df_dz_conj = df_dr * dr_dz_conj + df_dtheta * dtheta_dz_conj
        
        # 计算输入梯度
        grad_input = torch.conj(df_dz) * grad_output + torch.conj(df_dz_conj) * torch.conj(grad_output)
        
        # --- 3. 更新 Gamma (关键修正部分) ---
        output = self.output_cache
        df_dgamma = 1j * output * r
        
        # 计算每个样本点的梯度贡献: (Batch, Seq, Channels)
        d_gamma_per_element = torch.real(grad_output * torch.conj(df_dgamma))
        
        # [Fix]: 无论输入是 2D 还是 3D，都展平成 (Total_Samples, Channels) 然后求和
        # 这样就能正确聚合 Batch 和 Seq 维度的梯度
        total_samples = z.numel() // z.shape[-1]
        d_gamma_sum = torch.sum(d_gamma_per_element.view(-1, self.gamma.shape[0]), dim=0)
        
        d_gamma = d_gamma_sum / total_samples
        
        with torch.no_grad():
            self.gamma.data += learning_rate * d_gamma
            
        self.clear_cache()
        return grad_input
