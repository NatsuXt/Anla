import torch

def complex_normalize(z: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    复数归一化：保持相位不变，将模长归一化为 1。
    z_norm = z / |z|
    """
    return z / (torch.abs(z) + epsilon)

def complex_distance(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    计算两个复数向量之间的欧几里得距离。
    d = ||z1 - z2||
    """
    return torch.abs(z1 - z2)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    如果未来需要 RoPE，这里保留接口。
    但在 Anla 中，我们通常直接在 Embedding 阶段处理相位。
    """
    pass

def get_phase_difference(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    计算相对相位差。
    返回值的范围在 [-pi, pi]
    """
    return torch.angle(z1 * torch.conj(z2))
