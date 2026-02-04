import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer
from Anla.utils.complex_ops import complex_kaiming_normal_

class ComplexEmbedding(ComplexLayer):
    """
    [GPU Ready + High Viscosity]
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        weight_real = torch.empty(num_embeddings, embedding_dim)
        weight_imag = torch.empty(num_embeddings, embedding_dim)
        complex_kaiming_normal_(weight_real, weight_imag)
        
        raw_w = torch.complex(weight_real, weight_imag)
        init_w = raw_w / (raw_w.abs() + 1e-9)
        self.weight = nn.Parameter(init_w)
        
        # [FIX 1] 1e-5 -> 1e-3
        self.register_buffer('weight_energy', torch.full((num_embeddings, embedding_dim), 1e-3, dtype=torch.float32))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.input_cache = input_ids if self.training else None
        with torch.no_grad():
            return nn.functional.embedding(input_ids, self.weight)

    def manual_backward(self, grad_output: torch.Tensor, lr: float, weight_decay: float = 0.0):
        input_ids = self.input_cache
        grad_flat = grad_output.view(-1, self.embedding_dim)
        ids_flat = input_ids.view(-1)
        
        unique_ids, inverse_indices = torch.unique(ids_flat, return_inverse=True)
        grad_sum = torch.zeros(len(unique_ids), self.embedding_dim, 
                             dtype=grad_flat.dtype, device=grad_flat.device)
        grad_sum.index_add_(0, inverse_indices, grad_flat)
        
        with torch.no_grad():
            current_weights = self.weight.data[unique_ids]
            current_energy = self.weight_energy[unique_ids]
            
            curr_grad_sq = grad_sum.abs().pow(2)
            
            # [FIX 2] 0.99 -> 0.90
            beta = 0.90
            eps = 1e-5
            current_energy.mul_(beta).add_(curr_grad_sq, alpha=1-beta)
            
            denom = current_energy.sqrt().add_(eps)
            adaptive_step = grad_sum / denom * lr
            
            current_weights.sub_(adaptive_step)
            
            norm = current_weights.abs() + 1e-9
            projected_weights = current_weights / norm
            
            self.weight.data[unique_ids] = projected_weights
            self.weight_energy[unique_ids] = current_energy
            
        return None
