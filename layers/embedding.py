import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer
from Anla.utils.complex_ops import complex_kaiming_normal_

class ComplexEmbedding(ComplexLayer):
    """
    [Anla AGI Core] Manifold Embedding Layer
    支持：
    1. 融合自适应优化器 (Fused Adaptive Optimizer)
    2. 全局能量稳态 (Global Energy Homeostasis)
    3. 双向纠缠接口 (Bidirectional Entanglement Interface)
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 1. 复数流形初始化
        weight_real = torch.empty(num_embeddings, embedding_dim)
        weight_imag = torch.empty(num_embeddings, embedding_dim)
        complex_kaiming_normal_(weight_real, weight_imag)
        
        # 初始化归一化：保证起点在单位球体附近
        raw_w = torch.complex(weight_real, weight_imag)
        init_w = raw_w / (raw_w.abs() + 1e-9)
        self.weight = nn.Parameter(init_w)
        
        # 2. 能量缓冲 (用于自适应步长)
        # 初始给予较高的阻尼 (1e-3)，防止冷启动时相位剧烈震荡
        self.register_buffer('weight_energy', torch.full((num_embeddings, embedding_dim), 1e-3, dtype=torch.float32))
        
        self.input_cache = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 缓存 Input IDs，供默认反向传播路径使用
        self.input_cache = input_ids if self.training else None
        with torch.no_grad():
            return nn.functional.embedding(input_ids, self.weight)

    def manual_backward(self, grad_output: torch.Tensor, lr: float, weight_decay: float = 0.0):
        """
        Path A: 针对 forward 输入的默认反向传播
        """
        if self.input_cache is None:
            return
        
        self._apply_update(grad_output, self.input_cache, lr, weight_decay)
        self.input_cache = None # Clear cache

    def manual_backward_explicit(self, grad: torch.Tensor, indices: torch.Tensor, lr: float, weight_decay: float = 0.0):
        """
        Path B: [双向纠缠接口]
        允许外部直接对指定的 Embedding (如 Target IDs) 注入梯度。
        """
        self._apply_update(grad, indices, lr, weight_decay)

    def _apply_update(self, grad: torch.Tensor, indices: torch.Tensor, lr: float, weight_decay: float):
        """
        核心动力学更新逻辑
        包含: 梯度聚合 -> 自适应下降 -> 全局能量调节
        """
        # 1. 维度展平
        grad_flat = grad.reshape(-1, self.embedding_dim)
        ids_flat = indices.reshape(-1)
        
        # 2. 梯度聚合 (处理同一个 Batch 中重复的词)
        unique_ids, inverse_indices = torch.unique(ids_flat, return_inverse=True)
        
        grad_sum = torch.zeros(len(unique_ids), self.embedding_dim, 
                             dtype=grad_flat.dtype, device=grad_flat.device)
        grad_sum.index_add_(0, inverse_indices, grad_flat)
        
        with torch.no_grad():
            # 读取当前状态
            current_weights = self.weight.data[unique_ids]
            current_energy = self.weight_energy[unique_ids]
            
            # --- Step 1: 融合优化器 (Fused Optimizer) ---
            curr_grad_sq = grad_sum.abs().pow(2)
            
            # 动量参数
            beta = 0.90
            eps = 1e-5
            
            # 更新能量缓冲 (RMSProp style)
            current_energy.mul_(beta).add_(curr_grad_sq, alpha=1-beta)
            denom = current_energy.sqrt().add_(eps)
            
            # 计算自适应更新步长
            # Update = lr * (grad / sqrt(energy))
            adaptive_step = grad_sum / denom * lr
            
            # 施加 Weight Decay (重力)
            if weight_decay > 0:
                current_weights.mul_(1.0 - weight_decay)
            
            # 执行梯度下降
            # z_new = z_old - step
            current_weights.sub_(adaptive_step)
            
            # --- Step 2: 全局能量稳态 (Global Energy Homeostasis) ---
            # [Manifold Constraint Update]
            # 不再使用截断，而是计算当前这批活跃向量的 RMS 能量，进行集体缩放
            
            # 计算 RMS: sqrt(mean(|z|^2))
            # 形状: (Batch_Unique_Size, ) -> Scalar
            moduli_sq = current_weights.abs().pow(2)
            batch_rms_energy = torch.sqrt(moduli_sq.mean() + 1e-9)
            
            # 目标能量设定为 1.0 (单位球体)
            target_energy = 1.0
            
            # 计算调节因子 g
            # g = 1 / RMS
            scaling_factor = target_energy / (batch_rms_energy + 1e-9)
            
            # 应用集体调节
            # "模小集体调大，模大集体调小"，保持相对比例不变
            current_weights.mul_(scaling_factor)
            
            # --- Step 3: 状态回写 ---
            self.weight.data.index_put_((unique_ids,), current_weights)
            self.weight_energy.index_put_((unique_ids,), current_energy)

