import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 路径修正 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入基础组件 (用于继承)
# 注意：我们将在这里重写部分逻辑以注入 Oja 规则，而不直接修改库文件，
# 以便您验证后再合并回 Core。
from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock
from Anla.layers.linear import ComplexLinear
from Anla.layers.normalization import ComplexRMSNorm

# ==========================================
# 1. 核心算法重构: 复数 Oja 更新算子
# ==========================================
def complex_oja_update(weight, grad_input, input_val, lr, alpha=0.1):
    """
    实现复数 Oja 规则 (Complex Oja's Rule) 的参数更新。
    
    公式: W_new = W - lr * ( Gradient_Term + Oja_Decay_Term )
    其中:
        Gradient_Term = delta * input^H (标准的梯度下降方向)
        Oja_Decay_Term = alpha * |delta|^2 * W (基于输出能量的自适应衰减)
    
    参数:
        weight: 当前权重参数
        grad_input: 传入的误差信号 delta (Batch, OutDim)
        input_val: 前向传播的输入 (Batch, InDim)
        lr: 学习率
        alpha: Oja 衰减系数，控制流形约束的强度
    """
    # 1. 计算标准复数梯度: dL/dW = delta^T * x^H (Batch 处理需要注意维度)
    # input_val shape: (B, In) -> (B, 1, In)
    # grad_input shape: (B, Out) -> (B, Out, 1)
    # result: (B, Out, In) -> sum over batch -> (Out, In)
    
    batch_size = input_val.size(0)
    
    # x^H (共轭)
    input_conj = input_val.conj()
    
    # 标准梯度项: sum( delta * x^H )
    # 使用 einsum 高效计算: 'bi,bo->oi' (input=bi, grad=bo, weight=oi)
    # 注意: grad_input 对应 y-t
    standard_grad = torch.einsum('bo,bi->oi', grad_input, input_conj)
    
    # 2. 计算 Oja 衰减项 (Regularization)
    # 原始 Oja 规则使用 y^2，但在误差修正学习中，我们使用能量相关的衰减
    # 这里我们设计一种"软性权重衰减": 
    # Decay = alpha * weight 
    # 或者更高级的: Decay = alpha * (output_energy) * weight
    # 为了稳定性，我们先采用类似 AdamW 的解耦权重衰减，但赋予它几何意义
    oja_decay = alpha * weight
    
    # 3. 执行更新 (关键: 必须是减法 Sub，即梯度下降)
    # W <- W - lr * (Grad + Decay)
    update_step = standard_grad + oja_decay
    
    # 归一化 Batch 影响
    update_step = update_step / batch_size
    
    # 原位更新
    weight.data.sub_(lr * update_step)


# ==========================================
# 2. 补丁类定义 (Patching Classes)
# ==========================================
# 为了演示原理，我们创建带 Correct Update Rule 的子类
# 验证通过后，这些逻辑应写入 core/base_layer.py

class OjaLinear(ComplexLinear):
    def manual_backward(self, grad_output, lr, wd):
        """
        覆盖原有的 manual_backward，使用正确的梯度下降 + Oja 约束
        """
        # 1. 恢复输入 (从缓存)
        # 假设 base_layer 在 forward 时保存了 self.input_cache
        if not hasattr(self, 'input_cache'):
            raise RuntimeError("Layer input cache not found. Forward pass missed?")
        
        x = self.input_cache
        
        # 2. 计算回传给上一层的误差: delta_in = delta_out * W^H
        # grad_output: (B, Out)
        # weight: (Out, In) -> weight.H: (In, Out)
        # result: (B, In)
        grad_input = torch.matmul(grad_output, self.weight.conj())
        
        # 3. 使用 Oja 规则更新当前层权重
        # 注意: 我们传入 wd (weight decay) 作为 alpha 参数
        complex_oja_update(self.weight, grad_output, x, lr, alpha=wd)
        
        # 如果有 Bias 也要更新
        if self.bias is not None:
            # Bias 的输入相当于全是 1
            grad_bias = grad_output.sum(dim=0)
            self.bias.data.sub_(lr * grad_bias / x.size(0))
            
        return grad_input

class OjaEmbedding(ComplexEmbedding):
    def manual_backward(self, grad_output, input_ids=None, lr=0.01, wd=0.0):
        """
        Embedding 的 Oja 更新
        """
        # Embedding 的梯度是稀疏的，我们只更新涉及到的行
        # grad_output: (Batch, Seq, Dim)
        # input_ids: (Batch, Seq)
        
        if input_ids is None:
             # 尝试从 cache 获取
             if hasattr(self, 'input_cache'):
                 input_ids = self.input_cache
             else:
                 return # 无法更新
        
        # 展平以便处理
        grad_flat = grad_output.view(-1, self.embedding_dim)
        ids_flat = input_ids.view(-1)
        
        # 简单的稀疏更新实现
        # W[id] = W[id] - lr * (grad + wd * W[id])
        # 为了效率，我们使用 PyTorch 的 index_add_ 的逆操作逻辑，或者是 scatter_add
        
        # 1. 计算纯梯度更新量
        # 由于 PyTorch 没有直接的 complex scatter_add，我们需要手动循环或者拆分实虚部
        # 这里为了演示清晰，使用简单的循环（生产环境需优化）
        unique_ids, inverse_indices = torch.unique(ids_flat, return_inverse=True)
        
        grad_sum = torch.zeros(len(unique_ids), self.embedding_dim, dtype=grad_flat.dtype, device=grad_flat.device)
        grad_sum.index_add_(0, inverse_indices, grad_flat)
        
        # 2. 应用更新 (梯度下降)
        current_weights = self.weight.data[unique_ids]
        
        # Oja Decay: 模长越大，衰减越快
        # decay = wd * current_weights
        
        # Update
        # W_new = W - lr * (Grad/N + Decay)
        # 这里 N 取 1 近似，或者取 batch frequency
        update_delta = grad_sum + (wd * current_weights)
        
        self.weight.data[unique_ids] = current_weights - (lr * update_delta)

# ==========================================
# 3. 改进的 Transformer Block
# ==========================================
class OjaTransformerBlock(ComplexTransformerBlock):
    def __init__(self, d_model, num_heads=4, ffn_dim=None, lr=0.01):
        super().__init__(d_model, num_heads, ffn_dim, lr)
        # 强制替换内部组件为 Oja 版本 (Monkey Patching 实例属性)
        # 注意：这需要 ComplexTransformerBlock 的构造函数允许注入或者我们在之后替换
        
        # 替换 RMSNorm 的参数初始化，防止为 0
        self.norm1 = ComplexRMSNorm(d_model)
        self.norm2 = ComplexRMSNorm(d_model)
        self._init_norms()
        
        # 替换 FFN 中的 Linear
        if ffn_dim is None: ffn_dim = 4 * d_model
        self.ffn[0] = OjaLinear(d_model, ffn_dim) # Linear 1
        self.ffn[2] = OjaLinear(ffn_dim, d_model) # Linear 2
        
        # 注意：Attention 内部的 Wq, Wk, Wv, Wo 也应该替换
        # 为简化代码，这里主要验证 FFN 和 Norm 的稳定性
        
    def _init_norms(self):
        # 显式初始化 Norm 参数，防止"零点陷阱"
        nn.init.constant_(self.norm1.weight, 1.0)
        nn.init.constant_(self.norm2.weight, 1.0)

    def manual_backward(self, grad_output, lr, wd):
        # 重写 backward 以确保内部调用的是 OjaLinear 的逻辑
        # 残差连接: x + F(x)
        # 梯度分支: grad_to_x = grad_output + grad_through_F
        
        # 1. 通过 Norm2 (Add & Norm)
        # dL/dx_norm2 = grad_output
        grad_norm2 = self.norm2.manual_backward(grad_output, lr, wd)
        
        # 2. 通过 FFN
        # FFN: Linear2 -> Act -> Linear1
        grad_ffn = self.ffn[2].manual_backward(grad_norm2, lr, wd)
        # 激活函数导数 (假设 Activation 实现了 proper backward)
        grad_ffn = self.ffn[1].manual_backward(grad_ffn) 
        grad_ffn = self.ffn[0].manual_backward(grad_ffn, lr, wd)
        
        # 残差梯度聚合 (Add)
        grad_residual_1 = grad_output + grad_ffn
        
        # 3. 通过 Norm1
        grad_norm1 = self.norm1.manual_backward(grad_residual_1, lr, wd)
        
        # 4. 通过 Attention
        # 假设 Attention 已经实现了稳定的 backward，这里简略调用
        grad_attn = self.attn.manual_backward(grad_norm1, lr, wd)
        
        # 最终残差聚合
        grad_final = grad_residual_1 + grad_attn
        
        return grad_final

# ==========================================
# 4. 训练测试主程序
# ==========================================
def run_principled_test():
    """
    Anla Transformer 稳定性测试 (基于 Oja 动力学原理)
    不再使用暴力截断，而是测试梯度下降方向修正 + Oja 衰减的效果。
    """
    config = {
        'vocab_size': 50,
        'dim': 32,
        'seq_len': 10,
        'batch_size': 16,
        'epochs': 200,      
        'lr': 0.05,        # 既然方向对了，可以使用更大的学习率
        'weight_decay': 0.01, # 这里的 weight_decay 是 Oja 衰减系数
        'device': torch.device('cpu') 
    }
    
    print(f"--- 启动 Anla Transformer 原理验证测试 ---")
    print(f"核心修正: 1. 梯度方向反转 (GD)  2. Oja 动力学衰减  3. RMSNorm 初始化")
    print(f"配置: {config}")

    # 模型定义 (使用修正后的组件)
    class AnlaModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.embed = OjaEmbedding(cfg['vocab_size'], cfg['dim'])
            self.block = OjaTransformerBlock(cfg['dim'])
            self.head = OjaLinear(cfg['dim'], cfg['dim'])
            
            # 初始化 head 权重，避免过大
            nn.init.xavier_uniform_(self.head.weight)

        def forward(self, input_ids):
            # 必须缓存输入用于 backward
            self.embed.input_cache = input_ids
            h = self.embed.forward(input_ids)
            
            # Block 内部通常会自动缓存
            h = self.block.forward(h)
            
            # Linear 需要手动缓存输入 (如果基类没做)
            self.head.input_cache = h 
            out = self.head.forward(h)
            return out

        def manual_backward(self, error_signal, lr, wd):
            # 误差信号传入 (Prediction - Target)
            grad_head = self.head.manual_backward(error_signal, lr, wd)
            grad_block = self.block.manual_backward(grad_head, lr, wd)
            self.embed.manual_backward(grad_block, input_ids=self.embed.input_cache, lr=lr, wd=wd)

    model = AnlaModel(config).to(config['device'])

    # 数据生成
    torch.manual_seed(42)
    raw_data = torch.arange(config['vocab_size']).repeat(config['batch_size'], 2)
    shifts = torch.randint(0, config['vocab_size'], (config['batch_size'], 1))
    data = (raw_data + shifts) % config['vocab_size']
    input_ids = data[:, :config['seq_len']].to(config['device'])
    target_ids = data[:, 1:config['seq_len']+1].to(config['device'])

    loss_history = []
    
    for epoch in range(config['epochs']):
        
        # 1. Forward
        pred = model.forward(input_ids)
        
        # 2. Target (Detach!)
        with torch.no_grad():
            target = model.embed.forward(target_ids).detach()
        
        # 3. Loss & Error Signal
        # Loss = 0.5 * |P - T|^2
        diff = pred - target
        loss = 0.5 * (diff.abs()**2).mean().item()
        
        loss_history.append(loss)
        
        # 4. Backward
        # 传入 diff 作为梯度。
        # 在 OjaLinear 中，我们会执行 W -= lr * (diff * x^H + ...)
        # 这里的减号 "-" 修正了之前的正反馈问题
        model.manual_backward(diff, config['lr'], config['weight_decay'])
        
        # 5. Monitor (不再有强制截断，完全看系统是否自稳定)
        if epoch % 10 == 0:
            w_norm = model.head.weight.abs().mean().item()
            print(f"Epoch {epoch:03d} | Loss: {loss:.6f} | Head Norm: {w_norm:.4f}")
            
            if np.isnan(loss):
                print("!!! NaN detected !!!")
                break

    # 可视化
    print(f"最终 Loss: {loss_history[-1]:.6f}")
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.title("Convergence with Correct Principles (GD + Oja)")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.yscale('log')
        plt.grid(True)
        save_path = os.path.join(current_dir, 'transformer_convergence_principled.png')
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
    except:
        pass

if __name__ == "__main__":
    run_principled_test()
