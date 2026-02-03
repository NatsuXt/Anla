import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

# 导入我们构建的模块
from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.linear import ComplexLinear
from Anla.layers.activation import PhaseTwist
from Anla.layers.normalization import ComplexRMSNorm

def run_anla_test():
    # --- 0. 设备配置 ---
    # [紧急修正] 强制使用 CPU。
    # RTX 5070 (sm_120) 目前太新，PyTorch Stable 版本尚未支持其 CUDA 内核。
    # 待 PyTorch Nightly 更新支持 sm_120 后，可改回 'cuda'。
    device = torch.device('cpu') 
    print(f"Running Anla on device: {device} (Forced due to RTX 5070 compatibility)")

    # --- 1. 超参数配置 ---
    WEIGHT_DECAY = 1e-4 
    VOCAB_SIZE = 100
    DIM = 64            
    BATCH_SIZE = 32     
    SEQ_LEN = 10
    LEARNING_RATE = 0.002 # CPU 上批次较慢，稍微提高学习率以更快看到收敛
    EPOCHS = 10000         # CPU 跑 100 轮足够验证收敛性了
    
    print(f"\nInitializing Anla Core...")
    print(f"Architecture: Embedding({DIM}) -> PhaseTwist -> Linear({DIM}) -> PhaseTwist")
    
    # --- 2. 实例化层 ---
    embed = ComplexEmbedding(VOCAB_SIZE, DIM).to(device)
    
    # Layer 1
    linear1 = ComplexLinear(DIM, DIM).to(device)
    norm1 = ComplexRMSNorm(DIM).to(device) # [新增]
    act1 = PhaseTwist(DIM, init_gamma=0.01).to(device)
    
    # Layer 2
    linear2 = ComplexLinear(DIM, DIM).to(device)
    norm2 = ComplexRMSNorm(DIM).to(device) # [新增]
    act2 = PhaseTwist(DIM, init_gamma=0.01).to(device)
    
    # --- 3. 构造虚拟数据 ---
    input_ids = torch.randint(0, VOCAB_SIZE - 1, (BATCH_SIZE, SEQ_LEN)).to(device)
    target_ids = input_ids + 1
    
    loss_history = []
    gamma_history = []
    
    print(f"\nStarting Training Loop (Manual Backpropagation)...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # === Forward ===
        x0 = embed.forward(input_ids)
        x0_flat = x0.view(-1, DIM)
        
        # Block 1
        z1 = linear1.forward(x0_flat)
        n1 = norm1.forward(z1)    # Linear -> Norm
        a1 = act1.forward(n1)     # Norm -> Act
        
        # Block 2
        z2 = linear2.forward(a1)
        n2 = norm2.forward(z2)    # Linear -> Norm
        a2 = act2.forward(n2)     # Norm -> Act (Prediction)
        
        # 4. 获取 Target (Ground Truth)
        target_vecs = embed.forward(target_ids).detach().view(-1, DIM)
        
        # ================= Error Calculation =================
        delta_out = target_vecs - a2
        
        # 记录 Loss (标量)
        loss_scalar = 0.5 * torch.mean(torch.abs(delta_out)**2).item()
        loss_history.append(loss_scalar)
        
        # === Backward ===
        # 记得把 norm 层的 backward 加进去
        
        # Block 2 Backward
        grad_n2 = act2.manual_backward(delta_out, LEARNING_RATE)
        grad_z2 = norm2.manual_backward(grad_n2, LEARNING_RATE) # [新增]
        grad_a1 = linear2.manual_backward(grad_z2, LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Block 1 Backward
        grad_n1 = act1.manual_backward(grad_a1, LEARNING_RATE)
        grad_z1 = norm1.manual_backward(grad_n1, LEARNING_RATE) # [新增]
        grad_x0 = linear1.manual_backward(grad_z1, LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # ================= Monitoring =================
        # 简单的文本进度条
        if epoch % 10 == 0:
            gamma_val = act2.gamma.mean().item()
            gamma_history.append(gamma_val)
            # 计算速度
            elapsed = time.time() - start_time
            speed = (epoch + 1) / (elapsed + 1e-5)
            print(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {loss_scalar:.6f} | Gamma: {gamma_val:.5f} | Speed: {speed:.1f} it/s")

    total_time = time.time() - start_time
    print(f"\nTraining Finished in {total_time:.2f}s.")
    print(f"Final Loss: {loss_scalar:.6f}")
    
    # 可视化
    if 'DISPLAY' in os.environ or os.name == 'nt':
        try:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(loss_history, label='MSE Loss')
            plt.title("Anla Training Dynamics")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.yscale('log')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(gamma_history, color='orange', label='Twist Factor (Gamma)')
            plt.title("Phase Twist Evolution")
            plt.xlabel("Epoch (x10)")
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            print("Plot displayed successfully.")
        except Exception as e:
            print(f"Plotting failed: {e}")

if __name__ == "__main__":
    run_anla_test()
