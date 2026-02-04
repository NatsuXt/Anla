import torch
import torch.nn as nn
import numpy as np
import sys
import os
import random
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.positional import ComplexRotaryEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock
from Anla.layers.linear import ComplexLinear

# Configuration
CONFIG = {
    'vocab_size': 64,
    'd_model': 64,
    'num_heads': 4,
    'seq_len': 32,
    'mask_token_id': 64,
    'batch_size': 16,
    'lr': 0.005, 
    'weight_decay': 1e-4,
    'epochs': 5000, 
    'mask_span_prob': 0.3,
    'max_span_length': 5,
    'save_dir': 'checkpoints'
}

class RingSpanDataGenerator:
    def __init__(self, vocab_size, seq_len, mask_id):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_id = mask_id
        
    def generate_batch(self, batch_size):
        input_ids = []
        target_ids = []
        for _ in range(batch_size):
            start = random.randint(0, self.vocab_size - 1)
            seq = [(start + i) % self.vocab_size for i in range(self.seq_len)]
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            
            inp = seq_tensor.clone()
            tgt = torch.full_like(seq_tensor, -100)
            
            mask_len = random.randint(1, CONFIG['max_span_length'])
            mask_start = random.randint(0, self.seq_len - mask_len)
            
            inp[mask_start : mask_start + mask_len] = self.mask_id
            tgt[mask_start : mask_start + mask_len] = seq_tensor[mask_start : mask_start + mask_len]
            
            input_ids.append(inp)
            target_ids.append(tgt)
        return torch.stack(input_ids), torch.stack(target_ids)

class AnlaManifoldInpainter(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = ComplexEmbedding(vocab_size + 1, d_model)
        self.rotary = ComplexRotaryEmbedding(d_model, max_seq_len=128)
        # Block 内部已去除 Causal 限制
        self.block = ComplexTransformerBlock(d_model, num_heads=CONFIG['num_heads'])
        self.head = ComplexLinear(d_model, vocab_size, bias=False) 

    def forward(self, x):
        z = self.embedding.forward(x)
        z = self.rotary.forward(z)
        z_out = self.block.forward(z, mask=None) # Native Holographic Attention
        logits = self.head.forward(z_out)
        return torch.abs(logits)

    def manual_backward(self, grad_logits, lr):
        with torch.no_grad():
            z_transformer_out = self.head.input_cache
            complex_logits = nn.functional.linear(z_transformer_out, self.head.weight, self.head.bias)
            phase = torch.angle(complex_logits)
        
        grad_complex_logits = torch.polar(grad_logits, phase)
        grad_block_out = self.head.manual_backward(grad_complex_logits, lr, CONFIG['weight_decay'])
        grad_rotary_out = self.block.manual_backward(grad_block_out, lr, CONFIG['weight_decay'])
        grad_embed_out = self.rotary.manual_backward(grad_rotary_out)
        self.embedding.manual_backward(grad_embed_out, lr, CONFIG['weight_decay'])

def save_checkpoint(model, config, filename):
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    path = os.path.join(config['save_dir'], filename)
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, path)

def train_ring_reconstruction():
    print(f"=== Anla Logic Training: Native Holographic In-painting ===")
    print(f"Configuration: {CONFIG}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = RingSpanDataGenerator(CONFIG['vocab_size'], CONFIG['seq_len'], CONFIG['mask_token_id'])
    model = AnlaManifoldInpainter(CONFIG['vocab_size'], CONFIG['d_model']).to(device)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        input_ids, target_ids = generator.generate_batch(CONFIG['batch_size'])
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        
        logits_mag = model.forward(input_ids)
        loss = criterion(logits_mag.view(-1, CONFIG['vocab_size']), target_ids.view(-1))
        
        with torch.no_grad():
            probs = torch.softmax(logits_mag, dim=-1)
            grad_logits = probs.clone()
            flat_targets = target_ids.view(-1)
            flat_grad = grad_logits.view(-1, CONFIG['vocab_size'])
            mask = flat_targets != -100
            valid_targets = flat_targets[mask]
            
            row_indices = torch.arange(flat_grad.shape[0], device=device)[mask]
            flat_grad[row_indices, valid_targets] -= 1.0
            flat_grad[~mask, :] = 0.0
            num_valid = mask.sum()
            if num_valid > 0:
                grad_logits = flat_grad.view_as(logits_mag) / num_valid
            else:
                grad_logits = torch.zeros_like(logits_mag)

        model.manual_backward(grad_logits, CONFIG['lr'])
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f}")
            
            # [RESTORED DEBUG LOGIC]
            valid_mask = target_ids[0] != -100
            if valid_mask.any():
                # Find the first mask span
                idx = torch.where(valid_mask)[0][0].item()
                
                # Context visualization
                start_idx = max(0, idx - 2)
                end_idx = min(CONFIG['seq_len'], idx + 3)
                
                context_in = input_ids[0, start_idx:end_idx].tolist()
                
                # Prediction
                true_val = target_ids[0, idx].item()
                pred_val = torch.argmax(logits_mag[0, idx]).item()
                
                status = "CORRECT" if true_val == pred_val else "FAIL"
                
                # Replace 64 with 'M' for display
                disp_in = ['M' if x == 64 else x for x in context_in]
                
                print(f"   [Debug] ...{disp_in}... -> Expect: {true_val} | Pred: {pred_val} ({status})")

            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(model, CONFIG, 'best_ring_model.pth')

    save_checkpoint(model, CONFIG, 'final_ring_model.pth')
    print("Training Complete.")

if __name__ == "__main__":
    train_ring_reconstruction()
