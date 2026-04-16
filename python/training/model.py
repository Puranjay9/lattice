import torch
import torch.nn.functional as F
import numpy as np 

class LatticeTransformer:
    """
    Minimal 2-layer transformer. No nn.Module — weights are plain tensors
    so we can intercept gradients directly and package them for gossip.
    """
    
    def __init__(self, vocab_size= 65, d_model= 128, n_heads=4, n_layers=2):
        self.d_model = d_model
        self.n_heads = n_heads

        self.params = {}
        scale = d_model ** - 0.5

        self.params['embed'] = torch.randn(vocab_size, d_model) * scale

        for i in range(n_layers):
            # attention: Q, K, V projections + output
            self.params[f'l{i}_Wq'] = torch.randn(d_model, d_model) * scale
            self.params[f'l{i}_Wk'] = torch.randn(d_model, d_model) * scale
            self.params[f'l{i}_Wv'] = torch.randn(d_model, d_model) * scale
            self.params[f'l{i}_Wo'] = torch.randn(d_model, d_model) * scale
            #FFN 
            self.params[f'l{i}_W1'] = torch.randn(d_model * 4, d_model) * scale
            self.params[f'l{i}_W2'] = torch.randn(d_model, d_model * 4) * scale
            #layer norm weights 
            self.params[f'l{i}_g1'] = torch.ones(d_model)
            self.params[f'l{i}_g2'] = torch.ones(d_model)

        # output projections
        self.params['unembed'] = torch.randn(vocab_size, d_model) * scale
        
        for p in self.params.values():
            p.requires_grad_(True)

    def layer_norm(self, x, g):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return g * (x - mean) / std
    
    def attention(self, x, Wq, Wk, Wv, Wo):
        B, T, C = x.shape
        head_dim = C // self.n_heads
        Q = (x @ Wq.T).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        K = (x @ Wk.T).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        V = (x @ Wv.T).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        #casual mask
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        att = (Q @ K.transpose(-2, -1)) * (head_dim ** -0.5)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ V).transpose(1, 2).contiguous().view(B, T, C)
        return y @ Wo.T 

    def forward(self, idx):
    # idx: (B, T) token indices 
        x = self.params['embed'][idx]
        for i in range(2):
            p = self.params

            x = x + self.attention(
                self.layer_norm(x, p[f'l{i}_g1']),
                p[f'l{i}_Wq'], p[f'l{i}_Wk'],
                p[f'l{i}_Wv'], p[f'l{i}_Wo']
            )

            # FFN block with residual
            h = self.layer_norm(x, p[f'l{i}_g2'])
            h = F.gelu(h @ p[f'l{i}_W1'].T) @ p[f'l{i}_W2'].T
            x = x + h 
        
        return x @ self.params['unembed'].T 
