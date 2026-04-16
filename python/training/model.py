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
        self.n_heads = 
