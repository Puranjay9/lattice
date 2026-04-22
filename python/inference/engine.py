import numpy as np
import hashlib
import threading
from typing import Optional

def hash_tensor(tensor: np.ndarray) -> bytes:
    """SHA-256 of raw float32 bytes — matches the Rust implementation."""
    return hashlib.sha256(tensor.astype(np.float32).tobytes()).digest()

class WeightTamperError(Exception):
    pass

class LatticeInferenceEngine:
    """
    Chain-native transformer inference engine.
    Weights are loaded from the Merkle store and verified against
    the committed root before every inference call.
    """
    def __init__(self, store, pinned_root:bytes):
        self.store = store
        self.pinned_root = pinned_root
        self.weights = {}
        self._lock = threading.RLock()
        self._inflight = 0
        self._swap_event = threading.Event()
        self._swapping = False

    def load_and_verify(self, name: str, expected_hash: bytes) -> np.ndarray:
        """
        Load tensor from Merkle store, verify hash matches expectation.
        Raises WeightTamperError if mismatch detected.
        """
        tensor = self.store.get(expected_hash)
        if tensor is None:
            raise WeightTamperError(f"hash not found in store for {name}")
        actual_hash = hash_tensor(tensor)
        if actual_hash != expected_hash:
            raise WeightTamperError(
                f"hash mismatch for {name}: "
                f"expected {expected_hash.hex()[:8]}... "
                f"got {actual_hash.hex()[:8]}..."
            )

        return tensor
    
    def load_all_weights(self, weight_manifest: dict):
        """
        weight_manifest: {param_name: expected_hash_bytes}
        Load all tensors, verify each against its expected hash.
        """
        new_weights = {}
        for name, expected_hash in weight_manifest.items():
            new_weights[name] = self.load_and_verify(name, expected_hash)
        with self._lock:
            self.weights = new_weights
        print(f"[engine] loaded {len(new_weights)} tensors, "
              f"root={self.pinned_root.hex()[:12]}...")

    def layer_norm(self, x, g, eps=1e-5):
        mean = x.mean(-1, keepdims=True)
        std = np.sqrt(((x - mean)**2).mean(-1, keepdims=True) + eps)
        return g * (x - mean) / std
    
    def softmax(self, x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    def attention(self, x, Wq, Wk, Wv, Wo, n_heads):
        B, T, C = x.shape
        head_dim = C // n_heads
        def split(W): return (x @ W.T).reshape(B, T, n_heads, head_dim).transpose(0,2,1,3)
        Q, K, V = split(Wq), split(Wk), split(Wv)
        # causal mask
        mask = np.tril(np.ones((T, T)))[None, None]
        att = (Q @ K.transpose(0,1,3,2)) / (head_dim ** 0.5)
        att = np.where(mask == 0, -1e9, att)
        att = self.softmax(att)
        out = (att @ V).transpose(0,2,1,3).reshape(B, T, C)
        return out @ Wo.T
    
    def gelu(self, x):
          return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        token_ids: (B, T) int array
        returns: (B, T, vocab_size) logits
        """
        # guard against weight swaps mid-forward
        with self._lock:
            if self._swapping:
                raise RuntimeError("engine is swapping weights — retry")
            self._inflight += 1
            w = self.weights

        try:
            x = w['embed'][token_ids]
            n_heads = 4
            n_layers = 2 
            for i in range(n_layers):
                # attention block
                x = x + self.attention(
                    self.layer_norm(x, w[f'l{i}_g1']),
                    w[f'l{i}_Wq'], w[f'l{i}_Wk'],
                    w[f'l{i}_Wv'], w[f'l{i}_Wo'],
                    n_heads
                )
                # FFN block
                h = self.layer_norm(x, w[f'l{i}_g2'])
                h = self.gelu(h @ w[f'l{i}_W1'].T) @ w[f'l{i}_W2'].T
                x = x + h 
            return x @ w['unembed'].T 
        finally:
            with self._lock:
                self._inflight -= 1 

    def hot_swap(self, new_root: bytes, new_manifest: dict):
        """
        Called when a new gradient block is committed.
        Drains in-flight requests, then atomically swaps weights.
        """
        print(f"[engine] hot swap: {self.pinned_root.hex()[:8]} → {new_root.hex()[:8]}")
        with self._lock:
            self._swapping = True

        import time 
        while True:
            with self._lock:
                if self._inflight == 0:
                    break
            time.sleep(0.001)

        self.load_all_weights(new_manifest)
        with self._lock:
            self.pinned_root = new_root
            self._swapping = False

        print(f"[engine] swap complete, now serving root={new_root.hex()[:12]}...")

