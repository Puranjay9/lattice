import torch
import numpy as np
import torch.nn.functional as F
from multiprocessing import Process, Queue
from model import LatticeTransformer

class GradientPacket:
    """The unit of gossip — one training step's worth of gradients."""
    def __init__(self, node_id: int, step: int, grads: dict):
        self.node_id = node_id
        self.step = step
        self.grads = grads

def capture_gradients(model: LatticeTransformer) -> dict:
    """Extract all gradients after .backward() into a plain dict."""
    packet = {}
    for name, param in model.params.items():
        if param.grad is not None:
            packet[name] = param.grad.detach().numpy().copy()
    return packet

def apply_dp_noise(grads: dict, clip_norm=1.0, noise_mult=0.1, n_nodes=3) -> dict:
    """DP-SGD: clip then add Gaussian noise."""
    # compute global norm across all gradient tensors
    all_grads = [torch.tensor(g) for g in grads.values()]
    global_norm = torch.sqrt(sum(g.norm() ** 2 for g in all_grads))
    clip_coef = min(1.0, clip_norm / (global_norm + 1e-6))

    noisy = {}

    for name, grad in grads.items():
        g = torch.tensor(grad) * clip_coef
        noise = torch.randn_like(g) * (clip_norm * noise_mult / n_nodes)
        noisy[name] = (g + noise).numpy()

    return noisy

def fedavg(packets : list[GradientPacket]) -> dict:
    """Plain federated averaging — honest nodes only."""
    names = packets[0].grads.keys()
    averaged = {}
    for name in names:
        stacked = np.stack([p.grads[name] for p in packets])
        averaged[name] = stacked.mean(axis = 0)
    return averaged

def apply_gradients(model: LatticeTransformer, grads: dict, lr= 3e-4):
    """SGD update — apply the aggregated gradient."""
    with torch.no_grad():
        for name, param in model.params.items():
            if name in grads:
                param -= lr * torch.tensor(grads[name])

        for param in model.params.values():
            if param.grad is not None:
                param.grad.zero_()

def run_node(node_id: int, send_queues: list, recv_queue: Queue, steps = 200):
    """One federated node: train → gossip → aggregate → update."""
    torch.manual_seed(node_id * 42)
    model = LatticeTransformer()

    # tiny shakespeare: download from web or use a local file
    # for demo we use random token sequences
    
    vocab_size = 65

    print(f"[node {node_id}] starting")

    for step in range(steps):
        idx = torch.randint(0, vocab_size, (4, 32)) #(B=4, T=32)
        targets = torch.randint(0, vocab_size, (4, 32))

        logits = model.forward(idx)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )

        loss.backward()

        raw_grads = capture_gradients(model)
        noisy_grads = apply_dp_noise(raw_grads)
        packet = GradientPacket(node_id, step, noisy_grads)

        # gossip: send to all other nodes
        for q in send_queues:
            q.put(packet)

        # collect from all peers (blocking with timeout)
        peer_packets = [packet] #include own 
        for _ in range(len(send_queues)):
            try:
                peer_packets.append(recv_queue.get(timeout=2.0))
            except Exception:
                pass  # peer didn't respond in time

        aggregated = fedavg(peer_packets)
        apply_gradients(model, aggregated)

        if step % 20 == 0:
            print(f"[node {node_id}] step={step} loss={loss.item():.4f}")
    
    print(f"[node {node_id}] done")
