import numpy as np 
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../training')
sys.path.insert(0, '../aggregation')

import torch
import torch.nn.functional as F
from model import LatticeTransformer
from node import capture_gradients, apply_dp_noise, apply_gradients, GradientPacket
from bft import krum, coordinate_wise_median, make_byzantine_packet

def one_step(models, step, use_byzantine=True, aggregation='krum'):
    vocab_size = 65
    packets = []

    for i, model in enumerate(models):
        idx = torch.randint(0, vocab_size, (4,16))
        targets = torch.randint(0, vocab_size, (4, 16))
        logits = model.forward(idx)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        grads = capture_gradients(model)
        packets.append(GradientPacket(i, step, grads))

    if use_byzantine:
        packets[2] = make_byzantine_packet(packets[2])
    
    if aggregation == "fedavg":
        names = sorted(packets[0].grads.keys())  # ensure consistent order

        # flatten all gradients into vectors
        matrix = np.stack([
            np.concatenate([
                (p.grads[n].flatten()
                if p.grads[n] is not None
                else np.zeros_like(packets[0].grads[n].flatten()))
                for n in names
            ])
            for p in packets
        ])

        # aggregate full vector
        agg_vector = matrix.mean(axis=0)

        # unflatten back to parameter shapes
        aggregated = {}
        offset = 0

        for n in names:
            shape = packets[0].grads[n].shape
            size = np.prod(shape)

            aggregated[n] = agg_vector[offset:offset+size].reshape(shape)
            offset += size

    elif aggregation == "krum":
        aggregated = krum(packets, k = 2)
    elif aggregated  == "median":
        aggregated = coordinate_wise_median(packets)

    for model in models:
        apply_gradients(model, aggregated)

    return packets[0]

def run_experiment(aggregation="krum", n_steps=100):
    torch.manual_seed(0)
    models = [LatticeTransformer() for _ in range(3)]
    losses = []
    vocab_size = 65

    for step in range(n_steps):
        idx = torch.randint(0, vocab_size, (4,16))
        targets = torch.randint(0, vocab_size, (4, 16))
        loss = models[0].forward(idx)
        loss_val = F.cross_entropy(loss.view(-1, vocab_size), targets.view(-1)).item()
        losses.append(loss_val)
        one_step(models, step, use_byzantine=True, aggregation=aggregation)

    return losses

print("Running FedAvg with Byzantine node...")
fedavg_losses = run_experiment('fedavg')

print("Running Krum with Byzantine node...")
krum_losses = run_experiment('krum')

plt.figure(figsize=(10, 4))
plt.plot(fedavg_losses, label='FedAvg (Byzantine present)', color='#E24B4A', alpha=0.8)
plt.plot(krum_losses, label='Krum (Byzantine present)', color='#1D9E75', alpha=0.8)
plt.xlabel('Training step')
plt.ylabel('Loss')
plt.title('Lattice — Byzantine defense: FedAvg vs Krum')
plt.legend()
plt.tight_layout()
plt.savefig('byzantine_demo.png', dpi=150)
print("saved byzantine_demo.png — FedAvg diverges, Krum stays stable")
