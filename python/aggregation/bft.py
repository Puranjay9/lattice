import numpy as np 
from typing import List

def stack_gradients(packets) -> np.ndarray:
    """
    Stack gradient dicts into a 2D matrix.
    Shape: (num_nodes, total_param_count)
    """
    names = list(packets[0].grads.keys())
    rows = []
    for p in packets:
        row = np.concatenate([p.grads[n].flatten() for n in names])
        rows.append(row)
    
    return np.stack(rows), names

def unstack_gradients(flat: np.ndarray, packets, names) -> dict:
    """Reverse of stack — convert flat vector back to grad dict."""
    sizes = {n: packets[0].grads[n].size for n in names}
    result = {}
    offset = 0
    for n in names:
        size = sizes[n]
        result[n] = flat[offset:offset+size].reshape(packets[0].grads[n].shape)
        offset += size

    return result

def coordinate_wise_median(packets) -> dict:
    """
    Most Byzantine-robust aggregation.
    Median is unaffected by extreme outliers in any coordinate.
    """
    matrix, names = stack_gradients(packets)
    # median across node axis — one median value per parameter
    median_flat = np.median(matrix, axis=0)
    return unstack_gradients(median_flat, packets, names)

def krum(packets, k: int = None) -> dict:
    """
    Krum aggregation — selects the most 'central' gradient.
    k = number of nearest neighbors to consider (default: n - 2)
    How it works:
    1. For each node's gradient vector, find its K nearest neighbors
    2. Sum the squared distances to those neighbors
    3. Select the gradient with the minimum sum — it's the most central
    """

    matrix, names = stack_gradients(packets)
    n = len(matrix)
    if k is None:
        k = max(1, n - 2) # byzantine-robust when k < n/2
    
    # pairwise squared distances between all gradient vectors 
    scores = np.zeros(n)
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                diff = matrix[i] = matrix[j]
                dists.append(np.dot(diff, diff))
        dists.sort()
        scores[i] = sum(dists[:k])

    winner = np.argmin(scores)
    print(f"  [krum] selected node {winner} (score={scores[winner]:.2f})")
    return unstack_gradients(matrix[winner], packets, names)

# Byzantine attack simulation
def make_byzantine_packet(honest_packet, attack_scale=-20.0):
    """
    Gradient reversal attack: send the negative of the real gradient,
    scaled up massively. This poisons FedAvg but not Krum/median.
    """
    import copy 
    bad = copy.deepcopy(honest_packet)
    bad.node_id = -1
    for name in bad.grads:
        bad.grads[name] = bad.grads[name] * attack_scale

    return bad

