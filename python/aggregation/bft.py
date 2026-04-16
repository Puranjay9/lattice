import numopy as np 
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
    sizes = {n; packets[0].grads[n].size for n in names}
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

def krum(packets)
