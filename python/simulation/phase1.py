import numpy as np 
from lattice_bridge import WeightStore

store = WeightStore()

layer1 = np.random.randn(64, 64).astype(np.float32).flatten()
layer2 = np.random.randn(64, 10).astype(np.float32).flatten()

h1 = store.insert(layer1)
h2 = store.insert(layer2)

store.set_layer_order([h1, h2])

print("Merkel root:", store.merkle_root_hex())
