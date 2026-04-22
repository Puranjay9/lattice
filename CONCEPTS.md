# Lattice — Complete Concepts Reference

> Every concept taught in this project: Rust systems programming, peer-to-peer networking,
> and PyTorch deep learning — with diagrams and the math behind it all.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Rust Concepts](#2-rust-concepts)
3. [Networking Concepts](#3-networking-concepts)
4. [PyTorch Methods & the Math Behind Them](#4-pytorch-methods--the-math-behind-them)
5. [Federated Learning & Byzantine Fault Tolerance](#5-federated-learning--byzantine-fault-tolerance)
6. [Cryptographic Integrity (Merkle Trees)](#6-cryptographic-integrity-merkle-trees)
7. [Differential Privacy (DP-SGD)](#7-differential-privacy-dp-sgd)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       Lattice Node                          │
│                                                             │
│   ┌──────────────┐   ┌────────────┐   ┌──────────────────┐  │
│   │ Transformer  │──▶│  Gradients │──▶│  DP Noise + Clip │  │
│   │   Model      │   │  .backward │   │  (privacy)       │  │
│   └──────────────┘   └────────────┘   └────────┬─────────┘  │
│         ▲                                      │            │
│         │                                      ▼            │
│   ┌─────┴────────┐   ┌────────────┐   ┌──────────────────┐  │
│   │   Apply      │◀──│   Krum     │◀──│  TCP Broadcast   │  │
│   │   Gradient   │   │   BFT      │   │  (gossip)        │  │
│   └─────┬────────┘   └────────────┘   └──────────────────┘  │
│         │                                                   │
│         ▼                                                   │
│   ┌────────────────────────────────────────────────────┐    │
│   │  Merkle Root  →  Block Commit  →  /root endpoint   │    │
│   └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         ▲                ▲               ▲
         │    TCP P2P     │               │
         ▼                ▼               ▼
    [ Node 0 ]       [ Node 1 ]      [ Node 2 ]
```

**Data flow per training round:**

```
 1. Forward pass   ──▶  compute logits
 2. Loss           ──▶  cross-entropy against targets
 3. Backward pass  ──▶  ∂Loss/∂θ for every parameter
 4. DP-SGD         ──▶  clip + add Gaussian noise
 5. Broadcast      ──▶  TCP send to all peers
 6. Collect        ──▶  wait for peer gradients (round barrier)
 7. Krum           ──▶  BFT-robust aggregation
 8. Apply          ──▶  θ ← θ − lr · g_aggregated
 9. Merkle root    ──▶  SHA-256 tree over new weights
10. Block commit   ──▶  increment block height, store root
```

---

## 2. Rust Concepts

The project contains **5 Rust crates** organized as a Cargo workspace.

### 2.1 Cargo Workspace (`Cargo.toml`)

```toml
[workspace]
members = [
    "rust/merkle-store",   # content-addressed weight storage
    "rust/chain-state",    # gradient block chain
    "rust/consensus",      # consensus protocol (stub)
    "rust/p2p-net",        # libp2p networking
    "rust/lattice-bridge", # PyO3 FFI bridge
]
```

**Concept:** A workspace lets multiple crates share a single `Cargo.lock` and `target/` directory. Each crate compiles independently but can depend on siblings via `path = "../sibling"`.

---

### 2.2 Ownership & Borrowing

```
┌──────────────────────────────────────────────────────┐
│              Rust Ownership Rules                     │
│                                                      │
│  1. Every value has exactly ONE owner                │
│  2. When the owner goes out of scope, value is freed │
│  3. You can have EITHER:                             │
│     • One mutable reference   (&mut T)               │
│     • Multiple shared references  (&T)               │
│     … but NOT both at the same time                  │
└──────────────────────────────────────────────────────┘
```

**Where it appears in Lattice:**

| File | Example | Concept |
|------|---------|---------|
| `merkle-store/lib.rs` | `fn get(&self, hash: &[u8; 32]) -> Option<&[f32]>` | Shared borrow — returns a reference into `self.data` |
| `merkle-store/lib.rs` | `fn insert(&mut self, tensor: &[f32]) -> [u8; 32]` | Exclusive borrow — mutates internal `HashMap` |
| `chain-state/lib.rs` | `fn tip(&self) -> &GradientBlock` | Returns a borrow of the last block — no clone needed |
| `chain-state/lib.rs` | `fn append(&mut self, block: GradientBlock)` | Takes ownership of `block` by value — moves it in |

---

### 2.3 Structs, Enums & Derive Macros

```rust
// Struct — named product type
#[derive(Serialize, Deserialize, Clone, Debug)]    // ← derive macros
pub struct GradientBlock {
    pub height: u64,
    pub prev_root: [u8; 32],
    pub new_root: [u8; 32],
    pub gradient_delta: HashMap<String, Vec<f32>>,
    pub timestamp_ms: u64,
    pub proposer_id: String,
}

// Enum — sum type (tagged union)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum LatticeMessage {
    GradientShare { node_id: String, step: u64, gradient_bytes: Vec<u8> },
    BlockProposal  { block_bytes: Vec<u8> },
    BlockVote      { block_hash: [u8; 32], accept: bool, voter_id: String },
}
```

| Derive | Purpose |
|--------|---------|
| `Serialize` / `Deserialize` | serde — converts struct ↔ bytes (bincode / JSON) |
| `Clone` | `x.clone()` produces a deep copy |
| `Debug` | `{:?}` formatting for print debugging |
| `Default` | `WeightStore::default()` creates empty instance |

---

### 2.4 Error Handling — `Result<T, E>` and `thiserror`

```rust
#[derive(Debug, thiserror::Error)]
pub enum BlockError {
    #[error("height mismatch: expected {expected}, got {got}")]
    HeightMismatch { expected: u64, got: u64 },

    #[error("prev_root mismatch")]
    PrevRootMismatch,

    #[error("timestamp not after previous block")]
    TimestampNotMonotonic,
}

// Usage with Result:
pub fn validate(&self, block: &GradientBlock) -> Result<(), BlockError> {
    if block.height != tip.height + 1 {
        return Err(BlockError::HeightMismatch { ... });
    }
    Ok(())
}
```

```
┌─────────────────────────────────────────────┐
│          Rust Error Handling                 │
│                                             │
│   Result<T, E>  =  Ok(value) | Err(error)  │
│                                             │
│   ? operator:  propagates Err up the stack  │
│   .unwrap():   panics on Err (test-only!)   │
│   match:       explicit pattern match       │
└─────────────────────────────────────────────┘
```

---

### 2.5 HashMap & Content-Addressed Storage

```
┌──────────────── WeightStore ─────────────────┐
│                                              │
│   data: HashMap<[u8; 32], Vec<u8>>           │
│                                              │
│   Key   = SHA-256(tensor bytes)              │
│   Value = raw f32 bytes of the tensor        │
│                                              │
│   ┌──────────┐    insert(tensor)             │
│   │ [f32; N] │ ─────────────────▶ hash       │
│   └──────────┘                    │          │
│                                   ▼          │
│              data[hash] = bytes(tensor)      │
│                                              │
│   get(hash)  ─▶  data[hash] → &[f32]        │
└──────────────────────────────────────────────┘
```

**Concept:** Content-addressed storage means the key IS the hash of the value. If any bit of the tensor changes, the key changes — automatic tamper detection.

---

### 2.6 `bytemuck` — Zero-Copy Casts

```rust
use bytemuck::cast_slice;

// f32 slice → u8 slice (zero-copy, no allocation)
let bytes: &[u8] = bytemuck::cast_slice(tensor);

// u8 slice → f32 slice (zero-copy back)
let floats: &[f32] = bytemuck::cast_slice(bytes);
```

**Why:** Avoids copying data when hashing or storing tensors. The in-memory representation of `[f32]` **is** the `[u8]` bytes — `bytemuck` just reinterprets the pointer.

---

### 2.7 Traits & `impl` Blocks

```
┌────────────────────────────────────────────┐
│              Trait System                   │
│                                            │
│  trait Default {                            │
│      fn default() -> Self;                  │
│  }                                          │
│                                            │
│  #[derive(Default)]                        │
│  struct WeightStore { ... }                │
│                                            │
│  // Manual impl:                           │
│  impl WeightStore {                         │
│      pub fn new() -> Self {                 │
│          Self::default()                    │
│      }                                      │
│  }                                          │
└────────────────────────────────────────────┘
```

**Used in Lattice:**
- `Default` on `WeightStore` for empty initialization
- `NetworkBehaviour` trait (libp2p) on `LatticeBehaviour`
- `Serialize` / `Deserialize` traits for wire format

---

### 2.8 Generics & Lifetimes

```rust
// Lifetime annotations in lattice-bridge:
fn get<'py>(&self, py: Python<'py>, hash: &[u8])
    -> Option<Bound<'py, PyArray1<f32>>>
//     ^^^                  ^^^
//     The returned PyArray lives as long as the Python GIL ('py)
```

```
┌────────────────────────────────────────────┐
│              Lifetimes                      │
│                                            │
│  'a means "lives at least as long as 'a"   │
│                                            │
│  fn foo<'a>(x: &'a str) -> &'a str         │
│  // return value lives as long as input     │
│                                            │
│  'py in PyO3 = lifetime of Python GIL lock │
└────────────────────────────────────────────┘
```

---

### 2.9 PyO3 — Rust ↔ Python FFI Bridge

```
┌─────────── lattice-bridge crate ──────────────┐
│                                                │
│  Rust (merkle-store)         Python            │
│  ┌──────────────┐           ┌──────────┐       │
│  │ WeightStore  │◄── PyO3 ──│ import   │       │
│  │  .insert()   │           │ lattice_ │       │
│  │  .get()      │           │ bridge   │       │
│  │  .merkle_root│           └──────────┘       │
│  └──────────────┘                              │
│                                                │
│  #[pyclass]    → exposes struct to Python       │
│  #[pymethods]  → exposes methods                │
│  #[pymodule]   → entry point for `import`       │
│  crate-type = ["cdylib"] → shared library       │
└────────────────────────────────────────────────┘
```

Key annotations:

| Attribute | Purpose |
|-----------|---------|
| `#[pyclass]` | Make `WeightStore` importable from Python |
| `#[pymethods]` | Expose `insert`, `get`, `merkle_root`, etc. |
| `#[new]` | Maps to Python's `__init__` |
| `#[pymodule]` | Module initialization (`import lattice_bridge`) |
| `PyReadonlyArray1<f32>` | NumPy array passed from Python (zero-copy read) |
| `PyArray1::from_slice_bound` | Create NumPy array from Rust slice |

---

### 2.10 Async Rust — `tokio` & `tokio::select!`

```
┌──────────────── p2p-net crate ───────────────┐
│                                              │
│  Runtime: Tokio (multi-threaded async)        │
│                                              │
│  tokio::select! {                             │
│      // Branch 1: outgoing message to publish │
│      Some((topic, msg)) = rx.recv() => {      │
│          swarm.gossipsub.publish(topic, msg);  │
│      }                                        │
│                                              │
│      // Branch 2: incoming network event      │
│      event = swarm.next() => {               │
│          match event { ... }                  │
│      }                                        │
│  }                                            │
│                                              │
│  Both branches run concurrently.              │
│  Whichever fires first gets handled.          │
└──────────────────────────────────────────────┘
```

**Concept:**
- `async fn` suspends execution at `.await` points
- `tokio::select!` multiplexes multiple async branches
- `mpsc::Sender` / `mpsc::Receiver` — async message channels between tasks

---

### 2.11 Pattern Matching & `match`

```rust
match event {
    Some(SwarmEvent::Behaviour(LatticeBehaviourEvent::Gossipsub(
        gossipsub::Event::Message { message, .. }
    ))) => {
        // deeply nested destructure in one pattern
    }
    Some(SwarmEvent::Behaviour(LatticeBehaviourEvent::Mdns(
        mdns::Event::Discovered(peers)
    ))) => {
        for (peer_id, addr) in peers { ... }
    }
    _ => {}  // catch-all
}
```

Pattern matching in Rust is:
- **Exhaustive** — compiler ensures all variants are covered
- **Destructuring** — extracts fields inline
- `..` means "ignore remaining fields"

---

### 2.12 Testing with `#[cfg(test)]`

```rust
#[cfg(test)]                    // only compiled when running `cargo test`
mod tests {
    use super::*;               // import everything from parent module

    #[test]                     // marks function as a test case
    fn test_insert_and_retrieve() {
        let mut store = WeightStore::new();
        let tensor = vec![1.0f32, 2.0, 3.0, 4.0];
        let hash = store.insert(&tensor);
        let retrieved = store.get(&hash).unwrap();
        assert_eq!(retrieved, tensor.as_slice());
    }
}
```

| Macro | Purpose |
|-------|---------|
| `assert_eq!(a, b)` | Panics if `a != b` |
| `assert_ne!(a, b)` | Panics if `a == b` |
| `assert!(expr)` | Panics if `expr` is false |
| `assert!(matches!(...))` | Pattern-match assertion |

---

### 2.13 Iterators & Functional Combinators

```rust
// apply_delta: element-wise update via iterator chain
let updated: Vec<f32> = current.iter()
    .zip(delta.iter())         // pair up elements
    .map(|(w, d)| w + d)       // add weight + delta
    .collect();                // collect into Vec

// hex string from bytes
root.iter().map(|b| format!("{:02x}", b)).collect::<String>()
```

```
┌─────────────────────────────────────────┐
│        Iterator Pipeline                 │
│                                         │
│  source.iter()                          │
│    .zip(other)    // pair elements       │
│    .map(|x| ...)  // transform each      │
│    .filter(|x|...)// keep matching       │
│    .collect()     // materialize         │
│                                         │
│  Zero-cost: compiles to a simple loop    │
└─────────────────────────────────────────┘
```

---

## 3. Networking Concepts

### 3.1 TCP Socket Programming (Python Layer)

```
┌──────── TCP Message Exchange Protocol ────────┐
│                                               │
│  HEADER (8 bytes)         PAYLOAD             │
│  ┌──────────────┐  ┌─────────────────────┐    │
│  │ payload_len  │  │  pickle(object)     │    │
│  │ (little-end) │  │                     │    │
│  └──────────────┘  └─────────────────────┘    │
│                                               │
│  Sender:                                      │
│    data = pickle.dumps(packet)                │
│    header = len(data).to_bytes(8, 'little')   │
│    socket.sendall(header + data)              │
│                                               │
│  Receiver:                                    │
│    header = recv_exact(conn, 8)               │
│    size = int.from_bytes(header, 'little')    │
│    data = recv_exact(conn, size)              │
│    msg = pickle.loads(data)                   │
└───────────────────────────────────────────────┘
```

**Key functions:**

| Function | Purpose |
|----------|---------|
| `socket.create_connection((host, port), timeout)` | Client-side TCP connect |
| `socket.sendall(data)` | Send all bytes (handles partial sends) |
| `recv_exact(conn, n)` | Loop until exactly `n` bytes received |
| `socketserver.ThreadingTCPServer` | Multi-threaded TCP server |
| `socketserver.BaseRequestHandler` | Handler class per connection |

---

### 3.2 Threading Model

```
┌────────────────── Node Process ──────────────────┐
│                                                  │
│  Thread 1: TCP Gradient Server                   │
│    └─ ThreadingTCPServer.serve_forever()          │
│    └─ Spawns a new thread per connection          │
│       └─ MessageHandler.handle()                  │
│          └─ Dispatches to receive_gradient()       │
│             or receive_peer_announce()             │
│                                                  │
│  Thread 2: Training Loop                          │
│    └─ forward → backward → broadcast → wait →     │
│       aggregate → commit (repeats)                │
│                                                  │
│  Main Thread: FastAPI (Uvicorn)                   │
│    └─ /metrics, /root, /health, /generate         │
│                                                  │
│  Shared State (protected by threading.Lock):      │
│    • _inbox: {step → [GradientPacket]}             │
│    • peer_addrs: [(host, port)]                    │
└──────────────────────────────────────────────────┘
```

**Concurrency primitives used:**
- `threading.Lock` — mutual exclusion on shared state
- `threading.Thread(daemon=True)` — daemon threads die with main
- `threading.RLock` — re-entrant lock (inference engine)
- `threading.Event` — signal between threads

---

### 3.3 Peer Discovery & Gossip

```
                    Bootstrap
                   ┌─────────┐
         ┌─────── │ Node 0   │ ◄──────┐
         │        └─────────┘        │
         │ PeerAnnounce     PeerAnnounce
         ▼                           │
    ┌─────────┐              ┌─────────┐
    │ Node 1   │ ──────────▶ │ Node 2   │
    └─────────┘  PeerAnnounce └─────────┘

  Each node:
  1. Starts with --peers (initial known addresses)
  2. Sends PeerAnnounce to all known peers
  3. When a PeerAnnounce is received, adds sender to known list
  4. Re-announces periodically (every 50 steps)
  → Full mesh eventually established
```

---

### 3.4 Round Barrier Protocol

```
  Node 0              Node 1              Node 2
    │                   │                   │
    │ ── broadcast ────▶│                   │
    │ ── broadcast ──────────────────────▶  │
    │                   │── broadcast ────▶ │
    │ ◄── broadcast ────│                   │
    │                   │ ◄── broadcast ────│
    │ ◄── broadcast ──────────────────────  │
    │                   │                   │
    │◄─ ROUND_TIMEOUT ─▶│                   │
    │   (3 seconds)     │                   │
    │                   │                   │
    ├── aggregate ──────┼── aggregate  ─────┤
    │   (all have the same gradient set)    │
```

**Parameters:**
- `ROUND_TIMEOUT = 3.0s` — max wait per round
- `POLL_INTERVAL = 0.05s` — inbox check frequency
- `PEER_WAIT_TIMEOUT = 30.0s` — wait for minimum peers on startup

---

### 3.5 libp2p Networking (Rust Layer)

```
┌─────────── LatticeBehaviour (composed) ────────────┐
│                                                    │
│  ┌────────────┐  Pub/sub message flooding          │
│  │ GossipSub  │  Topics: lattice/gradients/v1       │
│  │            │          lattice/blocks/v1           │
│  └────────────┘                                    │
│                                                    │
│  ┌────────────┐  Distributed hash table             │
│  │ Kademlia   │  Peer routing & discovery           │
│  └────────────┘                                    │
│                                                    │
│  ┌────────────┐  Local network auto-discovery       │
│  │   mDNS     │  (multicast DNS, LAN only)          │
│  └────────────┘                                    │
│                                                    │
│  ┌────────────┐  Keepalive & latency                │
│  │   Ping     │  measurement                        │
│  └────────────┘                                    │
│                                                    │
│  ┌────────────┐  Exchange peer identity info        │
│  │ Identify   │  Protocol: /lattice/1.0.0           │
│  └────────────┘                                    │
│                                                    │
│  Transport: TCP + Noise (encryption) + Yamux (mux)  │
└────────────────────────────────────────────────────┘
```

| Protocol | What it does |
|----------|--------------|
| **GossipSub** | Publish/subscribe — message flooding across the network |
| **Kademlia (DHT)** | Distributed hash table for peer discovery and routing |
| **mDNS** | Zero-config LAN peer discovery via multicast DNS |
| **Noise** | Authenticated encryption for all connections (Diffie-Hellman) |
| **Yamux** | Stream multiplexer — multiple logical streams over one TCP connection |
| **Ping** | Keepalive probes, measures round-trip latency |
| **Identify** | Peers exchange protocol versions and public keys |

---

### 3.6 FastAPI / REST Layer

```
┌──── Each Node Exposes ──────────────────────────────┐
│                                                     │
│  GET  /metrics  → { node_id, step, loss,            │
│                     block_height, merkle_root,       │
│                     peers_seen, byz_rejected }       │
│                                                     │
│  GET  /root     → { merkle_root }                    │
│                                                     │
│  GET  /health   → { status: "ok" }                   │
│                                                     │
│  POST /generate → greedy token generation            │
│       body: { tokens: [int], max_tokens: int }       │
│       resp: { generated: [int], merkle_root: ... }   │
│                                                     │
│  Served by: Uvicorn (ASGI server)                    │
│  Framework: FastAPI (with Pydantic validation)       │
└─────────────────────────────────────────────────────┘
```

---

### 3.7 Serialization Formats

| Format | Where used | Library |
|--------|-----------|---------|
| **pickle** | TCP gradient exchange (Python ↔ Python) | `pickle` stdlib |
| **bincode** | libp2p message encoding (Rust ↔ Rust) | `bincode` crate |
| **JSON** | FastAPI REST responses | FastAPI auto-serialization |

---

## 4. PyTorch Methods & the Math Behind Them

### 4.1 `torch.randn(shape) * scale` — Xavier-like Initialization

**Used in:** `model.py` (weight initialization)

```python
scale = d_model ** -0.5   # = 1/√d_model
self.params['embed'] = torch.randn(vocab_size, d_model) * scale
```

**Math:**

Each weight is sampled from:

```
W_ij ~ N(0, 1/d_model)
```

This is a variance-scaled initialization. For `d_model = 128`:

```
scale = 1/√128 ≈ 0.0884
Var(W_ij) = (1/√128)² = 1/128
```

**Why:** Prevents activations from exploding or vanishing as they pass through layers. If weights are too large, outputs grow exponentially; too small, they shrink to zero.

---

### 4.2 `torch.randn_like(tensor)` — Gaussian Noise

**Used in:** `node.py` (DP-SGD noise injection)

```python
noise = torch.randn_like(g) * (clip_norm * noise_mult / n_nodes)
```

**Math:**

```
noise_ij ~ N(0, σ²)
where σ = C · σ_mult / n
      C = clip norm (1.0)
      σ_mult = noise multiplier (0.05)
      n = number of nodes
```

Creates a tensor of the same shape filled with samples from N(0,1), then scaled.

---

### 4.3 `tensor.requires_grad_(True)` — Enable Autograd

**Used in:** `model.py` (all parameters)

```python
for p in self.params.values():
    p.requires_grad_(True)
```

**What it does:** Tells PyTorch's autograd engine to track all operations on this tensor so gradients can be computed via backpropagation.

```
                   Computational Graph
                   ──────────────────
   input ──▶ op1 ──▶ op2 ──▶ op3 ──▶ loss
                                       │
              .backward()              │
                                       ▼
   ∂L/∂input ◀── ∂L/∂op1 ◀── ∂L/∂op2 ◀── ∂L/∂loss = 1
```

---

### 4.4 `tensor @ matrix.T` — Matrix Multiplication

**Used in:** `model.py` (projections, FFN)

```python
Q = x @ Wq.T     # (B, T, C) @ (C, C).T → (B, T, C)
```

**Math:**

```
(X · W^T)_ij = Σ_k  X_ik · W_jk

For a single token vector x ∈ ℝ^C and weight W ∈ ℝ^{C×C}:
q = x W^T    →   q_j = Σ_k x_k · W_{jk}
```

---

### 4.5 `tensor.view(B, T, n_heads, head_dim)` — Reshape

**Used in:** `model.py` (split heads for multi-head attention)

```python
Q = (x @ Wq.T).view(B, T, self.n_heads, head_dim).transpose(1, 2)
# (B, T, C) → (B, T, H, D) → (B, H, T, D)
```

**What it does:** Reinterprets the same memory as a different shape (no data copy). The `C = n_heads × head_dim` dimension is split into separate heads.

```
Before view:  [B, T, C=128]
After view:   [B, T, 4, 32]     ← 4 heads, 32 dims each
After transp: [B, 4, T, 32]     ← heads become a batch dimension
```

---

### 4.6 `tensor.transpose(dim0, dim1)` — Dimension Swap

**Used in:** `model.py` (attention computation)

```python
Q = Q.transpose(1, 2)             # (B,T,H,D) → (B,H,T,D)
att = Q @ K.transpose(-2, -1)     # (B,H,T,D) @ (B,H,D,T) → (B,H,T,T)
```

Swaps two dimensions. For the K transpose:

```
K:  shape (B, H, T, D)
K^T: shape (B, H, D, T)    ← last two dims swapped

Q @ K^T: (B,H,T,D) × (B,H,D,T) = (B,H,T,T)
This gives the attention score matrix: how much each token attends to every other.
```

---

### 4.7 `torch.tril(torch.ones(T, T))` — Causal Mask

**Used in:** `model.py` (masking future tokens)

```python
mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
att = att.masked_fill(mask == 0, float('-inf'))
```

**What it produces (T=4 example):**

```
┌─────────────────┐
│  1  0  0  0     │   Token 0 can only see token 0
│  1  1  0  0     │   Token 1 can see tokens 0, 1
│  1  1  1  0     │   Token 2 can see tokens 0, 1, 2
│  1  1  1  1     │   Token 3 can see all tokens
└─────────────────┘
```

Positions with 0 are filled with `-∞`, so after softmax they become probability 0. This enforces **autoregressive** (left-to-right) generation — a token cannot cheat by looking at future tokens.

---

### 4.8 `tensor.masked_fill(condition, value)` — Conditional Fill

**Used in:** `model.py`

```python
att = att.masked_fill(mask == 0, float('-inf'))
```

Everywhere `mask == 0` is True, set the attention score to -∞. After softmax, e^(-∞) = 0.

---

### 4.9 `F.softmax(tensor, dim=-1)` — Softmax Normalization

**Used in:** `model.py` (attention weights)

```python
att = F.softmax(att, dim=-1)
```

**Math:**

```
softmax(z_i) = e^{z_i} / Σ_j e^{z_j}
```

Properties:
- Output is a probability distribution (sums to 1)
- Larger values get exponentially more weight
- With `-inf` inputs: e^{-∞} = 0 → masked positions get zero probability

```
  Input scores:    [2.0,  1.0, -inf, -inf]
  After softmax:   [0.73, 0.27, 0.0,  0.0]
                    ─────────────
                    sums to 1.0
```

---

### 4.10 `tensor.contiguous()` — Memory Layout Fix

**Used in:** `model.py`

```python
y = (att @ V).transpose(1, 2).contiguous().view(B, T, C)
```

After `transpose`, tensor memory may not be contiguous (elements not in row-major order). `.contiguous()` copies data to contiguous memory so `.view()` can work.

```
  att @ V:        (B, H, T, D)
  .transpose(1,2): (B, T, H, D)  ← non-contiguous!
  .contiguous():   (B, T, H, D)  ← copy to contiguous memory
  .view(B, T, C):  (B, T, C)     ← reshape to merge heads
```

---

### 4.11 `x.mean(-1, keepdim=True)` and `x.std(-1, keepdim=True)` — Statistics

**Used in:** `model.py` (layer normalization)

```python
def layer_norm(self, x, g):
    mean = x.mean(-1, keepdim=True)
    std  = x.std(-1, keepdim=True)
    return g * (x - mean) / (std + 1e-5)
```

**Math — Layer Normalization:**

For each token vector x ∈ ℝ^C:

```
μ = (1/C) Σ_i x_i                     (mean over features)
σ = √((1/C) Σ_i (x_i - μ)²)          (std over features)

LayerNorm(x) = γ ⊙ (x - μ) / (σ + ε)

where:
  γ = learnable scale (the `g` parameter, initialized to 1)
  ε = 1e-5 (numerical stability)
  ⊙ = element-wise multiply
```

**Why:** Normalizes each token independently so activations stay in a stable range. Without it, deep networks suffer from internal covariate shift.

```
  Before LN:  x = [100.0, -50.0, 200.0, 0.0]   ← wild range
  After LN:   x = [-0.16, -1.37, 1.21, 0.32]    ← zero mean, unit variance
```

---

### 4.12 `F.gelu(tensor)` — GELU Activation Function

**Used in:** `model.py` (FFN block)

```python
h = F.gelu(h @ p[f'l{i}_W1'].T) @ p[f'l{i}_W2'].T
```

**Math:**

```
GELU(x) = x · Φ(x)

where Φ(x) = CDF of standard normal distribution

Approximation used in inference engine:
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
```

```
     Output
      │
  1.0 │              ╱
      │            ╱
  0.5 │          ╱
      │        ╱
  0.0 │──────╱───────────── Input
      │    ╱
 -0.2 │  ╱
      │╱
      └──────────────────
     -3  -2  -1   0   1   2   3

  GELU is a smooth version of ReLU.
  Negative values aren't fully zeroed — they're damped.
```

**Why:** GELU outperforms ReLU in transformer architectures because it provides a smooth non-linearity and doesn't completely kill negative values (like ReLU's hard zero).

---

### 4.13 `F.cross_entropy(logits, targets)` — Cross-Entropy Loss

**Used in:** `node.py` (training loss)

```python
logits = self.model.forward(idx)                              # (B, T, V)
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

**Math:**

```
For a single token prediction:

L = -log(softmax(z)_{target})

  = -log( e^{z_target} / Σ_j e^{z_j} )

  = -z_target + log(Σ_j e^{z_j})

where:
  z ∈ ℝ^V = logit vector (raw model output for each vocab token)
  target = index of the correct token
  V = vocab_size (65)
```

**Over the full batch:**

```
L_total = (1/N) Σ_{i=1}^{N} [ -z_i[target_i] + log(Σ_j e^{z_i[j]}) ]

where N = B × T (all token positions in the batch)
```

**Why:** Cross-entropy measures how well the model's probability distribution matches the true distribution (one-hot at the correct token). Minimizing it pushes the model to assign higher probability to correct tokens.

---

### 4.14 `loss.backward()` — Backpropagation

**Used in:** `node.py`

```python
loss.backward()
```

**Math — Chain Rule:**

```
For each parameter θ in the model:

∂L/∂θ = (∂L/∂z_n) · (∂z_n/∂z_{n-1}) · ... · (∂z_2/∂z_1) · (∂z_1/∂θ)

where z_1, z_2, ..., z_n are intermediate activations.
```

PyTorch walks the computational graph **backward** from loss to inputs, computing gradients via the chain rule. After `.backward()`:
- `param.grad` contains `∂L/∂param` for every parameter with `requires_grad=True`

---

### 4.15 `param.grad.detach().numpy().copy()` — Gradient Extraction

**Used in:** `node.py` (capture_gradients)

```python
packet[name] = param.grad.detach().numpy().copy()
```

| Method | Purpose |
|--------|---------|
| `.grad` | Access accumulated gradient (∂L/∂param) |
| `.detach()` | Remove from autograd graph (no more tracking) |
| `.numpy()` | Convert to NumPy array (shares memory!) |
| `.copy()` | Deep copy so it's safe after `grad.zero_()` |

---

### 4.16 `param.grad.zero_()` — Clear Gradients

**Used in:** `node.py`

```python
for param in model.params.values():
    if param.grad is not None:
        param.grad.zero_()
```

**Why:** PyTorch **accumulates** gradients by default. Without zeroing, gradients from the previous step would be added to the current step's gradients, corrupting the update.

---

### 4.17 `torch.no_grad()` — Disable Gradient Tracking

**Used in:** `node.py` (apply_gradients, generate)

```python
with torch.no_grad():
    param -= lr * torch.tensor(grads[name]) * clip_coef
```

**Why:** During weight updates and inference, we don't want PyTorch to record operations for backpropagation. `torch.no_grad()` disables autograd in its context, saving memory and compute.

---

### 4.18 `torch.manual_seed(seed)` — Reproducibility

**Used in:** `node.py`

```python
torch.manual_seed(GENESIS_SEED)      # same model init across all nodes
torch.manual_seed(step * 1000)        # same data batch across all nodes
```

**Why critical in Lattice:**
1. **Genesis weights:** All nodes must start with identical weights so their initial Merkle roots match. Same seed → same `torch.randn` sequence → same weights.
2. **Training batches:** All nodes compute gradients on the same random data so aggregation is meaningful.

---

### 4.19 `torch.randint(low, high, shape)` — Random Integer Tensor

**Used in:** `node.py` (synthetic training data)

```python
idx     = torch.randint(0, vocab_size, (4, 32))   # token IDs
targets = torch.randint(0, vocab_size, (4, 32))   # target labels
```

Creates a tensor filled with random integers from `[low, high)`. Here: batch of 4 sequences, each 32 tokens, vocab size 65.

---

### 4.20 `logits.view(-1, vocab_size)` — Flatten for Loss

**Used in:** `node.py`

```python
logits.view(-1, vocab_size)    # (B, T, V) → (B*T, V)
targets.view(-1)               # (B, T) → (B*T,)
```

**Why:** `F.cross_entropy` expects:
- `input`: `(N, C)` — N samples, C classes
- `target`: `(N,)` — N target indices

So we flatten the batch and sequence dimensions into one.

---

### 4.21 `logits[0, -1].argmax()` — Greedy Decode

**Used in:** `node.py` (/generate endpoint)

```python
next_t = int(logits[0, -1].argmax())
```

**Math:**

```
next_token = argmax_j  logits[0, T-1, j]

= the vocabulary index with the highest score at the last position
```

This is **greedy decoding** — always pick the most probable next token. Simple but can be repetitive. Alternatives: top-k sampling, nucleus sampling.

---

### 4.22 `tensor.norm()` — L2 Norm

**Used in:** `node.py` (gradient clipping)

```python
g_norm = torch.sqrt(sum(g.norm()**2 for g in all_g))
```

**Math:**

```
For individual tensor: ‖g‖₂ = √(Σ_i g_i²)

Global norm:  ‖G‖₂ = √(Σ_k ‖g_k‖₂²)   (across all parameter groups)

Clipping coefficient: clip_coef = min(1, C / (‖G‖₂ + ε))

Clipped gradient: g' = clip_coef · g
```

If the total gradient norm exceeds `C`, all gradients are scaled down proportionally. This prevents exploding gradients.

---

### 4.23 Summary Table of All PyTorch Methods

| Method | File | Purpose | Math |
|--------|------|---------|------|
| `torch.randn` | model.py | Weight init | W ~ N(0, 1/d) |
| `torch.randn_like` | node.py | DP noise | noise ~ N(0, σ²) |
| `requires_grad_` | model.py | Enable autograd | — |
| `@ (matmul)` | model.py | Linear projections | Y = XW^T |
| `.view` | model.py | Reshape (zero-copy) | — |
| `.transpose` | model.py | Swap dimensions | — |
| `torch.tril` | model.py | Causal mask | Lower triangular matrix |
| `.masked_fill` | model.py | Set -∞ for masked | — |
| `F.softmax` | model.py | Attention weights | e^z / Σe^z |
| `.contiguous` | model.py | Fix memory layout | — |
| `.mean` | model.py | Layer norm mean | μ = (1/C)Σx_i |
| `.std` | model.py | Layer norm std | σ = √((1/C)Σ(x-μ)²) |
| `F.gelu` | model.py | Activation | x·Φ(x) |
| `F.cross_entropy` | node.py | Loss function | -log(softmax(z)_target) |
| `.backward` | node.py | Backpropagation | Chain rule: ∂L/∂θ |
| `.detach` | node.py | Remove from graph | — |
| `.numpy` | node.py | Tensor → NumPy | — |
| `.grad.zero_` | node.py | Clear gradients | — |
| `torch.no_grad` | node.py | Disable autograd | — |
| `torch.manual_seed` | node.py | Reproducibility | Fix RNG state |
| `torch.randint` | node.py | Random data | Uniform integers |
| `.argmax` | node.py | Greedy decode | argmax(logits) |
| `.norm` | node.py | L2 norm | √(Σx²) |
| `torch.tensor` | node.py | NumPy → Tensor | — |

---

## 5. Federated Learning & Byzantine Fault Tolerance

### 5.1 Federated Averaging (FedAvg)

```python
def fedavg(packets):
    averaged[name] = stacked.mean(axis=0)
```

**Math:**

```
g_avg = (1/n) Σ_{i=1}^{n} g_i

where g_i = gradient from node i
      n   = number of nodes
```

```
  Node 0: g = [ 0.5, -0.3,  0.1]
  Node 1: g = [ 0.4, -0.2,  0.2]
  Node 2: g = [-5.0,  3.0, -4.0]  ← Byzantine!
  ──────────────────────────────────
  FedAvg:   [-1.37,  0.83, -1.23]  ← POISONED! 😱
```

**Problem:** A single Byzantine node can shift the average arbitrarily.

---

### 5.2 Krum — Byzantine-Robust Aggregation

```
┌──────────────── Krum Algorithm ─────────────────┐
│                                                 │
│  Input: n gradient vectors, parameter k          │
│  (k = n-2 tolerates 1 Byzantine among 3+ nodes) │
│                                                 │
│  For each gradient g_i:                          │
│    1. Compute distances to all other gradients   │
│       d_ij = ‖g_i - g_j‖²                       │
│    2. Sort distances                              │
│    3. Score(i) = sum of k smallest distances      │
│                                                 │
│  Select: winner = argmin_i Score(i)              │
│                                                 │
│  Output: g_winner (the most "central" gradient)  │
└─────────────────────────────────────────────────┘
```

**Math:**

```
Score(i) = Σ_{j ∈ KNN(i)} ‖g_i - g_j‖²

where KNN(i) = k nearest neighbors of g_i

winner = argmin_i Score(i)
```

**Why it works against Byzantine attacks:**

```
  Honest gradients cluster together:

       g₀ •  • g₁
            • g₂  (honest — close together)


                                    • g₃  (Byzantine — far away)

  Krum scores:
    Score(g₀) = dist²(g₀,g₁)                    ← small ✅
    Score(g₁) = dist²(g₁,g₀)                    ← small ✅
    Score(g₃) = dist²(g₃, nearest honest)        ← large ❌

  Winner: g₀ or g₁ (an honest gradient)
```

---

### 5.3 Coordinate-Wise Median

```python
def coordinate_wise_median(packets):
    median_flat = np.median(matrix, axis=0)
```

**Math:**

```
For each parameter index j:
  g_agg[j] = median({g_0[j], g_1[j], ..., g_{n-1}[j]})
```

**Why robust:** The median is unaffected by outliers. Even if a Byzantine node sends `g[j] = ±∞`, the median is still determined by the honest majority.

---

### 5.4 Gradient Poisoning Attack

```python
# Byzantine node in node.py:
grads = {k: v * -20.0 for k, v in grads.items()}
```

**Attack:** Reverse the gradient direction and scale by 20×. This pushes the model in the exact opposite direction of learning, scaled massively.

```
  Honest gradient:    g = [+0.1, -0.05, +0.2]
  Poisoned gradient: gₐ = [-2.0, +1.0,  -4.0]   (×-20)
```

---

## 6. Cryptographic Integrity (Merkle Trees)

### 6.1 SHA-256 Hashing

```python
def hash_tensor(arr: np.ndarray) -> bytes:
    return hashlib.sha256(arr.astype(np.float32).tobytes()).digest()
```

```rust
pub fn hash_tensor(data: &[f32]) -> [u8; 32] {
    Sha256::new().chain_update(bytemuck::cast_slice(data)).finalize().into()
}
```

**Properties:**
- **Deterministic:** Same input → same 256-bit output
- **Avalanche effect:** Flip 1 bit of input → ~50% of output bits change
- **Pre-image resistant:** Given hash, can't find the input
- **Collision resistant:** Can't find two inputs with the same hash

---

### 6.2 Merkle Tree Construction

```
          Root = H(AB + CD)
         ╱                ╲
    AB = H(A+B)       CD = H(C+D)
    ╱       ╲          ╱       ╲
  A=H(w₀)  B=H(w₁)  C=H(w₂)  D=H(w₂)
    │         │         │         │
   embed    l0_Wq     l0_Wk    (duplicate
  weights   weights   weights   if odd)
```

**Algorithm (bottom-up):**

```python
# 1. Hash each weight tensor → leaf
leaves = [SHA256(param) for param in sorted_params]

# 2. Pair-wise hash up the tree
while len(current) > 1:
    for i in range(0, len(current), 2):
        left  = current[i]
        right = current[i+1] if exists else current[i]  # duplicate if odd
        next.append(SHA256(left || right))
    current = next

# 3. Root is the final hash
merkle_root = current[0]
```

**Why Merkle trees in Lattice:**
1. **Convergence verification:** If two nodes have the same Merkle root, they provably have the exact same model weights
2. **Tamper detection:** If a single weight is altered, the root changes completely
3. **Efficient verification:** Can prove a single weight is correct with O(log n) hashes

---

### 6.3 Weight Integrity in Inference

```
┌────── Inference Engine Verification ──────┐
│                                           │
│  1. Load tensor from store by hash        │
│  2. Recompute SHA-256 of loaded tensor    │
│  3. Compare with expected hash            │
│  4. If mismatch → WeightTamperError!      │
│                                           │
│  This ensures no weight corruption during │
│  storage, transfer, or hot-swap.          │
└───────────────────────────────────────────┘
```

---

### 6.4 Gradient Block Chain

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Genesis  │────▶│ Block 1  │────▶│ Block 2  │───▶ ...
│ h=0      │     │ h=1      │     │ h=2      │
│ root=[0] │     │ prev=[0] │     │ prev=r1  │
│          │     │ new=r1   │     │ new=r2   │
└──────────┘     └──────────┘     └──────────┘

Validation rules:
  1. block.height == prev.height + 1
  2. block.prev_root == prev.new_root
  3. block.timestamp > prev.timestamp
```

Each block records the gradient delta that was applied, creating an auditable history of all model updates.

---

## 7. Differential Privacy (DP-SGD)

### 7.1 Gradient Clipping

```python
g_norm    = torch.sqrt(sum(g.norm()**2 for g in all_g))
clip_coef = min(1.0, clip_norm / (g_norm + 1e-6))
g_clipped = g * clip_coef
```

**Math:**

```
‖G‖ = √(Σ_k ‖g_k‖²)        (global gradient norm)

        ⎧ g                  if ‖G‖ ≤ C
g' =    ⎨
        ⎩ g · (C / ‖G‖)     if ‖G‖ > C

This bounds the sensitivity: max ‖g'‖ ≤ C
```

**Why:** Clipping ensures that no single training example can have unbounded influence on the gradient, which is the prerequisite for the privacy guarantee.

---

### 7.2 Gaussian Noise Addition

```python
noise = torch.randn_like(g) * (clip_norm * noise_mult / n_nodes)
noisy_grad = g_clipped + noise
```

**Math:**

```
g̃ = g' + N(0, σ² · I)

where σ = C · σ_mult / n
      C = clip_norm = 1.0
      σ_mult = noise multiplier = 0.05
      n = number of nodes = 4
```

**The DP Guarantee (informal):**

> After gradient clipping and noise addition, the gradients an adversary observes
> reveal only (ε, δ)-bounded information about any single training example.

```
┌──────────── DP-SGD Pipeline ────────────────┐
│                                             │
│  Raw gradients  ──▶  Clip to bound C        │
│                      ‖g‖ ≤ C               │
│                         │                   │
│                         ▼                   │
│                 Add Gaussian noise           │
│                 g̃ = g' + N(0, σ²)           │
│                         │                   │
│                         ▼                   │
│                 Broadcast noisy gradient     │
│                                             │
│  The noise masks individual contributions   │
│  while preserving the aggregate signal.     │
└─────────────────────────────────────────────┘
```

---

### 7.3 Hot Swap (Inference Engine)

```
┌──────── Hot Swap Protocol ─────────┐
│                                    │
│  1. Set _swapping = True           │
│  2. Drain in-flight requests       │
│     (wait for _inflight == 0)      │
│  3. Load new weights from store    │
│  4. Verify each tensor hash        │
│  5. Atomically swap weights dict   │
│  6. Update pinned_root             │
│  7. Set _swapping = False          │
│                                    │
│  During swap:                      │
│    • New requests → "retry" error  │
│    • No torn reads possible        │
└────────────────────────────────────┘
```

This is a **drain-then-swap** pattern common in production systems. It ensures that no inference request sees a partially-updated model.

---

## Appendix: Complete File Map

| File | Language | Lines | Purpose |
|------|----------|-------|---------|
| `python/training/model.py` | Python | 76 | Custom 2-layer transformer (no nn.Module) |
| `python/training/node.py` | Python | 105 | Phase 2: multiprocessing federated training |
| `python/aggregation/bft.py` | Python | 83 | Krum + Coordinate-wise Median |
| `python/node.py` | Python | 540 | Phase 8: TCP P2P node with round barriers |
| `python/inference/engine.py` | Python | 140 | Hash-verified inference with hot swap |
| `python/inference/server.py` | Python | 46 | FastAPI inference endpoint |
| `python/dashboard.py` | Python | 141 | Live terminal network monitor |
| `python/simulation/byzantine_demo.py` | Python | 103 | FedAvg vs Krum comparison |
| `rust/merkle-store/src/lib.rs` | Rust | 147 | Content-addressed weight storage + Merkle tree |
| `rust/chain-state/src/lib.rs` | Rust | 156 | Gradient block chain with validation |
| `rust/p2p-net/src/lib.rs` | Rust | 144 | libp2p: GossipSub + Kademlia + mDNS |
| `rust/lattice-bridge/src/lib.rs` | Rust | 59 | PyO3 FFI bridge (Rust ↔ Python) |
| `rust/consensus/src/lib.rs` | Rust | 15 | Consensus protocol (stub) |
| `launch.sh` | Bash | 115 | Network orchestration script |
