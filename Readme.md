# Lattice

**Decentralized AI training on a peer-to-peer network.**

Every node trains the same transformer model, shares gradients over TCP, aggregates them with Byzantine fault tolerance (Krum), and commits each update as a block with a SHA-256 Merkle root. Honest nodes converge to the same model state — verified by matching Merkle roots.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Lattice Node                      │
│                                                     │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │Transformer│─▶│ Gradients │─▶│  DP Noise + Clip │  │
│  │   Model   │  │  .backward│  │  (privacy)       │  │
│  └──────────┘  └───────────┘  └────────┬─────────┘  │
│       ▲                                │             │
│       │                                ▼             │
│  ┌────┴──────┐  ┌───────────┐  ┌──────────────────┐  │
│  │  Apply    │◀─│   Krum    │◀─│  TCP Broadcast   │  │
│  │  Gradient │  │   BFT     │  │  (gossip)        │  │
│  └────┬──────┘  └───────────┘  └──────────────────┘  │
│       │                                              │
│       ▼                                              │
│  ┌──────────────────────────────────────────────┐    │
│  │  Merkle Root  ──▶  Block Commit  ──▶  /root  │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
         ▲               ▲              ▲
         │    TCP P2P     │              │
         ▼               ▼              ▼
    [ Node 0 ]      [ Node 1 ]     [ Node 2 ]
```

## Quick Start

```bash
# 1. activate venv
source .venv/bin/activate

# 2. launch 3 honest nodes (one command)
./launch.sh

# 3. launch with a Byzantine attacker
./launch.sh --byzantine

# 4. monitor in another terminal
python python/dashboard.py
python python/dashboard.py --byzantine   # if attacker is running
```

## Manual Node Launch

```bash
cd python

# terminal 1 — bootstrap node
python node.py --id 0 --port 8000 --p2p-port 7000

# terminal 2
python node.py --id 1 --port 8001 --p2p-port 7001 --peers localhost:7000

# terminal 3
python node.py --id 2 --port 8002 --p2p-port 7002 --peers localhost:7000,localhost:7001

# terminal 4 — Byzantine attacker (optional)
python node.py --id 3 --port 8003 --p2p-port 7003 --peers localhost:7000,localhost:7001,localhost:7002 --byzantine
```

## API

Each node exposes a FastAPI server:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Node status: step, loss, block height, merkle root, peers |
| `/root` | GET | Current Merkle root hash |
| `/health` | GET | Health check |
| `/generate` | POST | Generate tokens from current model state |

```bash
# check if honest nodes converged
curl localhost:8000/root
curl localhost:8001/root
# roots should match ✅

# generate tokens
curl -X POST localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"tokens": [1,2,3,4,5], "max_tokens": 10}'
```

## Byzantine Demo

Standalone proof that Krum defends against gradient poisoning:

```bash
cd python/simulation
python byzantine_demo.py
# generates byzantine_demo.png — FedAvg diverges, Krum stays stable
```

## Rust Crates

| Crate | Purpose |
|-------|---------|
| `merkle-store` | Content-addressed weight storage with Merkle tree |
| `chain-state` | Gradient block chain with validation |
| `lattice-bridge` | PyO3 bridge exposing Rust to Python |
| `p2p-net` | libp2p networking (gossipsub + Kademlia) |
| `consensus` | Consensus protocol (stub) |

```bash
cargo test   # run all Rust tests
```

## Tech Stack

- **Training**: PyTorch (custom transformer, no nn.Module)
- **Aggregation**: Krum / Coordinate-wise Median (Byzantine-robust)
- **Privacy**: DP-SGD (gradient clipping + Gaussian noise)
- **Integrity**: SHA-256 Merkle trees over model weights
- **Networking**: TCP sockets (Python) / libp2p (Rust, in progress)
- **Chain**: Gradient block chain with height + prev_root validation
- **API**: FastAPI + Uvicorn
