# python/node.py
"""
Lattice node — Phase 8: round-based federated training with root convergence

Usage:
  # terminal 1 (bootstrap)
  python node.py --id 0 --port 8000 --p2p-port 7000

  # terminal 2
  python node.py --id 1 --port 8001 --p2p-port 7001 --peers localhost:7000

  # terminal 3
  python node.py --id 2 --port 8002 --p2p-port 7002 --peers localhost:7000,localhost:7001

  # terminal 4 (Byzantine)
  python node.py --id 3 --port 8003 --p2p-port 7003 --peers localhost:7000,localhost:7001,localhost:7002 --byzantine
"""

# All nodes use the same seed → identical genesis weights → same starting Merkle root
GENESIS_SEED = 0

# Max time (seconds) to wait for peer gradients each round
ROUND_TIMEOUT = 3.0

# How often to poll inbox during wait (seconds)
POLL_INTERVAL = 0.05

# Max time (seconds) to wait for minimum peers before training
PEER_WAIT_TIMEOUT = 30.0

import argparse
import hashlib
import json
import pickle
import socket
import socketserver
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, 'training')
sys.path.insert(0, 'aggregation')

from model import LatticeTransformer
from bft import krum, coordinate_wise_median

# ── gradient packet ───────────────────────────────────────────────
class GradientPacket:
    def __init__(self, node_id: int, step: int, grads: dict):
        self.node_id = node_id
        self.step    = step
        self.grads   = grads  # {param_name: np.ndarray}

def capture_gradients(model: LatticeTransformer) -> dict:
    packet = {}
    for name, param in model.params.items():
        if param.grad is not None:
            packet[name] = param.grad.detach().numpy().copy()
    return packet

def apply_dp_noise(grads: dict, clip_norm=1.0, noise_mult=0.05, n_nodes=4) -> dict:
    all_g     = [torch.tensor(g) for g in grads.values()]
    g_norm    = torch.sqrt(sum(g.norm()**2 for g in all_g))
    clip_coef = min(1.0, clip_norm / (g_norm + 1e-6))
    noisy = {}
    for name, grad in grads.items():
        g     = torch.tensor(grad) * clip_coef
        noise = torch.randn_like(g) * (clip_norm * noise_mult / n_nodes)
        noisy[name] = (g + noise).numpy()
    return noisy

def apply_gradients(model: LatticeTransformer, grads: dict, lr=3e-4, max_norm=1.0):
    with torch.no_grad():
        all_g     = [torch.tensor(g) for g in grads.values()]
        g_norm    = torch.sqrt(sum(g.norm()**2 for g in all_g))
        clip_coef = min(1.0, max_norm / (g_norm + 1e-6))
        for name, param in model.params.items():
            if name in grads:
                param -= lr * torch.tensor(grads[name]) * clip_coef
        for param in model.params.values():
            if param.grad is not None:
                param.grad.zero_()

# ── Merkle root helpers (pure Python — no Rust bridge needed here) ─
def hash_tensor(arr: np.ndarray) -> bytes:
    return hashlib.sha256(arr.astype(np.float32).tobytes()).digest()

def compute_merkle_root(model: LatticeTransformer) -> str:
    """SHA-256 Merkle root over all model weight tensors in fixed order."""
    leaves = []
    for name in sorted(model.params.keys()):
        arr = model.params[name].detach().numpy().astype(np.float32)
        leaves.append(hash_tensor(arr))

    # build tree bottom-up
    current = leaves
    while len(current) > 1:
        nxt = []
        for i in range(0, len(current), 2):
            left  = current[i]
            right = current[i + 1] if i + 1 < len(current) else current[i]
            nxt.append(hashlib.sha256(left + right).digest())
        current = nxt
    return current[0].hex() if current else "0" * 64

# ── TCP message exchange ──────────────────────────────────────────
HEADER_SIZE = 8   # 8-byte little-endian payload length prefix

# Message types sent over TCP
class PeerAnnounce:
    """Tells a peer about our p2p address so they can send gradients back."""
    def __init__(self, node_id: int, host: str, port: int):
        self.node_id = node_id
        self.host    = host
        self.port    = port

def send_packet(host: str, port: int, packet) -> bool:
    """Serialize and send any picklable object to a peer. Returns True on success."""
    try:
        data = pickle.dumps(packet)
        header = len(data).to_bytes(HEADER_SIZE, 'little')
        with socket.create_connection((host, port), timeout=2.0) as s:
            s.sendall(header + data)
        return True
    except Exception:
        return False

def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionResetError("peer closed connection")
        buf += chunk
    return buf

class MessageHandler(socketserver.BaseRequestHandler):
    """Handles one incoming TCP connection (gradient or peer announce)."""
    def handle(self):
        try:
            header = recv_exact(self.request, HEADER_SIZE)
            size   = int.from_bytes(header, 'little')
            data   = recv_exact(self.request, size)
            msg    = pickle.loads(data)
            if isinstance(msg, GradientPacket):
                self.server.node.receive_gradient(msg)
            elif isinstance(msg, PeerAnnounce):
                self.server.node.receive_peer_announce(msg)
        except Exception:
            pass

class GradientServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads      = True

    def __init__(self, addr, node):
        self.node = node
        super().__init__(addr, MessageHandler)

# ── FastAPI metrics app ───────────────────────────────────────────
# built per-node so each process has its own state reference
def make_app(node_ref):
    app = FastAPI(title=f"Lattice node {node_ref.node_id}")

    @app.get("/metrics")
    def metrics():
        return {
            "node_id":        node_ref.node_id,
            "step":           node_ref.step,
            "block_height":   node_ref.block_height,
            "merkle_root":    node_ref.current_root,
            "loss":           round(node_ref.last_loss, 4) if node_ref.last_loss else None,
            "peers_seen":     node_ref.peers_seen,
            "byz_rejected":   node_ref.byzantine_rejected,
            "byzantine_self": node_ref.args.byzantine,
        }

    @app.get("/root")
    def root():
        return {"merkle_root": node_ref.current_root}

    @app.get("/health")
    def health():
        return {"status": "ok", "node_id": node_ref.node_id}

    class GenerateRequest(BaseModel):
        tokens:     list[int]
        max_tokens: int = 20

    @app.post("/generate")
    def generate(req: GenerateRequest):
        """Simple greedy token generation — uses current model weights."""
        if node_ref.model is None:
            return {"error": "model not ready"}
        tokens = list(req.tokens)[-32:]
        results = []
        with torch.no_grad():
            for _ in range(req.max_tokens):
                idx    = torch.tensor([tokens[-32:]])
                logits = node_ref.model.forward(idx)
                next_t = int(logits[0, -1].argmax())
                results.append(next_t)
                tokens.append(next_t)
        return {
            "generated":   results,
            "merkle_root": node_ref.current_root[:12] + "...",
        }

    return app

# ── LatticeNode ───────────────────────────────────────────────────
class LatticeNode:
    def __init__(self, args):
        self.args = args
        self.node_id = args.id

        # training state
        self.model        = None
        self.step         = 0
        self.block_height = 0
        self.last_loss    = None
        self.current_root = "0" * 64

        # step-indexed gradient inbox
        # {step_number: [GradientPacket, ...]}
        self._inbox      = {}
        self._inbox_lock = threading.Lock()

        # metrics
        self.peers_seen         = 0
        self.byzantine_rejected = 0

        # peer addresses — initially from --peers, grows via discovery
        self.peer_addrs      = []
        self._known_peers    = set()   # (host, port) dedup
        self._peers_lock     = threading.Lock()
        if args.peers:
            for p in args.peers.split(','):
                h, port = p.strip().split(':')
                self._add_peer(h, int(port))

        self.running = True

    def _add_peer(self, host: str, port: int):
        """Add a peer address (deduplicated)."""
        key = (host, port)
        with self._peers_lock:
            if key not in self._known_peers:
                self._known_peers.add(key)
                self.peer_addrs.append(key)
                print(f"  [node {self.node_id}] discovered peer at {host}:{port}")

    def receive_peer_announce(self, announce: PeerAnnounce):
        """Called from TCP handler — register a new peer."""
        self._add_peer(announce.host, announce.port)

    def receive_gradient(self, packet: GradientPacket):
        """Called from TCP handler thread — file into step-indexed inbox."""
        with self._inbox_lock:
            if packet.step not in self._inbox:
                self._inbox[packet.step] = []
            self._inbox[packet.step].append(packet)
            # update peer count from all steps
            all_ids = set()
            for pkts in self._inbox.values():
                for p in pkts:
                    all_ids.add(p.node_id)
            self.peers_seen = len(all_ids)

    def _collect_step(self, step: int) -> list[GradientPacket]:
        """Drain gradient packets for a specific step."""
        with self._inbox_lock:
            packets = self._inbox.pop(step, [])
        return packets

    def _broadcast_gradient(self, packet: GradientPacket):
        """Send this node's gradient to all known peers."""
        with self._peers_lock:
            addrs = list(self.peer_addrs)
        for host, port in addrs:
            send_packet(host, port, packet)

    def _announce_to_peers(self):
        """Tell all known peers about our p2p address so they can reach us."""
        announce = PeerAnnounce(self.node_id, 'localhost', self.args.p2p_port)
        with self._peers_lock:
            addrs = list(self.peer_addrs)
        for host, port in addrs:
            send_packet(host, port, announce)

    def _wait_for_peers(self, step: int, expected: int) -> list[GradientPacket]:
        """
        Round barrier: wait up to ROUND_TIMEOUT for `expected` peer gradients
        for this step. Returns whatever we've collected when done.
        """
        deadline = time.time() + ROUND_TIMEOUT
        while time.time() < deadline:
            with self._inbox_lock:
                got = len(self._inbox.get(step, []))
            if got >= expected:
                break
            time.sleep(POLL_INTERVAL)
        return self._collect_step(step)

    def _aggregate(self, packets: list[GradientPacket]) -> dict | None:
        if len(packets) < 2:
            return None

        # Krum with k = n-2 (tolerates up to 1 Byzantine in n=3+)
        k = max(1, len(packets) - 2)
        try:
            aggregated, winner_id = krum(packets, k=k, return_winner=True)
            if winner_id == -1:
                self.byzantine_rejected += 1
                print(f"  [node {self.node_id}] KRUM REJECTED Byzantine gradient")
            return aggregated
        except Exception as e:
            print(f"  [node {self.node_id}] aggregation error: {e}")
            return None

    def _commit_block(self, aggregated: dict):
        """
        Gradient block commit:
        apply delta, update Merkle root, increment block height.
        """
        apply_gradients(self.model, aggregated)
        self.current_root = compute_merkle_root(self.model)
        self.block_height += 1
        if self.block_height % 20 == 0 or self.block_height <= 3:
            print(
                f"  [node {self.node_id}] "
                f"block {self.block_height:4d} committed  "
                f"root={self.current_root[:16]}..."
            )

    # ── training loop ─────────────────────────────────────────────
    def _wait_for_min_peers(self):
        """Block until we have at least --min-peers connections."""
        min_p = self.args.min_peers
        if min_p <= 0:
            return

        print(f"[node {self.node_id}] waiting for {min_p} peer(s) before training...")
        deadline = time.time() + PEER_WAIT_TIMEOUT
        while time.time() < deadline:
            with self._peers_lock:
                n = len(self.peer_addrs)
            if n >= min_p:
                print(f"[node {self.node_id}] {n} peer(s) connected — starting training")
                return
            # keep announcing so peers discover us too
            self._announce_to_peers()
            time.sleep(0.5)

        with self._peers_lock:
            n = len(self.peer_addrs)
        print(f"[node {self.node_id}] timeout — proceeding with {n} peer(s)")

    def training_loop(self):
        # All nodes use the SAME seed → identical genesis weights
        torch.manual_seed(GENESIS_SEED)
        self.model = LatticeTransformer()
        self.current_root = compute_merkle_root(self.model)
        vocab_size = 65

        print(f"[node {self.node_id}] training loop started")
        print(f"[node {self.node_id}] genesis root = {self.current_root[:16]}...")

        # announce ourselves to initial peers
        self._announce_to_peers()

        # ── WAIT for peers before training (prevents solo divergence) ──
        self._wait_for_min_peers()

        for step in range(self.args.steps):
            if not self.running:
                break

            # ── forward + backward ───────────────────────────────
            # Use step-based seed so all nodes compute on same random batch
            torch.manual_seed(step * 1000)
            idx     = torch.randint(0, vocab_size, (4, 32))
            targets = torch.randint(0, vocab_size, (4, 32))

            for p in self.model.params.values():
                if p.grad is not None:
                    p.grad.zero_()

            logits  = self.model.forward(idx)
            loss    = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            self.last_loss = loss.item()

            # ── capture + optionally poison ──────────────────────
            grads = capture_gradients(self.model)
            if self.args.byzantine:
                grads = {k: v * -20.0 for k, v in grads.items()}
                if step % 50 == 0:
                    print(f"  [node {self.node_id}] BYZANTINE: poisoning gradient")

            noisy  = apply_dp_noise(grads)
            packet = GradientPacket(self.node_id, step, noisy)

            # ── broadcast to peers ───────────────────────────────
            self._broadcast_gradient(packet)

            # ── round barrier: wait for peer gradients ───────────
            with self._peers_lock:
                n_peers = len(self.peer_addrs)
            peer_packets = self._wait_for_peers(step, expected=n_peers)
            all_packets  = [packet] + peer_packets

            # ── aggregate + commit ───────────────────────────────
            aggregated = self._aggregate(all_packets)
            if aggregated:
                self._commit_block(aggregated)
            else:
                # Not enough peers for aggregation — skip this step entirely.
                # Do NOT apply own gradient solo, that would cause divergence.
                if step % 20 == 0:
                    print(f"  [node {self.node_id}] step {step}: skipped (waiting for peers)")

            # ── periodic status ──────────────────────────────────
            if step % 20 == 0:
                print(
                    f"[node {self.node_id}] "
                    f"step={step:4d}  "
                    f"loss={self.last_loss:.4f}  "
                    f"peers={len(peer_packets)}/{n_peers}  "
                    f"block={self.block_height}  "
                    f"root={self.current_root[:12]}..."
                )

            # re-announce periodically so late-joining nodes find us
            if step % 50 == 0 and step > 0:
                self._announce_to_peers()

            self.step = step

        print(f"[node {self.node_id}] training complete — final root={self.current_root[:16]}...")
        self.running = False

    # ── start everything ──────────────────────────────────────────
    def start(self):
        # 1. TCP gradient server
        grad_server = GradientServer(('0.0.0.0', self.args.p2p_port), self)
        tcp_thread  = threading.Thread(
            target=grad_server.serve_forever,
            daemon=True,
            name=f"tcp-{self.node_id}"
        )
        tcp_thread.start()
        print(f"[node {self.node_id}] gradient server listening on :{self.args.p2p_port}")

        # 2. training loop
        train_thread = threading.Thread(
            target=self.training_loop,
            daemon=True,
            name=f"train-{self.node_id}"
        )
        train_thread.start()

        # 3. FastAPI — blocks main thread (Ctrl-C stops everything)
        app = make_app(self)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=self.args.port,
            log_level="warning",
        )

# ── bft.py patch — add return_winner support ─────────────────────
# If your bft.py krum() doesn't return the winner id, patch it here:
import aggregation.bft as _bft

_orig_krum = _bft.krum

def _krum_with_winner(packets, k=None, return_winner=False):
    """Thin wrapper that optionally returns the winning node_id."""
    matrix, names = _bft.stack_gradients(packets)
    n = len(matrix)
    if k is None:
        k = max(1, n - 2)

    matrix64 = matrix.astype(np.float64)
    scores = np.zeros(n)
    for i in range(n):
        dists = sorted(
            float(np.dot(matrix64[i] - matrix64[j], matrix64[i] - matrix64[j]))
            for j in range(n) if i != j
        )
        scores[i] = sum(dists[:k])

    if not np.all(np.isfinite(scores)):
        winner = 0
    else:
        winner = int(np.argmin(scores))

    winner_node_id = packets[winner].node_id
    aggregated = _bft.unstack_gradients(matrix[winner], packets, names)

    if return_winner:
        return aggregated, winner_node_id
    return aggregated

# monkey-patch so the rest of the codebase benefits too
_bft.krum = _krum_with_winner
krum = _krum_with_winner

# ── entrypoint ────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Lattice P2P training node")
    p.add_argument('--id',        type=int,   required=True)
    p.add_argument('--port',      type=int,   default=8000,  help="FastAPI HTTP port")
    p.add_argument('--p2p-port',  type=int,   default=7000,  help="TCP gradient exchange port")
    p.add_argument('--peers',     type=str,   default='',    help="comma-separated host:p2p-port list")
    p.add_argument('--byzantine', action='store_true',        help="inject gradient poisoning")
    p.add_argument('--steps',     type=int,   default=300)
    p.add_argument('--min-peers', type=int,   default=1,     help="wait for N peers before training")
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(f"\n{'='*55}")
    print(f"  Lattice node {args.id}")
    print(f"  HTTP  : http://localhost:{args.port}/metrics")
    print(f"  P2P   : localhost:{args.p2p_port}")
    print(f"  peers : {args.peers or 'none (bootstrap)'}")
    print(f"  byz   : {args.byzantine}")
    print(f"{'='*55}\n")
    node = LatticeNode(args)
    node.start()
