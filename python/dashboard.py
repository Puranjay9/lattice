#!/usr/bin/env python3
"""
Lattice Dashboard — live terminal monitor for all running nodes.

Usage:
    python dashboard.py                  # default: poll nodes 0-2
    python dashboard.py --nodes 4        # poll nodes 0-3
    python dashboard.py --byzantine      # include Byzantine node
"""

import argparse
import json
import time
import urllib.request
import os

# ── config ────────────────────────────────────────────────────────
BASE_HTTP = 8000
POLL_INTERVAL = 1.5  # seconds between refreshes

# ── ANSI colors ───────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
WHITE  = "\033[37m"
BG_GREEN  = "\033[42m"
BG_RED    = "\033[41m"

def clear():
    os.system('clear' if os.name != 'nt' else 'cls')

def fetch_metrics(port: int) -> dict | None:
    try:
        url = f"http://localhost:{port}/metrics"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

def format_root(root: str, width: int = 16) -> str:
    if not root:
        return "—"
    return root[:width] + "..."

def main():
    parser = argparse.ArgumentParser(description="Lattice Dashboard")
    parser.add_argument('--nodes', type=int, default=3, help="Number of honest nodes")
    parser.add_argument('--byzantine', action='store_true', help="Include Byzantine node")
    args = parser.parse_args()

    n_total = args.nodes + (1 if args.byzantine else 0)
    ports = [BASE_HTTP + i for i in range(n_total)]

    print(f"{BOLD}Lattice Dashboard{RESET} — polling {n_total} nodes every {POLL_INTERVAL}s")
    print(f"Press Ctrl-C to stop.\n")
    time.sleep(0.5)

    try:
        while True:
            clear()
            results = []
            for port in ports:
                m = fetch_metrics(port)
                results.append((port, m))

            # header
            print(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
            print(f"{BOLD}{CYAN}║                          LATTICE — NETWORK DASHBOARD                              ║{RESET}")
            print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════════════════════════╝{RESET}")
            print()

            # column headers
            print(
                f"  {BOLD}{'Node':>4}  {'Step':>5}  {'Loss':>7}  "
                f"{'Blocks':>6}  {'Peers':>5}  {'ByzRej':>6}  "
                f"{'Merkle Root':<22}  {'Status':<10}{RESET}"
            )
            print(f"  {'─'*4}  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*22}  {'─'*10}")

            honest_roots = []

            for port, m in results:
                if m is None:
                    nid = port - BASE_HTTP
                    print(f"  {DIM}{nid:>4}  {'—':>5}  {'—':>7}  {'—':>6}  {'—':>5}  {'—':>6}  {'offline':<22}  {'⬤ DOWN':<10}{RESET}")
                    continue

                nid   = m['node_id']
                step  = m.get('step', 0)
                loss  = m.get('loss')
                blk   = m.get('block_height', 0)
                peers = m.get('peers_seen', 0)
                byz_r = m.get('byz_rejected', 0)
                root  = m.get('merkle_root', '')
                is_byz = m.get('byzantine_self', False)

                loss_str = f"{loss:.4f}" if loss is not None else "—"
                root_str = format_root(root)

                if is_byz:
                    status = f"{RED}☠️ BYZANTINE{RESET}"
                else:
                    status = f"{GREEN}● HONEST{RESET}"
                    honest_roots.append(root)

                byz_color = RED if byz_r > 0 else WHITE
                line = (
                    f"  {BOLD}{nid:>4}{RESET}  {step:>5}  {loss_str:>7}  "
                    f"{blk:>6}  {peers:>5}  {byz_color}{byz_r:>6}{RESET}  "
                    f"{DIM}{root_str:<22}{RESET}  {status}"
                )
                print(line)

            # root convergence check
            print()
            if len(honest_roots) >= 2:
                if len(set(honest_roots)) == 1:
                    print(f"  {BG_GREEN}{BOLD} ✅ ROOTS CONVERGED {RESET}  All honest nodes share root: {DIM}{honest_roots[0][:24]}...{RESET}")
                else:
                    unique = len(set(honest_roots))
                    print(f"  {BG_RED}{BOLD} ❌ ROOTS DIVERGED  {RESET}  {unique} different roots across {len(honest_roots)} honest nodes")
            elif len(honest_roots) == 1:
                print(f"  {YELLOW}⏳ Only 1 honest node reporting — waiting for others...{RESET}")
            else:
                print(f"  {YELLOW}⏳ No honest nodes reporting yet...{RESET}")

            print(f"\n  {DIM}Last refresh: {time.strftime('%H:%M:%S')}  |  Polling every {POLL_INTERVAL}s  |  Ctrl-C to stop{RESET}")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n{BOLD}Dashboard stopped.{RESET}")

if __name__ == '__main__':
    main()
