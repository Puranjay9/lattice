#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  Lattice — launch a local P2P training network
#
#  Usage:
#    ./launch.sh                    # 3 honest nodes
#    ./launch.sh --nodes 4          # 4 honest nodes
#    ./launch.sh --byzantine        # 3 honest + 1 Byzantine
#    ./launch.sh --nodes 4 --byzantine --steps 200
# ─────────────────────────────────────────────────────────────────

set -e

N_NODES=3
STEPS=300
BYZANTINE=false
BASE_HTTP=8000
BASE_P2P=7000

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)     N_NODES="$2"; shift 2 ;;
        --steps)     STEPS="$2";   shift 2 ;;
        --byzantine) BYZANTINE=true; shift ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PIDS=()

cleanup() {
    echo ""
    echo "╔═══════════════════════════════════╗"
    echo "║   Shutting down all nodes...      ║"
    echo "╚═══════════════════════════════════╝"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All nodes stopped."
}
trap cleanup EXIT INT TERM

echo "╔═══════════════════════════════════════════════════╗"
echo "║          Lattice P2P Training Network             ║"
echo "╠═══════════════════════════════════════════════════╣"
printf "║  Honest nodes : %-33s║\n" "$N_NODES"
printf "║  Byzantine    : %-33s║\n" "$BYZANTINE"
printf "║  Steps        : %-33s║\n" "$STEPS"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# compute min-peers: each honest node should see at least (N-1) other honest nodes
TOTAL_NODES=$N_NODES
if [ "$BYZANTINE" = true ]; then
    TOTAL_NODES=$((N_NODES + 1))
fi
MIN_PEERS=$((TOTAL_NODES - 1))

cd "$(dirname "$0")/python"

# ── launch honest nodes ──────────────────────────────────────────
for i in $(seq 0 $((N_NODES - 1))); do
    HTTP_PORT=$((BASE_HTTP + i))
    P2P_PORT=$((BASE_P2P + i))

    # build --peers list: all previous nodes' p2p ports
    PEERS=""
    for j in $(seq 0 $((i - 1))); do
        if [ -n "$PEERS" ]; then
            PEERS="$PEERS,"
        fi
        PEERS="${PEERS}localhost:$((BASE_P2P + j))"
    done

    PEER_ARG=""
    if [ -n "$PEERS" ]; then
        PEER_ARG="--peers $PEERS"
    fi

    echo "Starting node $i  (HTTP :$HTTP_PORT  P2P :$P2P_PORT  min-peers: $MIN_PEERS  peers: ${PEERS:-none})"
    python node.py --id "$i" --port "$HTTP_PORT" --p2p-port "$P2P_PORT" $PEER_ARG --steps "$STEPS" --min-peers "$MIN_PEERS" &
    PIDS+=($!)
    sleep 0.5   # stagger startup slightly
done

# ── optionally launch Byzantine node ─────────────────────────────
if [ "$BYZANTINE" = true ]; then
    BYZ_ID=$N_NODES
    BYZ_HTTP=$((BASE_HTTP + BYZ_ID))
    BYZ_P2P=$((BASE_P2P + BYZ_ID))

    # connect to all honest nodes
    BYZ_PEERS=""
    for j in $(seq 0 $((N_NODES - 1))); do
        if [ -n "$BYZ_PEERS" ]; then
            BYZ_PEERS="$BYZ_PEERS,"
        fi
        BYZ_PEERS="${BYZ_PEERS}localhost:$((BASE_P2P + j))"
    done

    echo "Starting node $BYZ_ID  (HTTP :$BYZ_HTTP  P2P :$BYZ_P2P  BYZANTINE ☠️)"
    python node.py --id "$BYZ_ID" --port "$BYZ_HTTP" --p2p-port "$BYZ_P2P" --peers "$BYZ_PEERS" --byzantine --steps "$STEPS" --min-peers "$MIN_PEERS" &
    PIDS+=($!)
fi

echo ""
echo "All nodes launched. Press Ctrl-C to stop."
echo "Run 'python python/dashboard.py' in another terminal to monitor."
echo ""

# wait for all background processes
wait
