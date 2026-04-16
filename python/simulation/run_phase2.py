from multiprocessing import Process, Queue
import sys
sys.path.insert(0, '../training/')
from node import run_node

if __name__ == '__main__':
    n_nodes = 3
    # each node has an inbox queue; it sends to other nodes' inboxes
    queues = [Queue() for _ in range(n_nodes)]
    
    processes = []
    for i in range(n_nodes):
        # node i sends to all queues except its own
        send_qs = [q for j, q in enumerate(queues) if j != i]
        recv_q = queues[i]
        p = Process(target=run_node, args=(i, send_qs, recv_q))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("all nodes converged — check loss curves above")

