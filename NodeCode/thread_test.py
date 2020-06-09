import threading
import queue
import mcast_recv1


def test():
    threads = {}
    recv_nodes = [1, 1, 2]
    recv_stages = [1, 2, 1]
    cache_q = queue.Queue()
    recv_msg = 0
    for i in range(3):
        # create and start separate threads to receive from different nodes (node, stage, q, priority):
        threads[i] = threading.Thread(target=mcast_recv1.m_recv, args=(recv_nodes[i], recv_stages[i], cache_q, i + 1), daemon=False)
        threads[i].start()
        print('starting thread ', i+1)
    while recv_msg <= 5:
        for i in range(cache_q.qsize()):
            a = cache_q.get()
            print('Data received')
            print(' ')
            print(a)
            recv_msg = recv_msg + 1
    print('Script Finished')
    mcast_recv1.kill = True

