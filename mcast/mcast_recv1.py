"""
@author: mcrabtre
"""
import socket
import struct
import numpy as np
import queue
import sys


kill = False
prev = 0


def m_recv(node, stage, q, priority, point_size=785):
    global prev
    while not kill:
        if stage == 1:
            ip_switcher = {
                1: '224.3.29.71',
                2: '224.3.29.72',
                3: '224.3.29.73',
                4: '224.3.29.74',
                5: '224.3.29.75',
            }
            port_switcher = {
                1: 5001,
                2: 5002,
                3: 5003,
                4: 5004,
                5: 5005,
            }
        else:
            ip_switcher = {
                1: '224.3.29.76',
                2: '224.3.29.77',
                3: '224.3.29.78',
                4: '224.3.29.79',
                5: '224.3.29.80',
            }
            port_switcher = {
                1: 5006,
                2: 5007,
                3: 5008,
                4: 5009,
                5: 5010,
            }
        MCAST_GRP = ip_switcher.get(node, '0.0.0.0')
        MCAST_PORT = port_switcher.get(node, 0000)
        server_addr = ('', MCAST_PORT)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        #print('bound to ', server_addr)
        group = socket.inet_aton(MCAST_GRP)
        mreq = struct.pack("=4sl", group, socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 81920)  # max buff size for pi /proc/sys/net/core/rmem_max
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # toggle reuseaddr option TRUE to avoid addr in use conflicts
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_TTL, 20)   # allows for cleaner mcasting but not strictly necessary
        sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_LOOP, 1)
        buff = 1470  # max byte value per recv
        sock.bind(server_addr)
        recvd = b''
        recv_q = queue.PriorityQueue()
        load_symbols = ('.', '. .', '. . .')
        while True:
            part = sock.recv(buff)
            head = int.from_bytes(part[0:4], 'big')
            sys.stdout.write('\rReceiving '+load_symbols[head % 3])
            sys.stdout.flush()
            recv_q.put((head, part[4:])) # use of priority queue to sort out of order data-grams
            if len(part) < buff:
                break
        qsize = recv_q.qsize()
        while not recv_q.empty():
            a = recv_q.get()
            sys.stdout.write('\rQueuing Data ' + str(int(100*a[0]/qsize)) + '%')
            sys.stdout.flush()
            recvd = recvd + a[1]
        data_recvd = np.frombuffer(recvd, dtype='uint8')
        loss = len(data_recvd) % point_size
        if loss:
            print('missing ', point_size - loss, ' bytes of data, attempting to recover')
            patch = np.zeros((point_size-loss,), dtype='uint8')
            data_recvd = np.concatenate((data_recvd, patch))
        rec = np.reshape(data_recvd, (int(len(data_recvd)/point_size), point_size))

        if not(np.array_equal(rec, prev)):
            print('received ', len(recvd), ' bytes of data from node ', node, ' stage ', stage, ' priority ', priority)
            q.put((priority, rec))
            prev = rec
        sock.close()


