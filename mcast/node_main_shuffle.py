# -*- coding: utf-8 -*-
"""
Created 15OCT2019 by mcrabtre
"""

import queue
import SVM
import node_client
import node_server
import os
import data_manager as e
import threading
import time
import mcast_recv1
import mcast_send
import pad
'''
    Main for the nodes. Starts and waits for data from agg. once received it unpacks data and runs the coded shuffle
    svm. It then sends the w vector and loss functions back to the agg.
'''


def run():
    os.system('hostname -I > output.txt')  # these 6 commands reads ip address from linux OS
    ip_file = open('output.txt', 'r')
    n_ip = ip_file.readline()
    nodeIP = n_ip.rstrip(' \n')
    os.remove('output.txt')

    # the nodes need to start the receiving threads at the very beginning and keep them running to minimize data loss with
    # the UDP sockets.
    mcast_recv1.kill = False
    n = e.cycle5(int(nodeIP[-1]) + 1)  # extrapolates node number from IP address
    e.node = n
    recv_nodes = e.node_tracker()
    recv_stages = 1
    threads = {}
    cache_q = queue.Queue()  # use of the fifo queue structure ensures safe exchange of data between threads
    K = 1  # total global
    k = 0  # global iteration
    for i in range(1):  # single casting only requires one receiving thread
        # create and start separate threads to receive from different nodes (node, stage, q, priority):
        threads[i] = threading.Thread(target=mcast_recv1.m_recv, args=(recv_nodes, recv_stages, cache_q, i + 1),
                                      daemon=False)
        threads[i].start()
        print('starting thread ', i + 1)

    while True:  # main running loop

        print('I am ', nodeIP)
        print('Waiting for Agg')
        # listen for info from aggregator.
        # determine node number (n) and total number of nodes (Num) from agg Xmission
        # N is list of node
        # win,tau,k,d,N,aggIP = node_server.server(node_ip)
        info = node_server.server(
            nodeIP)  # data is received from AGG using TCP socket to allow it to work with windows OS.

        # n = info.node_dict[nodeIP] + 1  # node number (1,2,3,4,5)
        print('My node number is ', n)
        Num = len(info.node_dict)  # total number of nodes (should be 5)
        d = info.d  # number data points/node
        tau = info.tau  # number of local iterations
        k = info.k  # global iteration number
        K = info.K  # total global iterations
        pad_value = info.pad_value  # padding value of data for testing
        if k == 0:  # initialization for a new training cycle
            print('beginning session')
            e.set_flag = False
            mcast_recv1.prev = 0
            if not cache_q.empty():
                for i in range(cache_q.qsize()):
                    cache_q.get()
        host = info.host  # aggregator IP
        shuff = info.shuff  # number of shuffles per global iteration

        # these can be pulled from the agg if desired 
        weight = 1
        eta = .01
        lam = 1
        
        # pull data_pts for global update
        w = info.w
        D = d*Num# Total data points 
        filepath = r"train.csv"
        if not e.set_flag:
            e.create_workspace(n, filepath, D)  # initialization of the workspace file
        shuffcount = 1  # compare to iterations
        # this variable keeps track of which data part is needed next for receive, there are 3 needed for each recv
        time_init = time.time()
        # wait for other nodes to start receiving
        # time.sleep(0.1*(5-n))

        while shuffcount <= shuff:  # main coded shuffling running loop (for shuff iterations)
            for i in range(1):
                send_data = pad.pad(e.get_work_file(), pad_value)
                mcast_send.send(n, 1, send_data)  # send data to other node
            data_pts = e.get_work_file()  # data points for current iteration to use for training
            print('training')
            data_pts = data_pts.astype('float64')
            w, fn = SVM.svm(w, data_pts, eta, lam, tau, d, weight, shuff=0, N=Num, n=n - 1) #train on tau iterations data_pts d
            try_time = time.time()
            receive_flag = False
            while not receive_flag:
                if not cache_q.qsize() == 0:
                    a = cache_q.get()  # tuple in form a = (priority integer, DATA)
                    e.recv(pad.unpad(a[1], d))
                    time_init = time.time()
                    receive_flag = True
                    shuffcount = shuffcount + 1
                #if time.time() >= try_time + 5:  # re-sends after not receiving for 5s
                 #   print('Resending could not assert data')
                  #  break

            if time.time() >= (time_init + 180):  # times out 180s of not receiving data
                print("Error: threads timed out")
                break
        # send w to agg
        node_client.client(w, fn, host)
        k = k+1

    if not mcast_recv1.kill:  # kills threads by ending underlying function(s)
        mcast_recv1.kill = True
        print('Threads killed')


while True:
    run()
    time.sleep(1)
