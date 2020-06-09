# -*- coding: utf-8 -*-
"""
Created 15OCT2019 by mcrabtre
"""

import queue
import SVM
import node_client
import node_server
import os
import encode_manager as e
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
    pad_value = 1
    mcast_recv1.kill = False
    n = e.cycle5(int(nodeIP[-1]) + 1)
    e.node = n
    recv_nodes = e.node_tracker()
    recv_stages = (1, 1, 2)
    threads = {}
    cache_q = queue.Queue()  # use of the fifo queue structure ensures safe exchange of data between threads
    k = 1  # global counter
    for i in range(3):
        # create and start separate threads to receive from different nodes (node, stage, q, priority):
        threads[i] = threading.Thread(target=mcast_recv1.m_recv, args=(recv_nodes[i], recv_stages[i], cache_q, i + 1), daemon=False)
        threads[i].start()
        print('starting thread ', i + 1)

    while True:  # main running loop


        print('I am ', nodeIP)
        print('Waiting for Agg')
        # listen for info from aggregator.
        # determine node number (n) and total number of nodes (Num) from agg Xmission
        # N is list of node
        # win,tau,k,d,N,aggIP = node_server.server(node_ip)
        while True:
            info = node_server.server(nodeIP)
            if info == 'resend':
                print('## RESENDING PROTOCOL ##')
                s1_data = pad.pad(e.stage1_send(), pad_value)
                s2_data = pad.pad(e.stage2_send(), pad_value)
                mcast_send.send(n, 1, s1_data)  # send stage1 partition
                mcast_send.send(n, 2, s2_data)  # send stage2 partition
            else:
                break
        if info == 'restart':
            print('***Restarting Program***')
            break
        # n = info.node_dict[nodeIP] + 1  # node number (1,2,3,4,5)
        print('My node number is ', n)
        Num = len(info.node_dict)  # total number of nodes (should be 5)
        d = info.d  # number data points/node
        tau = info.tau  # number of local iterations
        k = info.k  # global iteration number
        K = info.K # total global iterations
        pad_value = info.pad_value # padding value of data for testing
        if k == 0:
            print('beginning session')
            e.set_flag = False
            mcast_recv1.prev = 0
        if not cache_q.empty(): #
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
        cpart = 1
        time_init = time.time()
        ans_data = [(0, 'no data'), (0, 'no data'), (0, 'no data')]
        # wait for other nodes to start receiving
        # time.sleep(0.1*(5-n))

        while shuffcount <= shuff:  # main coded shuffling running loop (for shuff iterations)
            s1_data = pad.pad(e.stage1_send(), pad_value)
            s2_data = pad.pad(e.stage2_send(), pad_value)
            mcast_send.send(n, 1, s1_data)  # send stage1 partition
            mcast_send.send(n, 2, s2_data)  # send stage2 partition
            data_pts = e.get_work_file()  # data points for current iteration to use for training
            print('training')
            data_pts = data_pts.astype('float64')
            w, fn = SVM.svm(w, data_pts, eta, lam, tau, d, weight, shuff=0, N=Num, n=n - 1) #train on tau iterations data_pts d
            try_time = time.time()
            while cpart < 4:
                cp = cpart  # need this to ensure fifo queue is completely cycled before searching for the next part
                for i in range(cache_q.qsize()):  # cycle through the queue for receive data
                    a = cache_q.get()  # tuple in form (part integer, DATA)
                    if a[0] == cp:  # if it is the part needed adds part to ans_data and increments to next part
                        cp = 'disabled'
                        ans_data[cpart - 1] = a
                        print('got part ', cpart)
                        cpart = cpart + 1
                    else:
                        cache_q.put(a)
                #if time.time() >= try_time + 5:  # resend if it doesnt find data after 5 seconds
                  #  print("Resending, couldn't find part ", cpart)
                   # break
            if time.time() >= (time_init + 5*60):  # times out 5*60s of not receiving all 3 pieces of data
                print("Error: threads timed out")
                exit(-1)
            if cpart >= 4:  # receive condition triggers when all data is ready
                print('data received')
                e.recv(pad.unpad(ans_data[0][1], int(d/4)),
                       pad.unpad(ans_data[1][1], int(d/4)),
                       pad.unpad(ans_data[2][1], int(d/4)))
                cpart = 1
                time_init = time.time()
                shuffcount = shuffcount + 1
        # send w to agg
        print('sending cleanup data')
        s1_data = pad.pad(e.stage1_send(), pad_value)
        s2_data = pad.pad(e.stage2_send(), pad_value)
        mcast_send.send(n, 1, s1_data)  # send stage1 partition
        mcast_send.send(n, 2, s2_data)  # send stage2 partition
        node_client.client(w, fn, host)
        k = k+1

    if not mcast_recv1.kill:  # kills threads by ending underlying function(s)
        mcast_recv1.kill = True
        print('Threads killed')

while True:
    run()
    time.sleep(1)
