# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 09:59:11 2019

@author: mcrabtre
"""


import socket
import time


def send(node, stage, data):
    print("sending no rest ")
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
    data = data  # data has properties .w .nodenum .tau .k
    #= b'0xff' + y
    #need to tune send buffers
    datasend = data.tobytes() #abandoned pickle serialization 3/18/2020
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
    buff = 1466

    if len(datasend) > buff:  # break large send data into buff sized pieces to send individually
        pcs = int(datasend.__len__()/buff)
        for i in range(pcs):
            head = i.to_bytes(4, 'big')
            sock.sendto(head + datasend[i*buff:(i+1)*buff], (MCAST_GRP, MCAST_PORT))
            # time.sleep(0.01) #gives socket time to recv
        head = pcs.to_bytes(4, 'big')
        sock.sendto(head + datasend[pcs*buff:datasend.__len__()], (MCAST_GRP, MCAST_PORT))
    else:
        head = int(0).to_bytes(4, 'big')
        sock.sendto(head + datasend, (MCAST_GRP, MCAST_PORT))
    print("sent stage ", stage)
    sock.close()



#source: https://stackoverflow.com/questions/603852/how-do-you-udp-multicast-in-python


