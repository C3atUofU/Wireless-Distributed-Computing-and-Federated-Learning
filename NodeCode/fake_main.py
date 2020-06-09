from NodeCode import delay
import threading
import queue
import random
import time


def run(itr=5):
    cache_q = queue.Queue() #use of the queue structure ensures safe exchange of data between threads
    icount = 1 #compare to iterations
    data = ['first', 'second', 'third'] #just needed for delay fn
    threads = {}
    cpart = 1 #this variable keeps track of which data part is needed next for recieve, there are 3 needed for each recv
    time_init = time.time()
    ans_data = [(0, 'data'), (0, 'data'), (0, 'data')]
    for i in range(3):
        sec = random.randint(1, 3) # for delay fn
        # create and start separate threads to receive from different nodes
        threads[i] = threading.Thread(target=delay.delay, args=(sec, cache_q, data[i], i + 1), daemon=False)
        threads[i].start()
        print('starting thread ', i)
    while icount <= itr: # main running loop
        if cpart >= 4: # receive condition triggers when all data is ready
            print(ans_data[0][1], ans_data[1][1], ans_data[2][1])
            cpart = 1
            time_init = time.time()
            icount = icount + 1
        for i in range(cache_q.qsize()): # cycle through the queue for receive data
            a = cache_q.get()
            if a[0] == cpart:
                ans_data[cpart - 1] = a
                cpart = cpart + 1
                break
            else:
                cache_q.put(a)

        if time.time() >= (time_init + 60): # times out 60s of not recieving all 3 pieces of data
            print("Error: treads timed out")
            break
        else: # not necessary, will delete later
            print('queue not full', threading.active_count(), cache_q.qsize())
            time.sleep(0.5)
    if not delay.get_kill(): # kills threads by ending underlying function(s)
        delay.kill = True
        print('Threads killed')


run()

