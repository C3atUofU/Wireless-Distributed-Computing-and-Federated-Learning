import time

kill = False


def delay(seconds, data_que, data, priority):
    while not kill:
        time.sleep(seconds)
        data_que.put((priority, data))


def get_kill():
    return kill

