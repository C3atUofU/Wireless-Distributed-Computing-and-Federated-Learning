"""
Created 15AUG2019 by mcrabtre
"""
import pandas as pd
import numpy as np

'''
    **data partitions**
    This module manages data partitions and subpartitions for encoded shuffling including 5 nodes.
    Total data size must be some multiple of 20 data pts. The data set is divided into 5 distinct files. 
    each node has 1 file in addition to 1/4 of each other nodes file to make a total of 2 file sizes
    per node. This data is stored in a work.csv file in the encode_manager directory, which is routinely 
    read from and rewritten as data shuffling occurs. 
    
    **encoding/decoding**
    encoding is a bitwise XOR operation on two file partitions, this results in an encoded partition 
    decoded is a bitwise XOR operation on the encoded partition and one of its original file partitions.
    This enables the data of both partitions to be contained within the encoded, a partitions data is 
    recovered using the other partition it was encoded to.
    
    **shuffling stages**
    shuffling takes place in two stages, the encode send stage, and the cleanup stage. The encode send stage 
    encodes two partitions and sends it, eventually this will be received by two other nodes. Stage 2 is a 
    single non-coded partition. These two stages were necessary for the entire data set to be shuffled among the nodes
    each iteration. 
    
    **receiving**
    this manager completely automates the receiving process, it takes in 3 different partitions. Stage1_m is the 
    coded partition received from the minor node, or the node wih the next successive node number. Stage1_M is the 
    coded partition received from the major node, or the node 2 successive node numbers away from the receiving node.
    The recv function automates the process of decoding and reordering the data in order to be used for training and 
    successive sends. This is done through a number of mutation arrays that track the location of each data piece.
    These arrays are mutated each shuffling iteration indicated by global variable itr. The use of cycle4 and cycle5
    functions (identical to mod4 and mod5) is to simplify the process of 
'''
data_size = 20
node = 0
itr = 1
set_flag = False

def create_workspace(nodeNum,filePath,totalDataSize): #creates a headerless workspace file containing working file and partitions
    if(totalDataSize % 20) != 0: #check that total data size is a multiple of 20
        if totalDataSize < 20:
            totalDataSize = 20
        else:
            totalDataSize = totalDataSize - (totalDataSize % 20)

    global data_size #assign to global variables for use by other functions
    data_size = totalDataSize
    global node
    node = nodeNum
    global itr
    itr = 1
    global set_flag
    set_flag = True
    otherNodes = [1,2,3,4,5]
    otherNodes.remove(nodeNum)
    fileSize = int(totalDataSize/5)
    f = pd.read_csv(filePath, dtype='uint8', skiprows=int((nodeNum - 1) * fileSize), nrows=fileSize).values # write working file from traning data into ndarray f
    for x in otherNodes: #get partitions of other nodes files
        if x < nodeNum:
            p = pd.read_csv(filePath, dtype='uint8', skiprows=int(((x-1)*fileSize)+((nodeNum-2)/4)*fileSize), nrows=fileSize/4).values
        else:
            p = pd.read_csv(filePath, dtype='uint8', skiprows=int(((x-1)*fileSize)+((nodeNum-1)/4)*fileSize), nrows=fileSize/4).values
        f = np.concatenate((f, p), axis=0)
    np.savetxt('work' + str(node) + '.csv', f, '%i', delimiter=",") #creates, and saves f into work.csv (in same directory as encode_manager.py)
    #return f


def get_work_file(): #returns current training data for the node. (dependant on shuffle iteration)
    global data_size
    file_size = data_size/5
    return pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=0, nrows=file_size).values


def get_itr(): #returns current iteration
    global itr
    return itr


def stage1_send(): #returns encoded data to send stage1
    global data_size
    global node
    global itr
    part_mutation = [4,1,2,3,4]
    file_mutation = [3,4,1,2,3]
    for i in range(1,itr):
        part_mutation.insert(4, part_mutation.pop(0))
        file_mutation[5-cycle5(i)] = cycle4(file_mutation[5-cycle5(i)]-1) #i call this an end - i mutation
    part_size = int(data_size/20)
    p1 = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=(file_mutation[node-1]-1)*part_size, nrows=part_size).values
    p2 = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=4*part_size + (part_mutation[node-1]-1)*part_size, nrows=part_size).values
    #print('node ', node, ' itr ', itr, ' stage1 is ', p1, '^', p2)
    data_send = encode(p1,p2)
    return data_send


def stage2_send(): #returns data to send for stage 2
    global data_size
    global node
    global itr
    file_mutation = [2,3,4,1,2]
    for i in range(1,itr):
        file_mutation[5-cycle5(i)] = cycle4(file_mutation[5-cycle5(i)]-1)
    part_size = int(data_size / 20)
    p1 = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=(file_mutation[node-1]-1)*part_size, nrows=part_size).values
    #print('node ', node, ' itr ', itr, ' stage2 is ', p1)
    return p1


def recv(stage1_m, stage1_M, stage2):
    global data_size
    global node
    global itr
    part_size = int(data_size / 20)
    decodem_mut = [1,2,3,4,1] #end - i
    decodeM_mut = [2,3,4,1,2] #shift left
    wret_part_mut = [1,2,3,4,1] #sl
    wret_file_mut = [1,2,3,4,4] #e-i must be offset by 1 iteration retained into wf
    ret_file_mut = [4,1,2,3,4] #e-i retained from wf
    ret_part_mut = [1,2,3,4,5] #sl position for retained from wf
    for i in range(1, itr): # this loop mutates each index for current iteration
        decodem_mut[5 - cycle5(i)] = cycle4(decodem_mut[5 - cycle5(i)] - 1)
        wret_file_mut[5 - cycle5(i+1)] = cycle4(wret_file_mut[5 - cycle5(i+1)] - 1)
        ret_file_mut[5 - cycle5(i)] = cycle4(ret_file_mut[5 - cycle5(i)] - 1)
        decodeM_mut.insert(4, decodeM_mut.pop(0))
        wret_part_mut.insert(4, wret_part_mut.pop(0))
        ret_part_mut.insert(4, ret_part_mut.pop(0))
    # the below file accesses are for each part of the new stored data (see line 131)
    data_file_size = 2*(data_size / 5)
    data_file = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=0, nrows=data_file_size).values
    # changed to single file access 2/25/2020
    #decM = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=(4*part_size)+(decodeM_mut[node-1]-1)*part_size, nrows=part_size).values
    decM = data_file[(4*part_size)+(decodeM_mut[node-1]-1)*part_size:(4*part_size)+(decodeM_mut[node-1])*part_size, :]
    #decm = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=(decodem_mut[node-1]-1)*part_size, nrows=part_size).values
    decm = data_file[(decodem_mut[node-1]-1)*part_size:(decodem_mut[node-1])*part_size, :]
    #R = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=(4*part_size)+(wret_part_mut[node-1]-1)*part_size, nrows=part_size).values
    R = data_file[(4*part_size)+(wret_part_mut[node-1]-1)*part_size:(4*part_size)+(wret_part_mut[node-1])*part_size, :]
    #print('node ', node, ' itr ', itr, ' R is ', R)
    if ret_part_mut[node-1] == 5: #bandaid bug fix :(
        #pre_ret = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=(5*part_size), nrows=(3*part_size)).values
        pre_ret = data_file[(5*part_size):(8*part_size), :]
    else:
        #pre_ret = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=(4*part_size), nrows=part_size*(ret_part_mut[node-1]-1)).values
        pre_ret = data_file[(4*part_size):part_size*(ret_part_mut[node-1]+3), :]
    #ret = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=(ret_file_mut[node-1]-1)*part_size, nrows=part_size).values
    ret = data_file[(ret_file_mut[node-1]-1)*part_size:(ret_file_mut[node-1])*part_size, :]
    if ret_part_mut[node-1] < 4: #bandaid bug fix :-(
        #post_ret = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows= part_size*(4+ret_part_mut[node-1]), nrows=part_size*(4 - ret_part_mut[node-1])).values
        post_ret = data_file[part_size*(4+ret_part_mut[node-1]):(part_size*8), :]
    else:
        #post_ret = pd.read_csv('work' + str(node) + '.csv', header=None, dtype='uint8', skiprows=0, nrows=0).values
        post_ret = data_file[0:0, :]
    #print('node ', node, ' itr ', itr, ' pre ret is ', pre_ret, ' ret is ', ret, ' post ret is ', post_ret)
    S1M = decode(decM, stage1_M) # decodes data recieved from the Major (furthest away) node
    S1m = decode(decm, stage1_m) # decodes data recieved from the minor (closest) node
    wf = np.concatenate((R, S1M, stage2, S1m), axis=0)
    # print('wf shape ', wf.shape)
    for i in range(part_size*1, part_size*(6 - wret_file_mut[node-1])): # this loop reorders the work file for current mutation
        wf = np.append(wf, [wf[0]], 0)
        wf = np.delete(wf, 0, 0)
    wf = np.concatenate((wf, pre_ret, ret, post_ret), axis=0)
    np.savetxt('work' + str(node) + '.csv', wf, '%i', delimiter=",")  # saves new wf into work.csv
    itr = itr + 1
    #return wf


def node_tracker():
    global node
    recv_nodes = (cycle5(node+1), cycle5(node+2), cycle5(node+1)) #(s1minor, s1Major, s2)
    return recv_nodes


def cycle5(value): # for ease of iterative counting
    if value > 0:
        if value <= 5:
            return value
        else:
            return cycle5(value-5)
    else:
        return cycle5(value+5)


def cycle4(value): # for ease of iterative counting
    if value > 0:
        if value <= 4:
            return value
        else:
            return cycle5(value-4)
    else:
        return cycle5(value+4)


def encode(part1, part2): #bitwise XOR encode
    #newPart = np.zeros(shape= part1.shape, dtype= np.uint8)
    if part1.shape != part2.shape:
        raise Exception('Error... both partitions must be of equal dimensions')
    newPart = part1 ^ part2
    return newPart


def decode(key, part): #partitions are decoded in the same way they are encoded
    new = encode(key, part)
    return new


#create_workspace(1,'test.csv',20)
'''
# misc testing code
create_workspace(4, 'test.csv', 40)
print(cycle5(27))
p1 = pd.read_csv('test.csv', dtype='uint8', skiprows=15, nrows=2).values
print('p1 is ',p1)

p2 = pd.read_csv('test.csv', dtype='uint8', skiprows=0, nrows=2).values
print('p2 is ',p2)

enc = encode(p1,p2)
print('enc is ',enc)
dec1 = decode(p2,enc)
print('dec1 is ',dec1)
print('enc is ',enc)
print('p1 is ',p1)
print('p2 is ',p2)


dec2 = decode(p1,enc)
print('dec2 is ',dec2)
print('enc is ',enc)
print('p1 is ',p1)
print('p2 is ',p2)

wf = get_work_file()
print('wf is ',wf)

print('dec1 = p1 is ', (dec1 == p1))
print('dec2 = p2 is ', (dec2 == p2))
'''
