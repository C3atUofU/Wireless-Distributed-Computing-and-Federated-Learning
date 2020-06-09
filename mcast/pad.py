
"""
Created 18FEB2020 by mcrabtre
"""
import numpy as np


def pad(data, pad_value=1):
    data_padded = data
    for i in range(pad_value-1):
        data_padded = np.concatenate((data_padded, data))
    return data_padded


def unpad(padded_data, file_size):
    data = padded_data[0:file_size]
    return data

