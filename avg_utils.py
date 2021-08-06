# Kamaleswaran Lab, Emory University
# Utilities for averaging over data points of SBP/DBP v.s. STT/PAT
# Author: Xiyang Wu, M.S. in ECE, Georgia Institute of Technology
# Date: 08/05/2021

import numpy as np

# Average the datapoints (Systolic/Diastolic BP v.s. STT) over 5 seconds (or other value defined by window_size)
def window_average(input_signal, r_peaks, window_size):
    scope_size = int(r_peaks[-1] // window_size)
    hash_table = np.zeros(scope_size)
    counter = np.zeros(scope_size)
    avg_signal = []

    for i in range(len(r_peaks)):
        if r_peaks[i] // window_size < scope_size:
            idx = int(r_peaks[i] // window_size)
            hash_table[idx] += 1
            counter[idx] += input_signal[i]

    for i in range(scope_size):
        if hash_table[i] > 0:
            avg_signal.append(counter[i] / hash_table[i])
    return np.array(avg_signal)
