# Kamaleswaran Lab, Emory University
# Utilities for computing SBP/DBP for ABP
# Author: Xiyang Wu, M.S. in ECE, Georgia Institute of Technology
# Date: 08/05/2021

import numpy as np

# Go through the whole ABP signal, extract the minimum and maximum points within the certain segment made by peaks detected
def abp_max_min_locating(input_signal, r_peaks, min_window, max_window):
    min_idx = []
    max_idx = []
    for i in range(len(r_peaks) - 1):
        start = r_peaks[i]
        end = r_peaks[i + 1]

        # Find out the minimum value within the min_window, start from very beginning of the segment
        min_point_temp = start + np.argmin(input_signal[start:start + min_window])
        min_idx.append(min_point_temp)

        # Find out the maximum value within the max_window, start from the end point of min_window segment
        max_point_temp = min_point_temp + np.argmax(input_signal[min_point_temp:min_point_temp + max_window])
        max_idx.append(max_point_temp)

    return max_idx, min_idx

# Only select Systolic_BP > 100 and 0 < Diastolic_BP < 100
def BP_elimination(Systolic_BP, Diastolic_BP):
    BP_valid_idx = []
    for i in range(len(Systolic_BP)):
        if Systolic_BP[i] > 100 and Diastolic_BP[i] < 100 and Diastolic_BP[i] > 0:
            BP_valid_idx.append(i)
    return np.array(BP_valid_idx)