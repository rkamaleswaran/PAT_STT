# Kamaleswaran Lab, Emory University
# Data Cleaning code (Error peak elimination) for PPG and ECG
# Author: Xiyang Wu, M.S. in ECE, Georgia Institute of Technology
# Date: 08/05/2021

import numpy as np
import heartpy as hp

# Elimiate the mistaken detected r-peaks. As shown in the figure above, the first
# two r-peaks are taken as mistake r-peaks (lower than the average ECG value over
# the interval between every 20 r-peaks, since the rest of the r-peaks almost have
# the same height)
def ecg_elimination(ECG, r_peaks, search_interval):
    curr = 0
    new_peaks = []
    while curr < len(r_peaks):
        # Extract the ECG signal segment for averaging
        search_start = curr
        search_end = min(curr + search_interval, len(r_peaks) - 1)

        ecg_search_start = r_peaks[search_start]
        ecg_search_end = r_peaks[search_end]
        ECG_selected = ECG[ecg_search_start:ecg_search_end]

        # Compute the average and eliminate problematic ECG peaks
        avg = np.average(ECG_selected)
        for i in range(search_start, search_end):
            if ECG[r_peaks[i]] > avg:
                new_peaks.append(r_peaks[i])

        curr += search_interval

    return np.array(new_peaks)

# Reject problematic PPG beat with HeartPy with a certain segment of PPG
# https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/
def reject_PPG(PPG_raw, segment_size):
    # Iteration number
    idx_num = len(PPG_raw) // segment_size
    test = np.zeros((segment_size,))
    test[:segment_size] = PPG_raw[:segment_size].copy()
    working_data, measures = hp.process(test, 125)
    removed_idx = working_data['removed_beats']

    # Separate the whole PPG signal into segments, concatenate the PPG peaks after rejection
    for i in range(1, idx_num):
        test = np.zeros((segment_size,))
        test[:segment_size] = PPG_raw[i * segment_size:(i + 1) * segment_size].copy()
        working_data, measures = hp.process(test, 125)
        removed_idx = np.concatenate((removed_idx, i * segment_size + working_data['removed_beats']))
    return removed_idx

# Since some normal peaks may exist in problematic PPG signal and this phenomenon may
# introduce failure in STT computation, add an extra peak rejection function
# Set a search_bound and check all peaks within this bound, reject all preaks within this bound when a removed peak is found
def r_peaks_rejection(r_peaks, removed_idx, search_bound):
    # Initialize the pointer
    ptr_removed = 0
    r_peaks_existing_idx = []
    r_peaks_rejected_idx = []
    while removed_idx[ptr_removed] < r_peaks[0]:
        ptr_removed += 1

    # Iteration number
    iter_num = len(r_peaks) // search_bound

    # Go through all removed peaks
    for i in range(iter_num - 1):
        if ptr_removed < len(removed_idx):
            if removed_idx[ptr_removed] >= r_peaks[i * search_bound] and removed_idx[ptr_removed] <= r_peaks[
                min((i + 1) * search_bound, len(r_peaks) - 1)]:
                # check all peaks within this bound, reject all preaks within this bound when a removed peak is found
                # Add removed peaks into rejected list
                for j in range(search_bound):
                    r_peaks_rejected_idx.append(i * search_bound + j)
                while removed_idx[ptr_removed] <= r_peaks[min((i + 1) * search_bound, len(r_peaks) - 1)]:
                    ptr_removed += 1
                    if ptr_removed >= len(removed_idx):
                        break
            # Otherwise, add all peaks into a new list
            else:
                for j in range(search_bound):
                    r_peaks_existing_idx.append(i * search_bound + j)
        else:
            for j in range(search_bound):
                r_peaks_existing_idx.append(i * search_bound + j)

    return r_peaks_existing_idx, r_peaks_rejected_idx


# Separate the signal into windows over 30s, if 10% r-peaks are rejected within the certain window, reject this 30s segment
def r_peaks_segment_rejection(r_peaks, r_peaks_existing_idx, r_peaks_rejected_idx, window_size, thres):
    segment_num = r_peaks[-1] // window_size + 1
    existing_table = []
    rejection_table = []

    for i in range(segment_num):
        existing_table.append([])
        rejection_table.append([])

    for i in range(len(r_peaks_existing_idx)):
        idx = r_peaks_existing_idx[i]
        existing_table[r_peaks[idx] // window_size].append(r_peaks_existing_idx[i])

    for i in range(len(r_peaks_rejected_idx)):
        idx = r_peaks_rejected_idx[i]
        rejection_table[r_peaks[idx] // window_size].append(r_peaks_rejected_idx[i])

    new_r_peaks_existing_idx = []
    for i in range(segment_num):
        total_num = len(existing_table[i]) + len(rejection_table[i])
        if total_num > 0:
            if (len(rejection_table[i]) / total_num) < thres:
                new_r_peaks_existing_idx += existing_table[i]
    return new_r_peaks_existing_idx