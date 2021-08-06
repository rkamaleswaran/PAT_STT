# Kamaleswaran Lab, Emory University
# Utilities for computing PAT and STT of PPG
# Author: Xiyang Wu, M.S. in ECE, Georgia Institute of Technology
# Date: 08/05/2021

import numpy as np

# Go through the whole PPG signal, extract the minimum and maximum points within the certain segment made by peaks detected
def ppg_min_max_locating(input_signal, r_peaks, min_window, max_window):
    min_idx = []
    max_idx = []
    # Separate the input signal into segments by peaks detected
    for i in range(len(r_peaks) - 1):
        start = r_peaks[i]
        end = r_peaks[i + 1]

        # Find out the minimum value within the min_window, start from very beginning of the segment
        min_point_temp = r_peaks[i] + np.argmin(input_signal[start:min(start + min_window, end - 1)])
        if np.isnan(min_point_temp):
            min_point_temp = start
        min_idx.append(min_point_temp)

        # Find out the maximum value within the max_window, start from the end point of min_window segment
        max_point_temp = min_point_temp + np.argmax(input_signal[min_point_temp:end])
        if np.isnan(max_point_temp):
            max_point_temp = end
        max_idx.append(max_point_temp)
        # print(min_point_temp, max_point_temp, 0)
        # print(r_peaks[i] + min_point_temp, r_peaks[i] + max_point_temp, 1)
    return max_idx, min_idx

# Determine the tangent point with differential method: Find out the point with maximum slope value between two proximate points
# Generate the tangent line on this point, compute the intersection point between it and the tangent line on the minimum point
def tangent_Comuputation(raw_signal, ppg_min_result, ppg_max_result):
    t_start = ppg_min_result
    t_end = ppg_max_result

    t_segment = np.array(range(t_start, t_end))
    y_segment = np.array(raw_signal[t_segment])

    slope = (y_segment[1:] - y_segment[:-1]) / (t_segment[1:] - t_segment[:-1])
    slope_max = np.max(slope)
    slope_max_pos_temp = np.argmax(slope)
    slope_max_t = t_start + slope_max_pos_temp
    slope_max_y = raw_signal[slope_max_t]
    slope_max_point = (slope_max_t, slope_max_y, slope_max)

    # Minimum point
    min_point_t = ppg_min_result
    min_point_y = raw_signal[ppg_min_result]

    # Intersection
    intersect_t = (min_point_y - slope_max_y) / slope_max + slope_max_t
    intersect_y = min_point_y

    return slope_max_t, slope_max_y, intersect_t, intersect_y, slope_max

# Compute STT value, which is the reciprocal of the maximum slope
def STT_compute(raw_signal, r_peaks, r_peaks_existing_idx, ppg_min_idx, ppg_max_idx, min_slope_thres,
                lower_slope_thres):
    slope_max_point_list = [[], []]
    stt_time_list = []

    r_peaks_valid = []

    # Separate the PPG signal into segments and extract the maximum slope value one by one
    for i in range(len(r_peaks) - 1):
        # start_segment = r_peaks[i]
        # end_segment = r_peaks[i + 1]

        # Ensure the minimum and maximum points locate far enough
        if (ppg_min_idx[i] + min_slope_thres) <= ppg_max_idx[i]:
            slope_max_t, slope_max_y, _, _, slope_max = \
                tangent_Comuputation(raw_signal, ppg_min_idx[i], ppg_max_idx[i])
            print(slope_max, i, ppg_min_idx[i], ppg_max_idx[i])
            # Ensure the slope is larger than the lower upper to eliminate the flat PPG signal
            if slope_max > lower_slope_thres and i in r_peaks_existing_idx:
                STT_time = 1 / slope_max

                r_peaks_valid.append(i)
                slope_max_point_list[0].append(slope_max_t)
                slope_max_point_list[1].append(slope_max_y)

                stt_time_list.append(STT_time)

    return r_peaks_valid, slope_max_point_list, stt_time_list

# Compute PAT value, which is the distance between r-peak and the intersection point
def PAT_compute(raw_signal, r_peaks, r_peaks_existing_idx, ppg_min_idx, ppg_max_idx, min_slope_thres,
                lower_slope_thres):
    slope_max_point_list = [[], []]
    intersection_point_list = [[], []]
    pat_time_list = []

    r_peaks_valid = []

    # Separate the PPG signal into segments and extract the maximum slope value one by one
    for i in range(len(r_peaks) - 1):
        # Ensure the minimum and maximum points locate far enough
        if (ppg_min_idx[i] + min_slope_thres) <= ppg_max_idx[i]:
            slope_max_t, slope_max_y, intersect_t, intersect_y, slope_max = \
                tangent_Comuputation(raw_signal, ppg_min_idx[i], ppg_max_idx[i])

            pat_time = intersect_t - r_peaks[i]

            # Ensure 0 < PAT < 120
            if pat_time > 0 and pat_time < 120 and (i in r_peaks_existing_idx):
                r_peaks_valid.append(i)
                slope_max_point_list[0].append(slope_max_t)
                slope_max_point_list[1].append(slope_max_y)
                intersection_point_list[0].append(intersect_t)
                intersection_point_list[1].append(intersect_y)

                pat_time_list.append(pat_time)

    return r_peaks_valid, slope_max_point_list, intersection_point_list, pat_time_list