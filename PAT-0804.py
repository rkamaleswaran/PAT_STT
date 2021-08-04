# Kamaleswaran Lab, Emory University
# PAT computation code for PPG and ECG, logistic regression between PAT and Systolic/Diastolic BP from ABP
# Author: Xiyang Wu, M.S. in ECE, Georgia Institute of Technology
# Date: 08/04/2021


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy

from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 

import glob

import os
import re
import wfdb
import dippykit as dip

import tensorflow as tf
from ecgdetectors import Detectors
import heartpy as hp

from scipy import signal


# In[2]:


# Data Loader
data = pd.read_csv("dataRaw_3000393.csv")


# In[3]:


# Data format checker
data


# In[4]:


ECG = data['ECG'][:-2000]
ABP = data['ABP'][:-2000]
PPG = data['PPG'][:-2000]


# In[5]:


ABP


# In[6]:


step_size = 2000
data_selected = 3000


# In[7]:


new_ECG = ECG[:step_size * data_selected]
new_ABP = ABP[:step_size * data_selected]
new_PPG = PPG[:step_size * data_selected]
print(len(new_ECG))


# In[8]:


new_PPG[np.isnan(new_PPG)] = 0
new_ABP[np.isnan(new_ABP)] = 0
new_ECG[np.isnan(new_ECG)] = 0


# In[9]:


# Extract r-packs with ecgdetectors package, while the frquency of imported signal is 125Hz
# https://github.com/berndporr/py-ecg-detectors
detectors = Detectors(125)
r_peaks = detectors.engzee_detector(np.array(new_ECG))


# In[10]:


print(r_peaks)


# In[11]:


# Visualize ECG signal and r-peaks extracted
plt.figure(figsize = (20,10))
plt.plot(new_ECG[:step_size * 2],'b')
plt.plot(r_peaks[:40], new_ECG[r_peaks[:40]],'ro')


# In[12]:


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


# In[13]:


new_r_peaks = ecg_elimination(new_ECG, r_peaks, 20)


# In[14]:


new_r_peaks = new_r_peaks[2:]


# In[15]:


# Visualize the eliminated r-peaks (Red)
plt.figure(figsize = (20,10))
plt.plot(new_ECG[:step_size * 2],'b')
plt.plot(new_r_peaks[:50], new_ECG[new_r_peaks[:50]],'ro')


# In[16]:


# Visualize PPG signal to be analyzed
plt.figure(figsize = (20,10))
plt.plot(new_PPG[new_r_peaks[4]:new_r_peaks[5]],'b')


# In[17]:


# Reject problematic PPG beat with HeartPy with a certain segment of PPG
# https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/
def reject_PPG(PPG_raw, segment_size):  
    # Iteration number
    idx_num = len(PPG_raw) // segment_size
    test = np.zeros((segment_size, ))
    test[:segment_size] = new_PPG[:segment_size].copy()
    working_data, measures = hp.process(test, 125)
    removed_idx = working_data['removed_beats']
    
    # Separate the whole PPG signal into segments, concatenate the PPG peaks after rejection
    for i in range(1, idx_num):
        test = np.zeros((segment_size, ))
        test[:segment_size] = new_PPG[i * segment_size:(i + 1) * segment_size].copy()
        working_data, measures = hp.process(test, 125)
        removed_idx = np.concatenate((removed_idx, i * segment_size + working_data['removed_beats']))
    return removed_idx


# In[21]:


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
            if removed_idx[ptr_removed] >= r_peaks[i * search_bound] and removed_idx[ptr_removed] <= r_peaks[min((i + 1) * search_bound, len(r_peaks) - 1)]:
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


# In[22]:


# Set the PPG segment size for rejection
rejection_segment_size = 200000

# Reject PPG peaks with HeartPy
ppg_removed_idx = reject_PPG(new_PPG, rejection_segment_size)


# In[23]:


# Further reject PPG peaks 
search_bound = 10
r_peaks_existing_idx, r_peaks_rejected_idx = r_peaks_rejection(new_r_peaks, ppg_removed_idx, search_bound)


# In[24]:


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


# In[25]:


frequency = 125
window_size = frequency * 30
rejection_window_size = frequency * 30
rejection_thres = 0.1


# In[26]:


print(len(r_peaks_existing_idx))
print(len(r_peaks_rejected_idx))


# In[27]:


new_r_peaks_existing_idx = r_peaks_segment_rejection(new_r_peaks, r_peaks_existing_idx, r_peaks_rejected_idx, rejection_window_size, rejection_thres)


# In[28]:


print(len(new_r_peaks_existing_idx))


# In[29]:


# Go through the whole PPG signal, extract the minimum and maximum points within the certain segment made by peaks detected
def ppg_min_max_locating(input_signal, r_peaks, min_window, max_window):
    min_idx = []
    max_idx = []
    # Separate the input signal into segments by peaks detected
    for i in range(len(r_peaks) - 1):
        start = r_peaks[i]
        end = r_peaks[i + 1]
        
        # Find out the minimum value within the min_window, start from very beginning of the segment
        min_point_temp = np.argmin(input_signal[start:min(start + min_window, end - 1)])
        if np.isnan(min_point_temp):
            min_point_temp = start
        min_idx.append(min_point_temp)
        
        # Find out the maximum value within the max_window, start from the end point of min_window segment
        max_point_temp = np.argmax(input_signal[min_point_temp:end])
        if np.isnan(max_point_temp):
            max_point_temp = end
        max_idx.append(max_point_temp)
        print(i)
        
    return max_idx, min_idx


# In[30]:


# Extract the minimum and maximum points from PPG
ppg_fitting_size = step_size * 2
min_window = 80
max_window = 40
ppg_max_idx, ppg_min_idx = ppg_min_max_locating(new_PPG, new_r_peaks, min_window, max_window)
print(ppg_max_idx, ppg_min_idx)


# In[31]:


# Visualize the ECG and PPG signal after coordination
plt.figure(figsize = (20,10))
plt.plot(new_ECG[:200] * 4,'r')
plt.plot(new_PPG[:200],'b')


# In[32]:


# Visualize the extracted maximum (Red) and minimum (Green) points of PPG 
plt.figure(figsize = (20,10))
plt.plot(new_PPG[:100 * 100],'b')
plt.plot(ppg_max_idx[:100], new_PPG[ppg_max_idx[:100]],'ro')
plt.plot(ppg_min_idx[:100], new_PPG[ppg_min_idx[:100]],'go')


# In[33]:


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


# In[35]:


# Find out the point with maximum slope
slope_max_t, slope_max_y, intersect_t, intersect_y, slope_max = tangent_Comuputation(new_PPG, ppg_min_idx[442], ppg_max_idx[442])


# In[36]:


# Visualize the PPG, maximum slope point (Green), minimum point (Red) and intersection point (Blue)
plt.figure(figsize = (20,10))
plt.plot(new_PPG[48000:50000],'b')
plt.plot(ppg_min_idx[442], new_PPG[ppg_min_idx[442]],'ro')
plt.plot(slope_max_t, slope_max_y,'go')
plt.plot(intersect_t, intersect_y,'bo')


# In[37]:


# Compute PAT value, which is the distance between r-peak and the intersection point
def PAT_compute(raw_signal, r_peaks, r_peaks_existing_idx, ppg_min_idx, ppg_max_idx, min_slope_thres, lower_slope_thres):
    
    slope_max_point_list = [[], []]
    intersection_point_list = [[], []]
    pat_time_list = []
    
    r_peaks_valid = []
    
    # Separate the PPG signal into segments and extract the maximum slope value one by one
    for i in range(len(r_peaks) - 1):
        start_segment = r_peaks[i]
        end_segment = r_peaks[i + 1]
        
        # Ensure the minimum and maximum points locate far enough
        if (ppg_min_idx[i] + min_slope_thres) <= ppg_max_idx[i]:
            signal_segment = r_peaks[start_segment:end_segment]
            slope_max_t, slope_max_y, intersect_t, intersect_y, slope_max =                 tangent_Comuputation(raw_signal, ppg_min_idx[i], ppg_max_idx[i])
            
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


# In[38]:


# Compute the PAT value
lower_slope_thres = 0.01
min_slope_thres = 5
r_peaks_valid, slope_max_point_list, intersection_point_list, pat_time_list = PAT_compute(new_PPG, new_r_peaks, new_r_peaks_existing_idx, ppg_min_idx, ppg_max_idx, min_slope_thres, lower_slope_thres)
print(pat_time_list)


# In[39]:


# Overview of PPG, maximum slope point (Yellow) and minimum point (Red)
plt.figure(figsize = (20,10))
plt.plot(new_PPG,'b')

plt.plot(slope_max_point_list[0], slope_max_point_list[1],'yo')
plt.plot(new_r_peaks[r_peaks_valid], new_PPG[new_r_peaks[r_peaks_valid]],'ro')


# In[40]:


# Visualize ABP signal to be analyzed
plt.figure(figsize = (20,10))
plt.plot(new_ABP[new_r_peaks[4]:new_r_peaks[5]],'b')


# In[41]:


# Go through the whole ABP signal, extract the minimum and maximum points within the certain segment made by peaks detected
def abp_max_min_locating(input_signal, r_peaks, min_window, max_window):
    min_idx = []
    max_idx = []
    for i in range(len(r_peaks) - 1):
        start = r_peaks[i]
        end = r_peaks[i + 1]
        
        # Find out the minimum value within the min_window, start from very beginning of the segment
        min_point_temp = np.argmin(input_signal[start:start + min_window])
        min_idx.append(min_point_temp)
        
        # Find out the maximum value within the max_window, start from the end point of min_window segment
        max_point_temp = np.argmax(input_signal[min_point_temp:min_point_temp + max_window])
        max_idx.append(max_point_temp)

    return max_idx, min_idx


# In[42]:


# Extract the minimum and maximum points from ABP
min_window = 40
max_window = 40
abp_max_idx, abp_min_idx = abp_max_min_locating(new_ABP, new_r_peaks, min_window, max_window)
print(abp_max_idx, abp_min_idx)


# In[43]:


# Visualize the extracted maximum (Red) and minimum (Green) points of ABP 
plt.figure(figsize = (20,10))
plt.plot(new_ABP[100:600],'b')
plt.plot(abp_max_idx[:4], new_ABP[abp_max_idx[:4]],'ro')
plt.plot(abp_min_idx[:4], new_ABP[abp_min_idx[:4]],'go')


# In[44]:


# Overview of the extracted maximum (Red) and minimum (Green) points of PPG 
plt.figure(figsize = (20,10))
plt.plot(new_ABP[:10000],'b')
plt.plot(abp_max_idx[:100], new_ABP[abp_max_idx[:100]],'ro')
plt.plot(abp_min_idx[:100], new_ABP[abp_min_idx[:100]],'go')


# In[45]:


Systolic_BP = np.array(new_ABP[abp_max_idx])
Diastolic_BP = np.array(new_ABP[abp_min_idx])
print(Systolic_BP)
print(Diastolic_BP)


# In[46]:


print(len(Systolic_BP[r_peaks_valid]))
print(len(Diastolic_BP[r_peaks_valid]))
print(len(pat_time_list))


# In[47]:


Systolic_BP_selected = np.array(Systolic_BP[r_peaks_valid])
Diastolic_BP_selected = np.array(Diastolic_BP[r_peaks_valid])
pat_time_selected = np.array(pat_time_list)


# In[48]:


# Only select Systolic_BP > 100 and 0 < Diastolic_BP < 100
def BP_elimination(Systolic_BP, Diastolic_BP):
    BP_valid_idx = []
    for i in range(len(Systolic_BP)):
        if Systolic_BP[i] > 100 and Diastolic_BP[i] < 100 and Diastolic_BP[i] > 0:
            BP_valid_idx.append(i)
    return np.array(BP_valid_idx)


# In[49]:


BP_valid_idx = BP_elimination(Systolic_BP_selected, Diastolic_BP_selected)


# In[50]:


Systolic_BP_valid = Systolic_BP_selected[BP_valid_idx]
Diastolic_BP_valid = Diastolic_BP_selected[BP_valid_idx]
pat_time_valid = pat_time_selected[BP_valid_idx]
r_peaks_valid_v2 = np.array(r_peaks_valid)[BP_valid_idx]

slope_max_point_valid_t = np.array(slope_max_point_list[0])[BP_valid_idx]
slope_max_point_valid_y = np.array(slope_max_point_list[1])[BP_valid_idx]
intersection_point_valid_t = np.array(intersection_point_list[0])[BP_valid_idx]
intersection_point_valid_y = np.array(intersection_point_list[1])[BP_valid_idx]


# In[51]:


print(slope_max_point_valid_t)


# In[52]:


print(new_r_peaks[r_peaks_valid_v2])


# In[53]:


print(np.array(abp_max_idx)[BP_valid_idx])


# In[54]:


# Visualize the eliminated r-peaks (Red)
plt.figure(figsize = (20,10))
plt.plot(new_ECG[:2000],'b')
plt.plot(new_r_peaks[r_peaks_valid_v2[:10]], new_ECG[new_r_peaks[r_peaks_valid_v2[:10]]],'ro')


# In[55]:


import scipy.stats as st


# In[56]:


# Use linear regression to find out the correlation between Systolic BP and PAT
slope_SBP, intercept_SBP, r_value_SBP, p_value_SBP, std_err_SBP = st.linregress(pat_time_valid, Systolic_BP_valid)


# In[57]:


print(slope_SBP, intercept_SBP, std_err_SBP)


# In[58]:


# Visualize the Systolic_BP-PAT distribution and the regression result
plt.figure(figsize = (10,10))

plt.xlabel('PAT', size=20) 
plt.ylabel('Systolic_BP', size=20) 

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.plot(pat_time_valid, Systolic_BP_valid, 'ro')

t_sbp = np.arange(0, 120, 0.1)
y_sbp = slope_SBP * t_sbp + intercept_SBP
plt.plot(t_sbp, y_sbp, 'b')


# In[59]:


# Use linear regression to find out the correlation between Diastolic BP and PAT
slope_DBP, intercept_DBP, r_value_DBP, p_value_DBP, std_err_DBP = st.linregress(pat_time_valid, Diastolic_BP_valid)


# In[60]:


print(slope_DBP, intercept_DBP, std_err_DBP)


# In[61]:


# Visualize the Diastolic_BP-PAT distribution and the regression result
plt.figure(figsize = (10,10))
plt.xlabel('PAT', size=20) 
plt.ylabel('Diastolic_BP', size=20) 

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.plot(pat_time_valid, Diastolic_BP_valid, 'ro')

t_dbp = np.arange(0, 120, 0.1)
y_dbp = slope_DBP * t_dbp + intercept_DBP
plt.plot(t_dbp, y_dbp, 'b')


# In[62]:


# Average the datapoints (Systolic/Diastolic BP v.s. PAT) over 5 seconds (or other value defined by window_size)
def window_average(input_signal, r_peaks, window_size):
    scope_size = int(r_peaks[-1] // window_size)
    hash_table = np.zeros(scope_size)
    counter = np.zeros(scope_size)
    avg_signal = np.zeros(scope_size)
    
    for i in range(len(r_peaks)):
        if r_peaks[i] // window_size < scope_size:
            idx = int(r_peaks[i] // window_size)
            hash_table[idx] += 1
            counter[idx] += input_signal[i]

    for i in range(scope_size):
        if hash_table[i] > 0:
            avg_signal[i] = counter[i] / hash_table[i]
    return np.array(avg_signal)


# In[63]:


Systolic_BP_avg = window_average(Systolic_BP_valid, new_r_peaks[r_peaks_valid_v2], window_size)
Diastolic_BP_avg = window_average(Diastolic_BP_valid, new_r_peaks[r_peaks_valid_v2], window_size)
pat_time_avg = window_average(pat_time_valid, new_r_peaks[r_peaks_valid_v2], window_size)


# In[64]:


print(Systolic_BP_avg[503])


# In[65]:


print(new_r_peaks[r_peaks_valid_v2])


# In[66]:


pat_time_result = pat_time_avg[pat_time_avg > 0]
Systolic_BP_result = Systolic_BP_avg[Systolic_BP_avg > 0]
Diastolic_BP_result = Diastolic_BP_avg[Diastolic_BP_avg > 0]


# In[67]:


# Use linear regression to find out the correlation between averaged Systolic BP and PAT
slope_SBP, intercept_SBP, r_value_SBP, p_value_SBP, std_err_SBP = st.linregress(pat_time_result, Systolic_BP_result)


# In[68]:


print(pat_time_result)
print(Systolic_BP_result)


# In[69]:


print(slope_SBP, intercept_SBP, std_err_SBP)


# In[70]:


# Visualize the averaged Systolic_BP-PAT distribution and the regression result
plt.figure(figsize = (10,10))

plt.xlabel('PAT', size=20) 
plt.ylabel('Systolic_BP', size=20) 

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.plot(pat_time_result, Systolic_BP_result, 'ro')

t_sbp = np.arange(0, 120, 0.1)
y_sbp = slope_SBP * t_sbp + intercept_SBP
plt.plot(t_sbp, y_sbp, 'b')


# In[71]:


# Use linear regression to find out the correlation between averaged Diastolic BP and PAT
slope_DBP, intercept_DBP, r_value_DBP, p_value_DBP, std_err_DBP = st.linregress(pat_time_result, Diastolic_BP_result)


# In[72]:


print(slope_DBP, intercept_DBP, std_err_DBP)


# In[73]:


# Visualize the averaged Diastolic_BP-PAT distribution and the regression result
plt.figure(figsize = (10,10))
plt.xlabel('PAT', size=20) 
plt.ylabel('Diastolic_BP', size=20) 

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.plot(pat_time_result, Diastolic_BP_result, 'ro')

t_dbp = np.arange(0, 120, 0.1)
y_dbp = slope_DBP * t_dbp + intercept_DBP
plt.plot(t_dbp, y_dbp, 'b')


# In[74]:


# Plot Systolic BP features versus time
plt.figure(figsize = (20,20))
plt.plot(new_r_peaks[r_peaks_valid_v2], Systolic_BP_valid,'ro')


# In[75]:


# Plot Diastolic BP features versus time
plt.figure(figsize = (20,20))
plt.plot(new_r_peaks[r_peaks_valid_v2], Diastolic_BP_valid,'ro')


# In[76]:


# Plot PAT features versus time
plt.figure(figsize = (20,20))
plt.plot(new_r_peaks[r_peaks_valid_v2], pat_time_valid,'ro')


# In[ ]:





# In[ ]:




