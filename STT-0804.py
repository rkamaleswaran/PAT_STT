# Kamaleswaran Lab, Emory University
# STT computation code for PPG and ECG, logistic regression between STT and Systolic/Diastolic BP from ABP
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


# In[18]:


# Since some normal peaks may exist in problematic PPG signal and this phenomenon may 
# introduce failure in STT computation, add an extra peak rejection function
# Set a search_bound and check all peaks within this bound, reject all preaks within this bound when a removed peak is found
def r_peaks_rejection(r_peaks, removed_idx, search_bound):
    # Initialize the pointer
    ptr_removed = 0
    r_peaks_existing_idx = []
    while removed_idx[ptr_removed] < r_peaks[0]:
        ptr_removed += 1
        
    # Iteration number
    iter_num = len(r_peaks) // search_bound
    
    for i in range(iter_num - 1):  
        # Go through all removed peaks
        if ptr_removed < len(removed_idx):
            if removed_idx[ptr_removed] >= r_peaks[i * search_bound] and removed_idx[ptr_removed] <= r_peaks[min((i + 1) * search_bound, len(r_peaks) - 1)]:
                # check all peaks within this bound, reject all preaks within this bound when a removed peak is found
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

    r_peaks = np.array(r_peaks)
    return r_peaks_existing_idx


# In[19]:


# Set the PPG segment size for rejection
rejection_segment_size = 200000

# Reject PPG peaks with HeartPy
ppg_removed_idx = reject_PPG(new_PPG, rejection_segment_size)


# In[20]:


# Further reject PPG peaks 
search_bound = 10
r_peaks_existing_idx = r_peaks_rejection(new_r_peaks, ppg_removed_idx, search_bound)


# In[21]:


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


# In[22]:


# Extract the minimum and maximum points from PPG
min_window = 80
max_window = 40
ppg_max_idx, ppg_min_idx = ppg_min_max_locating(new_PPG, new_r_peaks, min_window, max_window)
print(ppg_max_idx, ppg_min_idx)


# In[23]:


# Visualize the ECG and PPG signal after coordination
plt.figure(figsize = (20,10))
plt.plot(new_ECG[:200] * 4,'r')
plt.plot(new_PPG[:200],'b')


# In[24]:


# Visualize the extracted maximum (Red) and minimum (Green) points of PPG 
plt.figure(figsize = (20,10))
plt.plot(new_PPG[:100 * 100],'b')
plt.plot(ppg_max_idx[:100], new_PPG[ppg_max_idx[:100]],'ro')
plt.plot(ppg_min_idx[:100], new_PPG[ppg_min_idx[:100]],'go')


# In[25]:


# Determine the tangent point with differential method: Find out the point with maximum slope value between two proximate points
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
    
    return slope_max_t, slope_max_y, slope_max


# In[26]:


# Find out the point with maximum slope
slope_max_t, slope_max_y, slope_max = tangent_Comuputation(new_PPG, ppg_min_idx[442], ppg_max_idx[442])


# In[27]:


# Visualize the PPG, maximum slope point (Green) and minimum point (Red)
plt.figure(figsize = (20,10))
plt.plot(new_PPG[48000:50000],'b')
plt.plot(ppg_min_idx[442], new_PPG[ppg_min_idx[442]],'ro')
plt.plot(slope_max_t, slope_max_y,'go')


# In[28]:


# Compute STT value, which is the reciprocal of the maximum slope
def STT_compute(raw_signal, r_peaks, r_peaks_existing_idx, ppg_min_idx, ppg_max_idx, min_slope_thres, lower_slope_thres):
    
    slope_max_point_list = [[], []]
    stt_time_list = []
    
    r_peaks_valid = []
    
    # Separate the PPG signal into segments and extract the maximum slope value one by one
    for i in range(len(r_peaks) - 1):
        start_segment = r_peaks[i]
        end_segment = r_peaks[i + 1]
        
        # Ensure the minimum and maximum points locate far enough
        if (ppg_min_idx[i] + min_slope_thres) <= ppg_max_idx[i]:
            signal_segment = r_peaks[start_segment:end_segment]
            slope_max_t, slope_max_y, slope_max =                 tangent_Comuputation(new_PPG, ppg_min_idx[i], ppg_max_idx[i])
            
            # Ensure the slope is larger than the lower upper to eliminate the flat PPG signal
            if slope_max > lower_slope_thres and i in r_peaks_existing_idx:
                STT_time = 1 / slope_max

                r_peaks_valid.append(i)
                slope_max_point_list[0].append(slope_max_t)
                slope_max_point_list[1].append(slope_max_y)

                stt_time_list.append(STT_time)

    return r_peaks_valid, slope_max_point_list, stt_time_list


# In[29]:


# Compute STT value
lower_slope_thres = 0.01
min_slope_thres = 5
r_peaks_valid, slope_max_point_list, stt_time_list = STT_compute(new_PPG, new_r_peaks, r_peaks_existing_idx, ppg_min_idx, ppg_max_idx, min_slope_thres, lower_slope_thres)
print(stt_time_list)


# In[30]:


# Overview of PPG, maximum slope point (Yellow) and minimum point (Red)
plt.figure(figsize = (20,10))
plt.plot(new_PPG,'b')

plt.plot(slope_max_point_list[0], slope_max_point_list[1],'yo')
plt.plot(new_r_peaks[r_peaks_valid], new_PPG[new_r_peaks[r_peaks_valid]],'ro')


# In[31]:


# Visualize ABP signal to be analyzed
plt.figure(figsize = (20,10))
plt.plot(new_ABP[new_r_peaks[4]:new_r_peaks[5]],'b')


# In[32]:


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


# In[33]:


# Extract the minimum and maximum points from ABP
min_window = 40
max_window = 40
abp_max_idx, abp_min_idx = abp_max_min_locating(new_ABP, new_r_peaks, min_window, max_window)
print(abp_max_idx, abp_min_idx)


# In[34]:


# Visualize the extracted maximum (Red) and minimum (Green) points of ABP 
plt.figure(figsize = (20,10))
plt.plot(new_ABP[100:600],'b')
plt.plot(abp_max_idx[:4], new_ABP[abp_max_idx[:4]],'ro')
plt.plot(abp_min_idx[:4], new_ABP[abp_min_idx[:4]],'go')


# In[35]:


# Overview of the extracted maximum (Red) and minimum (Green) points of PPG 
plt.figure(figsize = (20,10))
plt.plot(new_ABP[:10000],'b')
plt.plot(abp_max_idx[:100], new_ABP[abp_max_idx[:100]],'ro')
plt.plot(abp_min_idx[:100], new_ABP[abp_min_idx[:100]],'go')


# In[36]:


Systolic_BP = np.array(new_ABP[abp_max_idx])
Diastolic_BP = np.array(new_ABP[abp_min_idx])
print(Systolic_BP)
print(Diastolic_BP)


# In[37]:


print(len(Systolic_BP[r_peaks_valid]))
print(len(Diastolic_BP[r_peaks_valid]))
print(len(stt_time_list))


# In[38]:


# Select Systolic and Diastolic BP, STT based on the r-peaks elimination result
Systolic_BP_selected = np.array(Systolic_BP[r_peaks_valid])
Diastolic_BP_selected = np.array(Diastolic_BP[r_peaks_valid])
stt_time_selected = np.array(stt_time_list)


# In[39]:


# Only select Systolic_BP > 100 and 0 < Diastolic_BP < 100
def BP_elimination(Systolic_BP, Diastolic_BP):
    BP_valid_idx = []
    for i in range(len(Systolic_BP)):
        if Systolic_BP[i] > 100 and Diastolic_BP[i] < 100 and Diastolic_BP[i] > 0:
            BP_valid_idx.append(i)
    return np.array(BP_valid_idx)


# In[40]:


BP_valid_idx = BP_elimination(Systolic_BP_selected, Diastolic_BP_selected)


# In[41]:


Systolic_BP_valid = Systolic_BP_selected[BP_valid_idx]
Diastolic_BP_valid = Diastolic_BP_selected[BP_valid_idx]
stt_time_valid = stt_time_selected[BP_valid_idx]
r_peaks_valid_v2 = np.array(r_peaks_valid)[BP_valid_idx]


# In[42]:


# Visualize the eliminated r-peaks (Red)
plt.figure(figsize = (20,10))
plt.plot(new_ECG[:2000],'b')
plt.plot(new_r_peaks[r_peaks_valid_v2[:10]], new_ECG[new_r_peaks[r_peaks_valid_v2[:10]]],'ro')


# In[43]:


import scipy.stats as st


# In[44]:


# Use linear regression to find out the correlation between Systolic BP and STT
slope_SBP, intercept_SBP, r_value_SBP, p_value_SBP, std_err_SBP = st.linregress(stt_time_valid, Systolic_BP_valid)


# In[45]:


print(slope_SBP, intercept_SBP, std_err_SBP)


# In[46]:


# Visualize the Systolic_BP-STT distribution and the regression result
plt.figure(figsize = (10,10))

plt.xlabel('STT', size=20) 
plt.ylabel('Systolic_BP', size=20) 

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.plot(stt_time_valid, Systolic_BP_valid, 'ro')

t_sbp = np.arange(0, 100, 0.1)
y_sbp = slope_SBP * t_sbp + intercept_SBP
plt.plot(t_sbp, y_sbp, 'b')


# In[47]:


# Use linear regression to find out the correlation between Diastolic BP and STT
slope_DBP, intercept_DBP, r_value_DBP, p_value_DBP, std_err_DBP = st.linregress(stt_time_valid, Diastolic_BP_valid)


# In[48]:


print(slope_DBP, intercept_DBP, std_err_DBP)


# In[49]:


# Visualize the Diastolic_BP-STT distribution and the regression result
plt.figure(figsize = (10,10))
plt.xlabel('STT', size=20) 
plt.ylabel('Diastolic_BP', size=20) 

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.plot(stt_time_valid, Diastolic_BP_valid, 'ro')

t_dbp = np.arange(0, 100, 0.1)
y_dbp = slope_DBP * t_dbp + intercept_DBP
plt.plot(t_dbp, y_dbp, 'b')


# In[50]:


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


# In[51]:


frequency = 125
window_size = frequency * 5


# In[52]:


Systolic_BP_avg = window_average(Systolic_BP_valid, r_peaks_valid_v2, window_size)
Diastolic_BP_avg = window_average(Diastolic_BP_valid, r_peaks_valid_v2, window_size)
stt_time_avg = window_average(stt_time_valid, r_peaks_valid_v2, window_size)


# In[53]:


# Use linear regression to find out the correlation between averaged Systolic BP and STT
slope_SBP, intercept_SBP, r_value_SBP, p_value_SBP, std_err_SBP = st.linregress(stt_time_avg, Systolic_BP_avg)


# In[54]:


print(stt_time_avg)
print(Systolic_BP_avg)


# In[55]:


print(slope_SBP, intercept_SBP, std_err_SBP)


# In[56]:


# Visualize the averaged Systolic_BP-STT distribution and the regression result
plt.figure(figsize = (10,10))

plt.xlabel('STT', size=20) 
plt.ylabel('Systolic_BP', size=20) 

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.plot(stt_time_avg, Systolic_BP_avg, 'ro')

t_sbp = np.arange(0, 100, 0.1)
y_sbp = slope_SBP * t_sbp + intercept_SBP
plt.plot(t_sbp, y_sbp, 'b')


# In[57]:


# Use linear regression to find out the correlation between averaged Diastolic BP and STT
slope_DBP, intercept_DBP, r_value_DBP, p_value_DBP, std_err_DBP = st.linregress(stt_time_avg, Diastolic_BP_avg)


# In[58]:


print(slope_DBP, intercept_DBP, std_err_DBP)


# In[59]:


# Visualize the averaged Diastolic_BP-STT distribution and the regression result
plt.figure(figsize = (10,10))
plt.xlabel('STT', size=20) 
plt.ylabel('Diastolic_BP', size=20) 

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.plot(stt_time_avg, Diastolic_BP_avg, 'ro')

t_dbp = np.arange(0, 100, 0.1)
y_dbp = slope_DBP * t_dbp + intercept_DBP
plt.plot(t_dbp, y_dbp, 'b')


# In[60]:


# Plot Systolic BP features versus time
plt.figure(figsize = (20,20))
plt.plot(new_r_peaks[r_peaks_valid_v2], Systolic_BP_valid,'ro')


# In[61]:


# Plot Diastolic BP features versus time
plt.figure(figsize = (20,20))
plt.plot(new_r_peaks[r_peaks_valid_v2], Diastolic_BP_valid,'ro')


# In[62]:


# Plot STT features versus time
plt.figure(figsize = (20,20))
plt.plot(new_r_peaks[r_peaks_valid_v2], stt_time_valid,'ro')


# In[ ]:




