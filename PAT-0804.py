# Kamaleswaran Lab, Emory University
# PAT computation code for PPG and ECG, logistic regression between PAT and Systolic/Diastolic BP from ABP
# Author: Xiyang Wu, M.S. in ECE, Georgia Institute of Technology
# Date: 08/04/2021

import scipy

from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import scipy.stats as st
from ecg_ppg_elimination_utils import ecg_elimination, r_peaks_rejection, reject_PPG, r_peaks_segment_rejection
from pat_stt_computation_utils import ppg_min_max_locating, PAT_compute, tangent_Comuputation
from abp_utils import abp_max_min_locating
from avg_utils import window_average

from ecgdetectors import Detectors
import heartpy as hp

# Data Loader
data = pd.read_csv("dataRaw_3000393.csv")

ECG = data['ECG'][:-2000]
ABP = data['ABP'][:-2000]
PPG = data['PPG'][:-2000]

step_size = 2000
data_selected = 3000

new_ECG = ECG[:step_size * data_selected]
new_ABP = ABP[:step_size * data_selected]
new_PPG = PPG[:step_size * data_selected]

new_PPG[np.isnan(new_PPG)] = 0
new_ABP[np.isnan(new_ABP)] = 0
new_ECG[np.isnan(new_ECG)] = 0

# Extract r-packs with ecgdetectors package, while the frquency of imported signal is 125Hz
# https://github.com/berndporr/py-ecg-detectors
detectors = Detectors(125)
r_peaks = detectors.engzee_detector(np.array(new_ECG))

new_r_peaks = ecg_elimination(new_ECG, r_peaks, 20)
new_r_peaks = new_r_peaks[2:]

# Set the PPG segment size for rejection
rejection_segment_size = 200000

# Reject PPG peaks with HeartPy
ppg_removed_idx = reject_PPG(new_PPG, rejection_segment_size)

# Further reject PPG peaks 
search_bound = 10
r_peaks_existing_idx, r_peaks_rejected_idx = r_peaks_rejection(new_r_peaks, ppg_removed_idx, search_bound)

frequency = 125
window_size = frequency * 30
rejection_window_size = frequency * 30
rejection_thres = 0.1

new_r_peaks_existing_idx = r_peaks_segment_rejection(new_r_peaks, r_peaks_existing_idx, r_peaks_rejected_idx, rejection_window_size, rejection_thres)

# Extract the minimum and maximum points from PPG
ppg_fitting_size = step_size * 2
min_window = 80
max_window = 40
ppg_max_idx, ppg_min_idx = ppg_min_max_locating(new_PPG, new_r_peaks, min_window, max_window)

# Compute the PAT value
lower_slope_thres = 0.01
min_slope_thres = 5
r_peaks_valid, slope_max_point_list, intersection_point_list, pat_time_list = PAT_compute(new_PPG, new_r_peaks, new_r_peaks_existing_idx, ppg_min_idx, ppg_max_idx, min_slope_thres, lower_slope_thres)

# Extract the minimum and maximum points from ABP
min_window = 40
max_window = 40
abp_max_idx, abp_min_idx = abp_max_min_locating(new_ABP, new_r_peaks, min_window, max_window)

# Visualize the extracted maximum (Red) and minimum (Green) points of ABP 
plt.figure(figsize = (20,10))
plt.plot(new_ABP[100:600],'b')
plt.plot(abp_max_idx[:4], new_ABP[abp_max_idx[:4]],'ro')
plt.plot(abp_min_idx[:4], new_ABP[abp_min_idx[:4]],'go')

# Overview of the extracted maximum (Red) and minimum (Green) points of PPG 
plt.figure(figsize = (20,10))
plt.plot(new_ABP[:10000],'b')
plt.plot(abp_max_idx[:100], new_ABP[abp_max_idx[:100]],'ro')
plt.plot(abp_min_idx[:100], new_ABP[abp_min_idx[:100]],'go')

Systolic_BP = np.array(new_ABP[abp_max_idx])
Diastolic_BP = np.array(new_ABP[abp_min_idx])

Systolic_BP_selected = np.array(Systolic_BP[r_peaks_valid])
Diastolic_BP_selected = np.array(Diastolic_BP[r_peaks_valid])
pat_time_selected = np.array(pat_time_list)

# Only select Systolic_BP > 100 and 0 < Diastolic_BP < 100
def BP_elimination(Systolic_BP, Diastolic_BP):
    BP_valid_idx = []
    for i in range(len(Systolic_BP)):
        if Systolic_BP[i] > 100 and Diastolic_BP[i] < 100 and Diastolic_BP[i] > 0:
            BP_valid_idx.append(i)
    return np.array(BP_valid_idx)

BP_valid_idx = BP_elimination(Systolic_BP_selected, Diastolic_BP_selected)

Systolic_BP_valid = Systolic_BP_selected[BP_valid_idx]
Diastolic_BP_valid = Diastolic_BP_selected[BP_valid_idx]
pat_time_valid = pat_time_selected[BP_valid_idx]
r_peaks_valid_v2 = np.array(r_peaks_valid)[BP_valid_idx]

slope_max_point_valid_t = np.array(slope_max_point_list[0])[BP_valid_idx]
slope_max_point_valid_y = np.array(slope_max_point_list[1])[BP_valid_idx]
intersection_point_valid_t = np.array(intersection_point_list[0])[BP_valid_idx]
intersection_point_valid_y = np.array(intersection_point_list[1])[BP_valid_idx]

# Use linear regression to find out the correlation between Systolic BP and PAT
slope_SBP, intercept_SBP, r_value_SBP, p_value_SBP, std_err_SBP = st.linregress(pat_time_valid, Systolic_BP_valid)
print('SBP - PAT:', slope_SBP, intercept_SBP, std_err_SBP)

# Visualize the Systolic_BP-PAT distribution and the regression result
non_avg = plt.figure(figsize = (20,10))
ax1 = non_avg.add_subplot(1, 2, 1)

ax1.set_xlabel('STT', size=20)
ax1.set_ylabel('Systolic_BP', size=20)

ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax1.plot(pat_time_valid, Systolic_BP_valid, 'ro')

t_sbp = np.arange(0, 100, 0.1)
y_sbp = slope_SBP * t_sbp + intercept_SBP
ax1.plot(t_sbp, y_sbp, 'b')

# Use linear regression to find out the correlation between Diastolic BP and PAT
slope_DBP, intercept_DBP, r_value_DBP, p_value_DBP, std_err_DBP = st.linregress(pat_time_valid, Diastolic_BP_valid)
print('DBP - PAT:', slope_DBP, intercept_DBP, std_err_DBP)

# Visualize the Diastolic_BP-PAT distribution and the regression result
ax2 = non_avg.add_subplot(1, 2, 2)
ax2.set_xlabel('STT', size=20)
ax2.set_ylabel('Diastolic_BP', size=20)

ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
ax2.plot(pat_time_valid, Diastolic_BP_valid, 'ro')

t_dbp = np.arange(0, 100, 0.1)
y_dbp = slope_DBP * t_dbp + intercept_DBP
ax2.plot(t_dbp, y_dbp, 'b')
non_avg.savefig('non_avg_pat.png')

Systolic_BP_avg = window_average(Systolic_BP_valid, new_r_peaks[r_peaks_valid_v2], window_size)
Diastolic_BP_avg = window_average(Diastolic_BP_valid, new_r_peaks[r_peaks_valid_v2], window_size)
pat_time_avg = window_average(pat_time_valid, new_r_peaks[r_peaks_valid_v2], window_size)

pat_time_result = pat_time_avg[pat_time_avg > 0]
Systolic_BP_result = Systolic_BP_avg[Systolic_BP_avg > 0]
Diastolic_BP_result = Diastolic_BP_avg[Diastolic_BP_avg > 0]

# Use linear regression to find out the correlation between averaged Systolic BP and PAT
slope_SBP, intercept_SBP, r_value_SBP, p_value_SBP, std_err_SBP = st.linregress(pat_time_result, Systolic_BP_result)
print('Average SBP - PAT:', slope_SBP, intercept_SBP, std_err_SBP)

# Visualize the averaged Systolic_BP-STT distribution and the regression result
avg = plt.figure(figsize = (20,10))
ax1_avg = avg.add_subplot(1, 2, 1)

ax1_avg.set_xlabel('PAT', size=20)
ax1_avg.set_ylabel('Systolic_BP', size=20)

ax1_avg.tick_params(axis='x', labelsize=20)
ax1_avg.tick_params(axis='y', labelsize=20)
ax1_avg.plot(pat_time_avg, Systolic_BP_avg, 'ro')

t_sbp = np.arange(0, 100, 0.1)
y_sbp = slope_SBP * t_sbp + intercept_SBP
ax1_avg.plot(t_sbp, y_sbp, 'b')

# Use linear regression to find out the correlation between averaged Diastolic BP and PAT
slope_DBP, intercept_DBP, r_value_DBP, p_value_DBP, std_err_DBP = st.linregress(pat_time_result, Diastolic_BP_result)
print('Average DBP - PAT:', slope_DBP, intercept_DBP, std_err_DBP)

# Visualize the averaged Diastolic_BP-STT distribution and the regression result
ax2_avg = avg.add_subplot(1, 2, 2)
ax2_avg.set_xlabel('PAT', size=20)
ax2_avg.set_ylabel('Diastolic_BP', size=20)

ax2_avg.tick_params(axis='x', labelsize=20)
ax2_avg.tick_params(axis='y', labelsize=20)
ax2_avg.plot(pat_time_avg, Diastolic_BP_avg, 'ro')

t_dbp = np.arange(0, 100, 0.1)
y_dbp = slope_DBP * t_dbp + intercept_DBP
ax2_avg.plot(t_dbp, y_dbp, 'b')
avg.savefig('avg_pat.png')
