# Kamaleswaran Lab, Emory University
# STT computation code for PPG and ECG, logistic regression between STT and Systolic/Diastolic BP from ABP
# Author: Xiyang Wu, M.S. in ECE, Georgia Institute of Technology
# Date: 08/04/2021

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 

import scipy.stats as st
from ecg_ppg_elimination_utils import ecg_elimination, r_peaks_rejection, reject_PPG
from pat_stt_computation_utils import ppg_min_max_locating, STT_compute, tangent_Comuputation
from abp_utils import BP_elimination, abp_max_min_locating
from avg_utils import window_average

from ecgdetectors import Detectors
import heartpy as hp

################ Data Loader ################
data = pd.read_csv("dataRaw_3000393.csv")

ECG = data['ECG'][:-2000]
ABP = data['ABP'][:-2000]
PPG = data['PPG'][:-2000]

step_size = 2000
data_selected = 3000
frequency = 125

new_ECG = ECG[:step_size * data_selected]
new_ABP = ABP[:step_size * data_selected]
new_PPG = PPG[:step_size * data_selected]
# print(len(new_ECG))

new_PPG[np.isnan(new_PPG)] = 0
new_ABP[np.isnan(new_ABP)] = 0
new_ECG[np.isnan(new_ECG)] = 0

################ ECG & PPG Peak Extraction and Elimination ################
# Extract r-packs with ecgdetectors package, while the frquency of imported signal is 125Hz
# https://github.com/berndporr/py-ecg-detectors
detectors = Detectors(frequency)
r_peaks = detectors.engzee_detector(np.array(new_ECG))

new_r_peaks = ecg_elimination(new_ECG, r_peaks, 20)
new_r_peaks = new_r_peaks[2:]

# Set the PPG segment size for rejection
rejection_segment_size = 200000

# Reject PPG peaks with HeartPy
ppg_removed_idx = reject_PPG(new_PPG, rejection_segment_size)

# Further reject PPG peaks 
search_bound = 10
r_peaks_existing_idx, _ = r_peaks_rejection(new_r_peaks, ppg_removed_idx, search_bound)

################ STT Computation ################
# Extract the minimum and maximum points from PPG
min_window = 80
max_window = 40
ppg_max_idx, ppg_min_idx = ppg_min_max_locating(new_PPG, new_r_peaks, min_window, max_window)

# print(ppg_max_idx)
# print(ppg_min_idx)

# Compute STT value
lower_slope_thres = 0.01
min_slope_thres = 5
r_peaks_valid, slope_max_point_list, stt_time_list = STT_compute(new_PPG, new_r_peaks, r_peaks_existing_idx, ppg_min_idx, ppg_max_idx, min_slope_thres, lower_slope_thres)
################ ABP Peak Extraction and Elimination ################
# Extract the minimum and maximum points from ABP
min_window = 40
max_window = 40
abp_max_idx, abp_min_idx = abp_max_min_locating(new_ABP, new_r_peaks, min_window, max_window)

Systolic_BP = np.array(new_ABP[abp_max_idx])
Diastolic_BP = np.array(new_ABP[abp_min_idx])


# Select Systolic and Diastolic BP, STT based on the r-peaks elimination result
Systolic_BP_selected = np.array(Systolic_BP[r_peaks_valid])
Diastolic_BP_selected = np.array(Diastolic_BP[r_peaks_valid])
stt_time_selected = np.array(stt_time_list)

BP_valid_idx = BP_elimination(Systolic_BP_selected, Diastolic_BP_selected)

Systolic_BP_valid = Systolic_BP_selected[BP_valid_idx]
Diastolic_BP_valid = Diastolic_BP_selected[BP_valid_idx]
stt_time_valid = stt_time_selected[BP_valid_idx]
r_peaks_valid_v2 = np.array(r_peaks_valid)[BP_valid_idx]

################ Logistic Regression and Data Distribution Plotting ################
# Use linear regression to find out the correlation between Systolic BP and STT
slope_SBP, intercept_SBP, r_value_SBP, p_value_SBP, std_err_SBP = st.linregress(stt_time_valid, Systolic_BP_valid)
print('SBP - STT:', slope_SBP, intercept_SBP, std_err_SBP)

# Visualize the Systolic_BP-STT distribution and the regression result
non_avg = plt.figure(figsize = (20,10))
ax1 = non_avg.add_subplot(1, 2, 1)

ax1.set_xlabel('STT', size=20)
ax1.set_ylabel('Systolic_BP', size=20)

ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax1.plot(stt_time_valid, Systolic_BP_valid, 'ro')

t_sbp = np.arange(0, 100, 0.1)
y_sbp = slope_SBP * t_sbp + intercept_SBP
ax1.plot(t_sbp, y_sbp, 'b')

# Use linear regression to find out the correlation between Diastolic BP and STT
slope_DBP, intercept_DBP, r_value_DBP, p_value_DBP, std_err_DBP = st.linregress(stt_time_valid, Diastolic_BP_valid)
print('DBP - STT:', slope_DBP, intercept_DBP, std_err_DBP)


# Visualize the Diastolic_BP-STT distribution and the regression result
ax2 = non_avg.add_subplot(1, 2, 2)
ax2.set_xlabel('STT', size=20)
ax2.set_ylabel('Diastolic_BP', size=20)

ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
ax2.plot(stt_time_valid, Diastolic_BP_valid, 'ro')

t_dbp = np.arange(0, 100, 0.1)
y_dbp = slope_DBP * t_dbp + intercept_DBP
ax2.plot(t_dbp, y_dbp, 'b')
non_avg.savefig('non_avg_stt.png')


################ Logistic Regression and Data Distribution Plotting ################
window_size = frequency * 5

Systolic_BP_avg = window_average(Systolic_BP_valid, r_peaks_valid_v2, window_size)
Diastolic_BP_avg = window_average(Diastolic_BP_valid, r_peaks_valid_v2, window_size)
stt_time_avg = window_average(stt_time_valid, r_peaks_valid_v2, window_size)

# Use linear regression to find out the correlation between averaged Systolic BP and STT
slope_SBP, intercept_SBP, r_value_SBP, p_value_SBP, std_err_SBP = st.linregress(stt_time_avg, Systolic_BP_avg)
print('Average SBP - STT:', slope_SBP, intercept_SBP, std_err_SBP)

# Visualize the averaged Systolic_BP-STT distribution and the regression result
avg = plt.figure(figsize = (20,10))
ax1_avg = avg.add_subplot(1, 2, 1)

ax1_avg.set_xlabel('STT', size=20)
ax1_avg.set_ylabel('Systolic_BP', size=20)

ax1_avg.tick_params(axis='x', labelsize=20)
ax1_avg.tick_params(axis='y', labelsize=20)
ax1_avg.plot(stt_time_avg, Systolic_BP_avg, 'ro')

t_sbp = np.arange(0, 100, 0.1)
y_sbp = slope_SBP * t_sbp + intercept_SBP
ax1_avg.plot(t_sbp, y_sbp, 'b')

# Use linear regression to find out the correlation between averaged Diastolic BP and STT
slope_DBP, intercept_DBP, r_value_DBP, p_value_DBP, std_err_DBP = st.linregress(stt_time_avg, Diastolic_BP_avg)
print('Average DBP - STT:', slope_DBP, intercept_DBP, std_err_DBP)

# Visualize the averaged Diastolic_BP-STT distribution and the regression result
ax2_avg = avg.add_subplot(1, 2, 2)
ax2_avg.set_xlabel('STT', size=20)
ax2_avg.set_ylabel('Diastolic_BP', size=20)

ax2_avg.tick_params(axis='x', labelsize=20)
ax2_avg.tick_params(axis='y', labelsize=20)
ax2_avg.plot(stt_time_avg, Diastolic_BP_avg, 'ro')

t_dbp = np.arange(0, 100, 0.1)
y_dbp = slope_DBP * t_dbp + intercept_DBP
ax2_avg.plot(t_dbp, y_dbp, 'b')
avg.savefig('avg_stt.png')