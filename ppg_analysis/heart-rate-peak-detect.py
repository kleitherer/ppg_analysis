"""
Heart Rate Monitoring in Time Domain!

A real PPG waveform sample set is in the file labeled HW5.1A.PPG.txt. 

These samples are normalized to full scale of the ADC, and are sampled at 125Hz. 
The trace length is 4096 samples (about 30 seconds). 

In the time domain, using the scipy.find_peaks function (or write your own), 
write a simple peak detection algorithm and use it to compute the time
differences between consecutive peaks of the PPG waveform. 

From here, determine the heart rate (the average difference between peaks, in beats-per-minute), 
as well as the maximum heart rate variability as well as the RMS heart rate variability (HRV). 
Submit your code, as well as the BPM and max/RMS HRV values

"""
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

ppg_waveform = np.loadtxt('HW5.1A.PPG.txt')

peak_indices, properties = find_peaks(ppg_waveform, width=15) # chose width of 15 after tuning between 10-50
dt = 1/125
print("Number of peaks found in ~33 second dataset:", len(peak_indices), "peaks")

time_dif_between_peaks = []
i = 0
while (i < len(peak_indices)-1):
    time_value = peak_indices[i+1]*dt - peak_indices[i]*dt
    time_dif_between_peaks.append(time_value.item())
    i+=1

avg_time_per_beat = sum(time_dif_between_peaks)/len(time_dif_between_peaks)
beats_per_s = 1/avg_time_per_beat
bpm = beats_per_s * 60
print("Heart Rate:", bpm, "beats per minute")

# maximum heart rate variability
max_interval = max(time_dif_between_peaks)
max_index = np.argmax(time_dif_between_peaks)
t_max_start = peak_indices[max_index] * dt
t_max_end = peak_indices[max_index + 1] * dt
print("Max of", max_interval*1000, "ms found between", t_max_start, "and", t_max_end, "s")

# find root mean square heart rate variability (RMSSD)
successive_diffs_sec = []
for i in range(len(time_dif_between_peaks) - 1):
    diff = time_dif_between_peaks[i + 1] - time_dif_between_peaks[i]
    successive_diffs_sec.append(diff)

sum_sq = sum(d ** 2 for d in successive_diffs_sec)
rms_hrv_sec = (sum_sq / len(successive_diffs_sec)) ** 0.5
rms_hrv_ms = rms_hrv_sec * 1000
print("RMS HRV:", rms_hrv_ms, "ms")

t = np.arange(0, dt*4096, dt)
plt.plot(t, ppg_waveform, label='PPG')
plt.plot(peak_indices * dt, ppg_waveform[peak_indices], 'ro', markersize=4, label='Peaks')
# highlight the max interval on the plot (between peak max_index and max_index+1)

plt.axvspan(t_max_start, t_max_end, alpha=0.2, color='red', label=f'Max interval = {max_interval:.3f} s')
plt.xlabel("Time (s)")
plt.ylabel("Waveform (Raw ADC Values)")
plt.title("PPG Waveform Sample")
plt.legend()
plt.show()


