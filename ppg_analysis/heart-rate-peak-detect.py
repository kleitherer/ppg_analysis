"""
Heart Rate and HRV from PPG using Time-Domain Peak Detection

Loads a 4096-sample PPG waveform (125 Hz, ~33 seconds) from HW5.1A.PPG.txt.
Uses scipy.find_peaks to locate heartbeat peaks, then computes:
  - Heart rate (BPM) from the average peak-to-peak interval
  - Max HRV: the largest beat-to-beat interval change
  - RMS HRV (RMSSD): root-mean-square of successive interval changes

HRV measures how much the time between consecutive beats *changes* from
one beat to the next — not the intervals themselves, but the differences
between successive intervals.

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

# differences between consecutive IBIs (for HRV)
interval_changes = []
for i in range(len(time_dif_between_peaks) - 1):
    delta = time_dif_between_peaks[i + 1] - time_dif_between_peaks[i]
    interval_changes.append(delta)

# maximum heart rate variability = largest IBI change
abs_changes = [abs(d) for d in interval_changes]
max_hrv_sec = max(abs_changes)
max_hrv_index = np.argmax(abs_changes)
t_max_start = peak_indices[max_hrv_index + 1] * dt
t_max_end = peak_indices[max_hrv_index + 2] * dt
print(f"Max HRV: {max_hrv_sec * 1000:.2f} ms found between {t_max_start} and {t_max_end} s")

# root mean square heart rate variability (RMSSD)
sum_sq = sum(d ** 2 for d in interval_changes)
rms_hrv_sec = (sum_sq / len(interval_changes)) ** 0.5
rms_hrv_ms = rms_hrv_sec * 1000
print(f"RMS HRV: {rms_hrv_ms:.2f} ms")

t = np.arange(0, dt*4096, dt)
plt.plot(t, ppg_waveform, label='PPG')
plt.plot(peak_indices * dt, ppg_waveform[peak_indices], 'ro', markersize=4, label='Peaks')
# highlight the max interval on the plot (between peak max_index and max_index+1)
plt.axvspan(t_max_start, t_max_end, alpha=0.2, color='red', label=f'Max HRV = {max_hrv_sec*1000:.1f} ms')
plt.xlabel("Time (s)")
plt.ylabel("Waveform (Raw ADC Values)")
plt.title("PPG Waveform Sample")
plt.xlim(0,33)
plt.legend()
plt.show()


