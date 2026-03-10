"""
Heart Rate Monitoring in Frequency Domain!

Same function as heart-rate-peak-detect but use FFT instead of peak detection algorithm.

"""
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

ppg_waveform = np.loadtxt('HW5.1A.PPG.txt')

dt = 1/125
N = 4096
t = np.arange(0, dt*N, dt)

# remove DC offset (zero-mean)
ppg_demean = ppg_waveform - np.mean(ppg_waveform)
# apply Hanning window to reduce spectral leakage
window = np.hanning(N)
ppg_windowed = ppg_demean * window

# max frequency we'd see: 220bpm --> 3.67Hz
fft_values = np.fft.fft(ppg_windowed)
freq = np.fft.fftfreq(N, dt)
magnitude = np.abs(fft_values) / N
# peak in positive frequencies only (skip DC at index 0)
mag_pos = magnitude[1:N//2]
freq_pos = freq[1:N//2]
fundamental_index = 1 + np.argmax(mag_pos)
fundamental_freq_hz = freq[fundamental_index]
fundamental_mag = magnitude[fundamental_index]
print("Fundamental frequency at", fundamental_freq_hz, "Hz -->", fundamental_freq_hz*60, "bpm")

plt.stem(freq[:N//2], magnitude[:N//2])
plt.scatter(fundamental_freq_hz, fundamental_mag, color='red', s=80, zorder=5, label=f'Fundamental ({fundamental_freq_hz:.2f} Hz)')
plt.title('Frequency Domain (Magnitude Spectrum)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, 3)   #
plt.legend()
plt.tight_layout()
plt.show()