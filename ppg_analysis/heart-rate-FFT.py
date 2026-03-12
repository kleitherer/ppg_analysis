"""
Heart Rate Monitoring in Frequency Domain!

Same function as heart-rate-peak-detect but use FFT instead of peak detection algorithm.

I tried a few different windowing techniques, but ultimately chose Hamming.

"""
import numpy as np
import matplotlib.pyplot as plt

ppg_waveform = np.loadtxt('HW5.1A.PPG.txt')

dt = 1/125
N = 4096
t = np.arange(0, dt*N, dt)

# remove DC offset (zero-mean)
ppg_demean = ppg_waveform - np.mean(ppg_waveform)
freq = np.fft.fftfreq(N, dt)
freq_pos = freq[1:N//2]

# FFT *without* windowing
fft_nowin = np.fft.fft(ppg_demean)
magnitude_nowin = np.abs(fft_nowin) / N


# windows = {
#     'Hanning': np.hanning(N),
#     'Hamming': np.hamming(N),
#     'Blackman': np.blackman(N),
#     'Bartlett': np.bartlett(N),
# }

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
ax1.stem(freq_pos, magnitude_nowin[1:N//2], label='No window')
ax1.set_ylabel('Magnitude')
ax1.set_title('Frequency Domain — Without Windowing')
ax1.set_xlim(0, 3)
ax1.legend()
ax1.grid(True, alpha=0.3)

# FFT with each window (to visualize different windows)
# for name, window in windows.items():
#     ppg_windowed = ppg_demean * window
#     fft_win = np.fft.fft(ppg_windowed)
#     mag_win = np.abs(fft_win) / N
#     ax2.plot(freq_pos, mag_win[1:N//2], label=name)


ppg_hamming = ppg_demean * np.hamming(N)
magnitude = np.abs(np.fft.fft(ppg_hamming)) / N
mag_pos = magnitude[1:N//2]
fundamental_index = 1 + np.argmax(mag_pos)
fundamental_freq_hz = freq[fundamental_index]
fundamental_mag = magnitude[fundamental_index]
ax2.stem(freq_pos,mag_pos, label="Hamming")
ax2.scatter(fundamental_freq_hz, fundamental_mag, color='red', s=60, zorder=5, label=f'Fundamental ({fundamental_freq_hz:.4f} Hz)')
print("Fundamental frequency at", fundamental_freq_hz, "Hz -->", fundamental_freq_hz*60, "bpm")

ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Magnitude')
ax2.set_title('Frequency Domain With Hamming Window')
ax2.set_xlim(0, 3)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()