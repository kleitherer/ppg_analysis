"""
Heart Rate Extraction from a Motion-Corrupted PPG (HW5.1C)

Loads a PPG waveform with finger-tapping motion artifacts (~3x larger than
the heart rate signal) and attempts to recover the true heart rate using
three filtering approaches:
  1. Raw peak detection (no filtering)
  2. Time-domain bandpass filter (1-3 Hz via Butterworth)
  3. Frequency-domain harmonic zeroing: keeps the fundamental HR band
     (1.4-1.6 Hz) and zeros out its harmonics at n * [1.4, 1.6] Hz
     for n = 2, 3, 4, ... to clean the waveform without a traditional filter

Plots time-domain waveforms (raw, bandpass, harmonic-zeroed) and their
frequency spectra, and reports heart rate from each method.
"""
from scipy.signal import find_peaks, butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

ppg_waveform = np.loadtxt('HW5.1C.PPG.txt')

dt = 1/125
fs = 1 / dt  # 125 Hz
N = 4096


# addressing motion artifacts:
# we can exclude any harmonics that are outside of 40-220 bpm range
# i.e. 0.666 Hz < signal < 3.67 Hz

def bpm_from_peaks(peak_indices, dt):
    intervals = np.diff(peak_indices) * dt
    avg_interval = np.mean(intervals)
    return 60.0 / avg_interval

unfiltered_peak_indices, _ = find_peaks(ppg_waveform, width=15)
print("Number of peaks found in unfiltered waveform:", len(unfiltered_peak_indices), "peaks")
print("Unfiltered HR:", bpm_from_peaks(unfiltered_peak_indices, dt), "bpm")
t = np.arange(0, dt * N, dt)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9))
ax1.plot(t, ppg_waveform, label='PPG raw')
ax1.set_ylabel("Waveform (Raw ADC Samples)")
ax1.set_title("Raw Time Signal from PPG")
ax1.plot(unfiltered_peak_indices * dt, ppg_waveform[unfiltered_peak_indices], 'ro', markersize=4, label='Unfiltered peaks')
ax1.legend()
ax1.grid(True, alpha=0.3)
# Bandpass filter: keep only 60–180 BPM (1–3 Hz), remove DC and out-of-band motion
# We SPECIFY the filter by frequency (which band to keep), but we APPLY it in the TIME domain:
# butter() gives coefficients (b, a) for a difference equation. filtfilt() runs that
# equation on the signal sample-by-sample (convolution with the filter's impulse response).
# No FFT is used when applying the filter—only when we later plot the spectrum.
min_f, max_f = 1, 3  # Hz
order = 4
b, a = butter(order, [min_f, max_f], btype='band', fs=fs)
ppg_filtered = filtfilt(b, a, ppg_waveform)

peak_indices, _ = find_peaks(ppg_filtered, width=15)
print("Number of peaks found in bandpass filtered waveform:", len(peak_indices), "peaks")
print("Bandpass filtered HR:", bpm_from_peaks(peak_indices, dt), "bpm")
ax2.plot(t, ppg_filtered, label=f'PPG filtered {min_f:.2f}-{max_f:.2f} Hz')
ax2.plot(peak_indices * dt, ppg_filtered[peak_indices], 'ro', markersize=4, label='Filtered Peaks')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Waveform")
ax2.set_title("Bandpass (1-3Hz) Filtered Signal from PPG (no FFT filtering or manipulation)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# plot waveforms (pre and post bandpass filtered) in fourier domain 
ppg_demean_filtered = ppg_filtered - np.mean(ppg_filtered)
ppg_demean = ppg_waveform-np.mean(ppg_waveform)
freq = np.fft.fftfreq(N, dt)
freq_pos = freq[1:N//2]

# filter in frequency domain... don't apply hamming
fft_orig = np.fft.fft(ppg_demean)

# Keep the fundamental band (1.4–1.6 Hz) and zero out its harmonics
# Harmonics fall at n * [1.4, 1.6] Hz for n = 2, 3, 4, ...
fund_lo, fund_hi = 1.4, 1.6
fft_zeroed = fft_orig.copy()

# zero out DC drift and very low-frequency noise (below ~42 bpm)
fft_zeroed[np.abs(freq) < 0.6] = 0

# zero out each harmonic band: n * [fund_lo, fund_hi] for n = 2, 3, ...
n = 2
while n * fund_lo < fs / 2:
    harm_lo = n * fund_lo
    harm_hi = n * fund_hi
    fft_zeroed[(np.abs(freq) >= harm_lo) & (np.abs(freq) <= harm_hi)] = 0
    n += 1
ppg_zeroed = np.real(np.fft.ifft(fft_zeroed))
# Magnitude for frequency plot (after zeroing)
magnitude_zeroed = np.abs(fft_zeroed) / N
mag_zeroed_pos = magnitude_zeroed[1:N//2]

zeroed_peak_indices, _ = find_peaks(ppg_zeroed, width=15)
print("Number of peaks found in freq-domain filtered waveform:", len(zeroed_peak_indices), "peaks")
print("Freq-domain filtered HR:", bpm_from_peaks(zeroed_peak_indices, dt), "bpm")
ax3.plot(t, ppg_zeroed, label='PPG freq-domain filtered')
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Waveform")
ax3.set_title("Reconstructed Signal (inverse FFT after harmonics zeroed in fourier)")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print("***************** Frequency Domain Detection *********************")

ppg_hamming_filtered = ppg_demean_filtered * np.hamming(N)
ppg_hamming = ppg_demean*np.hamming(N)
magnitude_filtered = np.abs(np.fft.fft(ppg_hamming_filtered)) / N
magnitude = np.abs(np.fft.fft(ppg_hamming)) / N
mag_pos_filtered = magnitude_filtered[1:N//2]
mag_pos = magnitude[1:N//2]


fund_idx_raw = 1 + np.argmax(mag_pos)
fund_freq_raw = freq[fund_idx_raw]
fund_mag_raw = mag_pos[fund_idx_raw - 1]
print(f"Raw fundamental: {fund_freq_raw:.4f} Hz --> {fund_freq_raw*60:.2f} bpm")

fund_idx_bp = 1 + np.argmax(mag_pos_filtered)
fund_freq_bp = freq[fund_idx_bp]
fund_mag_bp = mag_pos_filtered[fund_idx_bp - 1]
print(f"Bandpass filtered fundamental: {fund_freq_bp:.4f} Hz --> {fund_freq_bp*60:.2f} bpm")

fund_idx_zeroed = 1 + np.argmax(mag_zeroed_pos)
fund_freq_zeroed = freq[fund_idx_zeroed]
fund_mag_zeroed = mag_zeroed_pos[fund_idx_zeroed - 1]
print(f"Freq-domain zeroed fundamental: {fund_freq_zeroed:.4f} Hz --> {fund_freq_zeroed*60:.2f} bpm")

fig, (ax4, ax5, ax6) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
ax4.stem(freq_pos, mag_pos, label="Hamming Pre-Filtered")
ax4.scatter(fund_freq_raw, fund_mag_raw, color='red', s=60, zorder=5, label=f'Fundamental ({fund_freq_raw:.4f} Hz)')
ax4.set_ylabel('Magnitude')
ax4.set_title('Frequency Domain of Raw Signal')
ax4.grid(True, alpha=0.3)

ax5.stem(freq_pos, mag_pos_filtered, label="Hamming Post Time Domain Bandpass Filtered")
ax5.scatter(fund_freq_bp, fund_mag_bp, color='red', s=60, zorder=5, label=f'Fundamental ({fund_freq_bp:.4f} Hz)')
ax5.set_ylabel('Magnitude')
ax5.set_title('Frequency Domain (after bandpass 1–3 Hz)')
ax5.set_xlim(0, 6)
ax5.grid(True, alpha=0.3)

ax6.stem(freq_pos, mag_zeroed_pos, label="Harmonics zeroed in Freq Domain")
ax6.scatter(fund_freq_zeroed, fund_mag_zeroed, color='red', s=60, zorder=5, label=f'Fundamental ({fund_freq_zeroed:.4f} Hz)')
ax6.set_ylabel('Magnitude')
ax6.set_title('Frequency Domain (harmonics zeroed)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
