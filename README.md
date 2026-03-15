# PPG Analysis — HW5 Part 1

**Data:** `HW5.1A.PPG.txt` (clean) and `HW5.1C.PPG.txt` (motion-corrupted), 4096 samples at 125 Hz (~33 s).

## 1A — Peak Detection (`heart-rate-peak-detect.py`)

Uses `scipy.signal.find_peaks` (width=15) on the raw PPG waveform to locate heartbeat peaks. Computes:
- **Heart rate (BPM)** from average inter-peak interval
- **Max HRV** — longest single inter-peak interval
- **RMS HRV** — root-mean-square of successive inter-peak differences (RMSSD)

## 1B — FFT (`heart-rate-FFT.py`)

Determines heart rate in the frequency domain. Removes DC offset, applies a Hamming window to reduce spectral leakage, then takes the FFT. The fundamental (highest-magnitude) frequency bin maps directly to BPM. Plots spectra with and without windowing for comparison.

## 1C — Extra Credit: Motion Artifact Removal (`part_one_c_extra_credit.py`)

Analyzes `HW5.1C.PPG.txt`, which contains finger-tapping motion artifacts overlapping the HR band. Applies three filtering strategies and compares results via both peak-detected BPM and fundamental frequency:

1. **Time-domain bandpass (Butterworth)** — 4th-order 1–3 Hz bandpass via `filtfilt` to isolate the heart rate band.
2. **Frequency-domain zeroing** — FFT the demeaned signal (no window, to preserve amplitude for IFFT reconstruction), zero out bins outside the HR band and in the 3–4 Hz motion artifact region, then IFFT back to time domain.
3. **Spectral comparison** — Hamming-windowed FFT of raw, bandpass-filtered, and freq-zeroed signals plotted side-by-side with fundamental frequencies marked.
