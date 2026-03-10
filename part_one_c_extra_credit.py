"""
Motion artifact added...
"""
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

ppg_waveform = np.loadtxt('HW5.1C.PPG.txt')