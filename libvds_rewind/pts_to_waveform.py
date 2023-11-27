import numpy as np

def pts_to_waveform(times, amplitudes, gradRasterTime):
    # Adapted from pulseq pts2waveform. Removed the half GRT shift.
    grd = np.arange(round(min(times) / gradRasterTime), round(max(times) / gradRasterTime)) * gradRasterTime
    waveform = np.interp(grd, times, amplitudes)
    return waveform