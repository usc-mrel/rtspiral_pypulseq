import numpy as np

def pts_to_waveform(times, amplitudes, gradRasterTime):
    grd = np.arange(round(min(times) / gradRasterTime)+0.5, round(max(times) / gradRasterTime)+1.5) * gradRasterTime
    waveform = np.interp(grd, times, amplitudes)
    return waveform