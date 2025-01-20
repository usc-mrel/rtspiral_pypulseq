import numpy as np

def round_to_GRT(t, GRT=1e-5):
    # Rounds the input time to the nearest 1e-5. 
    # GRT = gradient raster time
    return round(t / GRT) * GRT

def round_up_to_GRT(t, GRT=1e-5):
    # Rounds up the input time to the nearest 1e-5.
    # GRT = gradient raster time
    return np.ceil(t / GRT) * GRT 