from libvds.vds import vds_design
import matplotlib.pyplot as plt
from libvds.vds import plotgradinfo
from libvds_rewind.design_rewinder_exact_time import design_rewinder_exact_time
from libvds_rewind.design_rewinder_time_optimal import design_rewinder_time_optimal
from libvds_rewind.pts_to_waveform import pts_to_waveform

import numpy as np

sys = {
    'max_slew'  :  170,  # [T/m/s] 
    'max_grad'  :   38,  # [mT/m] 
    'adc_dwell' : 1e-6, # [s]
    'os'        :    8
    }

Nint = 19
fov  = [25.6] # [cm]
res = 2 # [mm]
Tread = 3e-3

k, g, s, t = vds_design(sys, Nint, fov, res, Tread)

# === design rewinder ===
# calculate moment of each gradient for rewinder M0-nulled.
M = np.cumsum(g, axis=0) * sys['adc_dwell']
[times_x, amplitudes_x] = design_rewinder_exact_time(g[-1, 0], 0, 1e-3, -M[-1,0], sys)
[times_y, amplitudes_y] = design_rewinder_exact_time(g[-1, 1], 0, 1e-3, -M[-1,1], sys)

sys['Tpud'] = 1e-3
sys['area_tol'] = 1e-4
[times_x, amplitudes_x] = design_rewinder_time_optimal(g[-1,0], 0, -M[-1,0], sys)
[times_y, amplitudes_y] = design_rewinder_time_optimal(g[-1,1], 0, -M[-1,1], sys)

g_rewind_x = pts_to_waveform(times_x, amplitudes_x, sys['adc_dwell'])
g_rewind_y = pts_to_waveform(times_y, amplitudes_y, sys['adc_dwell'])

# add zeros to the end of g_rewind_x or g_rewind_y to make them the same length (in case they are not).
if len(g_rewind_x) > len(g_rewind_y):
    g_rewind_y = np.concatenate((g_rewind_y, np.zeros(len(g_rewind_x) - len(g_rewind_y))))
else:
    g_rewind_x = np.concatenate((g_rewind_x, np.zeros(len(g_rewind_y) - len(g_rewind_x))))

# concatenate g and g_rewind, and plot.
g = np.concatenate((g, np.stack([g_rewind_x, g_rewind_y]).T))
plotgradinfo(g, sys['adc_dwell'])
plt.show()