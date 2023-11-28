import numpy as np
import matplotlib.pyplot as plt
from pypulseq import Opts
from utils.load_params import load_params
from libvds.vds import vds_fixed_ro, plotgradinfo

params = load_params('example_config', './')

system = Opts(
    max_grad = params['system']['max_grad'], grad_unit="mT/m",
    max_slew = params['system']['max_slew'], slew_unit="T/m/s",
    grad_raster_time = params['system']['grad_raster_time'],  # [s] ( 10 us)
    rf_raster_time   = params['system']['rf_raster_time'],    # [s] (  1 us)
    rf_ringdown_time = params['system']['rf_ringdown_time'],  # [s] ( 10 us)
    rf_dead_time     = params['system']['rf_dead_time'],      # [s] (100 us)
    adc_dead_time    = params['system']['adc_dead_time'],     # [s] ( 10 us)
)

spiral_sys = {
    'max_slew'  :  params['system']['max_slew'],        # [T/m/s] 
    'max_grad'  :  params['system']['max_grad'],        # [mT/m] 
    'adc_dwell' :  params['spiral']['adc_dwell'],  # [s]
    'os'        :  8
    }

fov   = [params['acquisition']['fov']] # [cm]
res   =  params['acquisition']['resolution'] # [mm]
Tread =  params['spiral']['ro_duration'] # [s]

k, g, t, n_int = vds_fixed_ro(spiral_sys, fov, res, Tread)

print(f'Number of interleaves for fully sampled trajectory: {n_int}.')

plotgradinfo(g, spiral_sys['adc_dwell'])
plt.show()