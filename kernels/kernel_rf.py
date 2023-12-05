from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_delay import make_delay
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.calc_duration import calc_duration
from libvds_rewind.rounding import round_up_to_GRT
import numpy as np

def kernel_rf(seq, prep_param, params, system):
    phase_offset = np.deg2rad(prep_param['phase'])

    flip_angle = prep_param['rf_scale']*params['acquisition']['flip_angle']/180*np.pi
  
    rf, gz, gzr = make_sinc_pulse(flip_angle=flip_angle, 
                                duration=params['acquisition']['rf_duration'],
                                slice_thickness=params['acquisition']['slice_thickness']*1e-3, # [mm] -> [m]
                                time_bw_product=2,
                                return_gz=True,
                                phase_offset=phase_offset,
                                use='excitation', system=system)
    
    seq.add_block(gzr) # hack to have the second rewinder.... should fix later.
    seq.add_block(rf, gz)
    seq.add_block(gzr)

    delay_value = prep_param['TR_scale']*params['acquisition']['TR']
    if delay_value != 0:
        delay_value = delay_value - (calc_duration(rf, gz) + calc_duration(gzr))
        assert delay_value >= 0, "Prep pulse delay is negative"
        delay_block = make_delay(round_up_to_GRT(delay_value, params['system']['grad_raster_time']))
        seq.add_block(delay_block)