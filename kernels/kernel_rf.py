from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_delay import make_delay
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.calc_duration import calc_duration
from pypulseq.add_gradients import add_gradients
from libspiralutils import round_up_to_GRT
import numpy as np
import copy

def kernel_rf(seq, prep_param, params, system, flip_angle=None, rf=None, gz=None):
    phase_offset = np.deg2rad(prep_param['phase'])

    scale = prep_param['rf_scale']
    # if scale is negative, use the last flip angle.
    if scale < 0:
        fa_last = params["flip_angle_last"]
        fa = params["acquisition"]["flip_angle"]
        scale = (fa_last/fa) * np.abs(scale)
    else:
        fa = params["acquisition"]["flip_angle"]

    if rf is None or gz is None:
        if flip_angle is None:
            flip_angle = np.abs(scale)*params['acquisition']['flip_angle']/180*np.pi
        rf, gz, gzr = make_sinc_pulse(flip_angle=flip_angle, 
                                    duration=params['acquisition']['rf_duration'],
                                    slice_thickness=params['acquisition']['slice_thickness']*1e-3, # [mm] -> [m]
                                    time_bw_product=2,
                                    return_gz=True,
                                    phase_offset=phase_offset,
                                    use='excitation', system=system)
        gzrr = copy.deepcopy(gzr)
        gzz = add_gradients([gzrr, gz, gzr], system=system)

        gzrr.delay = 0 #gz.delay
        rf.delay = calc_duration(gzrr) + gz.rise_time
        gz.delay = calc_duration(gzrr)
        gzr.delay = calc_duration(gzrr, gz)
        seq.add_block(rf, gzz)
    else:
        rf_2 = copy.deepcopy(rf)
        rf_2.signal = rf_2.signal * scale
        rf_2.phase_offset = phase_offset
        seq.add_block(rf_2, gz)

    delay_value = prep_param['TR_scale']*params['acquisition']['TR']
    if delay_value != 0:
        delay_value = delay_value - (calc_duration(rf, gz))
        assert delay_value >= 0, "Prep pulse delay is negative"
        delay_block = make_delay(round_up_to_GRT(delay_value, params['system']['grad_raster_time']))
        seq.add_block(delay_block)