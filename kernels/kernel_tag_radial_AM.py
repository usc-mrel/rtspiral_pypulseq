import numpy as np
import matplotlib.pyplot as plt

from kernels.kernel_crusher import kernel_crusher

from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_arbitrary_grad import make_arbitrary_grad

def kernel_tag_radial_AM(seq, prep_param, params, system):
    # Radial Tagging RF pulse. Converted from AM radial tagging pulse in RTHawk.
    # TODO: convert units to T and mT/m etc for consistency with the rest of the code.

    num_tags = prep_param['num_tags']
    tagging_time = prep_param['tagging_time']

    # use 0.2 of the max gradient
    g_amplitude = 0.2 * system.max_grad * 100 / system.gamma # in G/cm
    rf_amplitude = 0.120 # in G

    grad_sampling_rate = np.int32(1e-3 / system.grad_raster_time) # [samples/ms]
    rf_sampling_rate = np.int32(1e-3 / system.rf_raster_time) # [samples/ms]

    pulse_samples = int(tagging_time * grad_sampling_rate) + 1
    rf_pulse_samples = int(tagging_time * rf_sampling_rate) + 1

    slew_limit = system.max_slew / system.gamma / 10 # [G/cm/ms]

    tagging_rf_pulse = np.zeros(rf_pulse_samples)
    tagging_grad_pulse = np.zeros((pulse_samples, 2))

    grad_pulse_freq = np.pi / tagging_time # in round per ms
    rf_pulse_freq = grad_pulse_freq * num_tags # in round per ms

    for i in range(rf_pulse_samples):
        tagging_rf_pulse[i] = rf_amplitude * np.sin(
            rf_pulse_freq * (i - 1) * tagging_time / (rf_pulse_samples - 1)
        )

    for i in range(pulse_samples):
        tagging_grad_pulse[i, 0] = g_amplitude * np.sin(
            grad_pulse_freq * (i - 1) * tagging_time / (pulse_samples - 1)
        )
        tagging_grad_pulse[i, 1] = g_amplitude * np.cos(
            grad_pulse_freq * (i - 1) * tagging_time / (pulse_samples - 1)
        )
    
    n_points_add = int(np.ceil(tagging_grad_pulse[0, 1] / (slew_limit / 500)))
    gy_add_start = np.arange(0, tagging_grad_pulse[0, 1], tagging_grad_pulse[0, 1] / n_points_add)
    gx_add_start = np.zeros(n_points_add)
    rf_add_start = np.zeros(n_points_add)

    n_points_add = int(np.abs(np.ceil(tagging_grad_pulse[-1, 1] / (slew_limit / 500))))
    gy_add_end = np.arange(tagging_grad_pulse[-1, 1], 0, -tagging_grad_pulse[-1, 1] / n_points_add)
    gx_add_end = np.zeros(n_points_add)
    rf_add_end = np.zeros(n_points_add)

    gx = np.concatenate([gx_add_start, tagging_grad_pulse[:, 0], gx_add_end])
    gy = np.concatenate([gy_add_start, tagging_grad_pulse[:, 1], gy_add_end])
    rf = np.concatenate([rf_add_start, tagging_rf_pulse, rf_add_end])

    # convert to Hz/m and Hz
    gx = gx * 1e-2 * system.gamma # [G/cm] -> [Hz/m]
    gy = gy * 1e-2 * system.gamma # [G/cm] -> [Hz/m]
    rf = rf * 1e-4 * system.gamma # [G] -> [Hz]

    gx = make_arbitrary_grad(channel='x', waveform=gx, system=system)
    gy = make_arbitrary_grad(channel='y', waveform=gy, system=system)

    # flip angle is meaningless because of no_signal_scaling=True??
    rf = make_arbitrary_rf(rf, flip_angle=np.deg2rad(90), no_signal_scaling=True, phase_offset=np.deg2rad(90), time_bw_product=3, system=system)

    seq.add_block(rf, gx, gy)

    # add crusher
    kernel_crusher(seq, {'channel': 'z', 'cycles_per_thickness': 4}, params, system)

