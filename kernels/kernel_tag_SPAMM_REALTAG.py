from pypulseq import make_block_pulse, make_trapezoid 
from pypulseq.calc_duration import calc_duration
import numpy as np


def kernel_tag_SPAMM_REALTAG(seq, prep_param, params, system):
    duration = 1e-3
    RF_1_x = make_block_pulse(np.deg2rad(22.5), delay=0, time_bw_product=2, duration=duration,system=system)
    RF_2_x = make_block_pulse(np.deg2rad(67.5), delay=0, time_bw_product=2, duration=duration,system=system)

    clip_area = prep_param['grid_tag_spacing'] * 100

    gradx = make_trapezoid(channel="x", area=clip_area,system=system)

    # hack to test if the gradx is too short.
    if calc_duration(gradx) <= 0.0002:
        gradx = make_trapezoid(channel="x", area=clip_area, duration=0.00028,system=system)

    RF_1_y = make_block_pulse(np.deg2rad(22.5), delay=0, duration=duration, time_bw_product=2, phase_offset=np.deg2rad(90),system=system)
    RF_2_y = make_block_pulse(np.deg2rad(67.5), delay=0, duration=duration, time_bw_product=2, phase_offset=np.deg2rad(90),system=system)

    grady = make_trapezoid(channel="y", area=clip_area,system=system)

    # hack to test if the grady is too short.
    if calc_duration(grady) <= 0.0002:
        grady = make_trapezoid(channel="y", area=clip_area, duration=0.00028,system=system)

    # spoiler, so large area.
    thickness = params['acquisition']['slice_thickness'] * 1e-3 # [mm] -> [m]
    gradz_spoiler = make_trapezoid(channel="z", area=(12/thickness), system=system)

    # SPAMM tagging for x
    seq.add_block(RF_1_x)
    seq.add_block(gradx)
    seq.add_block(RF_2_x)
    seq.add_block(gradx)
    seq.add_block(RF_2_x)
    seq.add_block(gradx)
    seq.add_block(RF_1_x)
    seq.add_block(gradx)

    seq.add_block(gradz_spoiler)

    # SPAMM tagging for y
    seq.add_block(RF_1_y)
    seq.add_block(grady)
    seq.add_block(RF_2_y)
    seq.add_block(grady)
    seq.add_block(RF_2_y)
    seq.add_block(grady)
    seq.add_block(RF_1_y)
    seq.add_block(grady)

    seq.add_block(gradz_spoiler)