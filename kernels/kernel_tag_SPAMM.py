from pypulseq import make_block_pulse, make_trapezoid 
import numpy as np


def kernel_tag_SPAMM(seq, prep_param, params, system):
    RF_1_x = make_block_pulse(np.deg2rad(11.25), delay=0, time_bw_product=2, duration=1e-3,system=system)
    RF_2_x = make_block_pulse(np.deg2rad(33.75), delay=0, time_bw_product=2, duration=1e-3,system=system)

    clip_area = prep_param['grid_tag_spacing'] * 100

    gradx = make_trapezoid(channel="x", area=clip_area,system=system)

    RF_1_y = make_block_pulse(np.deg2rad(11.25), delay=0, duration=1e-3, time_bw_product=2, phase_offset=np.deg2rad(90),system=system)
    RF_2_y = make_block_pulse(np.deg2rad(33.75), delay=0, duration=1e-3, time_bw_product=2, phase_offset=np.deg2rad(90),system=system)

    grady = make_trapezoid(channel="y", area=clip_area,system=system)

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