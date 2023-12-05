from pypulseq import make_block_pulse, make_trapezoid 
import numpy as np


def kernel_tag_CSPAMM(seq, system):
    RF_1_x = make_block_pulse(np.deg2rad(22.5), delay=0, time_bw_product=2, duration=1e-3,system=system)
    RF_2_x = make_block_pulse(np.deg2rad(67.5), delay=0, time_bw_product=2, duration=1e-3,system=system)

    clip_area = 100

    gradx = make_trapezoid(channel="x", area=clip_area,system=system)

    RF_1_y = make_block_pulse(np.deg2rad(22.5), delay=0, duration=1e-3, time_bw_product=2, phase_offset=np.deg2rad(90),system=system)
    RF_2_y = make_block_pulse(np.deg2rad(67.5), delay=0, duration=1e-3, time_bw_product=2, phase_offset=np.deg2rad(90),system=system)

    grady = make_trapezoid(channel="y", area=clip_area,system=system)

    # spoiler, so large area.
    gradz_spoiler = make_trapezoid(channel="z", area=(clip_area * 13), system=system)

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