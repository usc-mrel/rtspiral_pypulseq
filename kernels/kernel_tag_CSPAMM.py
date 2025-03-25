from pypulseq import make_block_pulse, make_trapezoid, calc_duration
import numpy as np


# the CSPAMM kernel is unique in that it has two prep tags... so we need different kernels for each.
def kernel_tag_CSPAMM(seq, prep_param, params, system):
    
    if "2ndpolarity" in prep_param:
        polarity = prep_param["2ndpolarity"]
    else:
        polarity = 1
    
    print("2ndpolarity: ", polarity)

    duration = 1.5e-3
    RF_x_1 = make_block_pulse(np.deg2rad(90), delay=0, duration=duration, time_bw_product=2,system=system)
    RF_x_2 = make_block_pulse(np.deg2rad(90), delay=0, duration=duration, phase_offset=np.deg2rad(180*polarity), time_bw_product=2,system=system)
    print("RF_x_2 phase offset: ", np.rad2deg(RF_x_2.phase_offset))

    clip_area = (1/prep_param['grid_tag_spacing']) * 100
    gradx = make_trapezoid(channel="x", area=clip_area, system=system)

    min_duration = 0.00028

    # hack to test if the gradx is too short.
    if calc_duration(gradx) <= min_duration:
        gradx = make_trapezoid(channel="x", area=clip_area, duration=min_duration,system=system)

    RF_y_1 = make_block_pulse(np.deg2rad(90), delay=0, duration=duration, time_bw_product=2, phase_offset=np.deg2rad(90),system=system)
    RF_y_2 = make_block_pulse(np.deg2rad(90), delay=0, duration=duration, time_bw_product=2, phase_offset=np.deg2rad(90 + (180*polarity)),system=system)
    print("RF_y_2 phase offset: ", np.rad2deg(RF_y_2.phase_offset))

    grady = make_trapezoid(channel="y", area=clip_area,system=system)

    # hack to test if the gradx is too short.
    if calc_duration(grady) <= min_duration:
        grady = make_trapezoid(channel="y", area=clip_area, duration=min_duration,system=system)

    # spoiler, so large area.
    thickness = params['acquisition']['slice_thickness'] * 1e-3 # [mm] -> [m]
    gradz_spoiler = make_trapezoid(channel="z", area=(12/thickness), system=system)

    # SPAMM tagging for x
    seq.add_block(RF_x_1)
    seq.add_block(gradx)
    seq.add_block(RF_x_2)

    seq.add_block(gradz_spoiler)

    # SPAMM tagging for y
    seq.add_block(RF_y_1)
    seq.add_block(grady)
    seq.add_block(RF_y_2)

    seq.add_block(gradz_spoiler)