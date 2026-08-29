# %%
import argparse
import copy
import os
import warnings
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from libspiral import calcgradinfo, plotgradinfo, raster_to_grad, vds_fixed_ro
from pypulseq import (
    Opts,
    add_gradients,
    calc_duration,
    calc_rf_center,
    make_adc,
    make_arbitrary_grad,
    make_delay,
    make_digital_output_pulse,
    make_label,
    make_sinc_pulse,
    make_trapezoid,
    rotate,
)
from pypulseq.Sequence.sequence import Sequence

from kernels.kernel_handle_preparations import (
    kernel_handle_end_preparations,
    kernel_handle_preparations,
)
from librewinder.design_rewinder import design_rewinder
from utils import load_params, schedule_FA
from utils.traj_utils import save_metadata

# Cmd args
parser = argparse.ArgumentParser(
    prog="WriteSpiralSVR",
    description="Generates a 2D multislice spiral Pulseq sequence optimized for SVR.",
)

parser.add_argument(
    "-c", "--config", type=str, default="config", help="Config file path."
)

args = parser.parse_args()


print(f"Using config file: {args.config}.")
# Load and prep system and sequence parameters
params = load_params(args.config, "./")

system = Opts(
    max_grad=params["system"]["max_grad"],
    grad_unit="mT/m",
    max_slew=params["system"]["max_slew"],
    slew_unit="T/m/s",
    grad_raster_time=params["system"]["grad_raster_time"],  # [s] ( 10 us)
    rf_raster_time=params["system"]["rf_raster_time"],  # [s] (  1 us)
    rf_ringdown_time=params["system"]["rf_ringdown_time"],  # [s] ( 10 us)
    rf_dead_time=params["system"]["rf_dead_time"],  # [s] (100 us)
    adc_dead_time=params["system"]["adc_dead_time"],  # [s] ( 10 us)
)

GRT = params["system"]["grad_raster_time"]

spiral_sys = {
    "max_slew": params["system"]["max_slew"]
    * params["spiral"]["slew_ratio"],  # [T/m/s]
    "max_grad": params["system"]["max_grad"] * 0.99,  # [mT/m]
    "adc_dwell": params["spiral"]["adc_dwell"],  # [s]
    "grad_raster_time": GRT,  # [s]
    "os": 8,
}

fov = params["acquisition"]["fov"]  # [cm]
res = params["acquisition"]["resolution"]  # [mm]
Tread = params["spiral"]["ro_duration"]  # [s]

# Design the spiral trajectory
k, g, t, n_int = vds_fixed_ro(spiral_sys, fov, res, Tread)

"""
# convert the max grad and slew
spiral_sys['spiral_type']=0
k, g2, grew, s, t = spiralgen_design(spiral_sys, n_int, fov[0]*1e-2, res*1e-3, Tread)
g2 = np.concatenate((g2, grew), axis=0)
plotgradinfo(g2, GRT)
plt.savefig('grad2.pdf')
"""
print(f"Number of interleaves for fully sampled trajectory: {n_int}.")

t_grad, g_grad = raster_to_grad(g, spiral_sys["adc_dwell"], GRT)

if params["spiral"]["rotate_grads"]:
    g_rewind_x, g_rewind_y, g_grad = design_rewinder(
        g_grad,
        params["spiral"]["rewinder_time"],
        system,  # type: ignore
        slew_ratio=params["spiral"]["slew_ratio"],
        grad_rew_method=params["spiral"]["grad_rew_method"],
        M1_nulling=params["spiral"]["M1_nulling"],
        rotate_grads=params["spiral"]["rotate_grads"],
    )
else:
    g_rewind_x, g_rewind_y = design_rewinder(
        g_grad,
        params["spiral"]["rewinder_time"],
        system,  # type: ignore
        slew_ratio=params["spiral"]["slew_ratio"],
        grad_rew_method=params["spiral"]["grad_rew_method"],
        M1_nulling=params["spiral"]["M1_nulling"],
    )

# concatenate g and g_rewind, and plot.
g_grad = np.concatenate((g_grad, np.stack([g_rewind_x[0:], g_rewind_y[0:]]).T))

if params["user_settings"]["show_plots"]:
    _, _, _, m1, _, _, _ = calcgradinfo(g_grad, T=GRT)
    print(f"m1x: {m1[-1, 0]:.4f}, m1y: {m1[-1, 1]:.4f}")
    plotgradinfo(g_grad, GRT)
    plt.show()

# %%
# Excitation
tbwp = params["acquisition"]["tbwp"]
rf, gz, gzr = make_sinc_pulse(
    flip_angle=params["acquisition"]["flip_angle"] / 180 * np.pi,
    duration=params["acquisition"]["rf_duration"],
    slice_thickness=params["acquisition"]["slice_thickness"] * 1e-3,  # [mm] -> [m]
    time_bw_product=tbwp,
    return_gz=True,
    use="excitation",
    system=system,
)

"""
gzrr = copy.deepcopy(gzr)
gzrr.delay = 0 #gz.delay

if 'partial_dephasing' in params['acquisition']:
    dephasing = int(params['acquisition']['partial_dephasing'])
    use_dephasing = True
    moment_dephase = 1000*dephasing/params['acquisition']['slice_thickness']/(180*2)
    # re-do the gzrr gradient.
    gzrr = make_trapezoid(area=gzrr.area + moment_dephase, channel='z', system=system, max_grad=system.max_grad)

rf.delay = calc_duration(gzrr) + gz.rise_time
gz.delay = calc_duration(gzrr)
gzr.delay = calc_duration(gzrr, gz)
gzz = add_gradients([gzrr, gz, gzr], system=system)
"""

gzrr = copy.deepcopy(gzr)
gzr.delay = calc_duration(gz)
gzz = add_gradients([gz, gzr], system=system)

# ADC
ndiscard = 10  # Number of samples to discard from beginning
num_samples = np.floor(Tread / spiral_sys["adc_dwell"]) + ndiscard
adc = make_adc(num_samples, dwell=spiral_sys["adc_dwell"], delay=0, system=system)

# NOTE: we shift by GRT/2 and round to GRT because the grads will be shifted by GRT/2, and if we don't, last GRT/2 ADC samples discarded will be non-zero k-space.
# Basically we will miss the center of k-space. Caveat this way is, now we have GRT/2 ADC samples that are at 0, and we potentially lost 10 us, both are no biggie.
discard_delay_t = (
    ceil((ndiscard * spiral_sys["adc_dwell"] + GRT / 2) / GRT) * GRT
)  # [s] Time to delay grads.

# Readout gradients
gsp_x = make_arbitrary_grad(
    channel="x", waveform=g_grad[:, 0] * 42.58e3, delay=discard_delay_t, system=system
)  # [mT/m] -> [Hz/m]
gsp_x.first = 0
gsp_x.last = 0

gsp_y = make_arbitrary_grad(
    channel="y", waveform=g_grad[:, 1] * 42.58e3, delay=discard_delay_t, system=system
)  # [mT/m] -> [Hz/m]
gsp_y.first = 0
gsp_y.last = 0

# create a crusher gradient (only for FLASH)
if params["spiral"]["contrast"] == "FLASH" or params["spiral"]["contrast"] == "FISP":
    crush_area = (4 / (params["acquisition"]["slice_thickness"] * 1e-3)) + (
        -1 * gzr.area
    )
    rew_area = gzrr.area
    gzrr = make_trapezoid(
        channel="z", area=crush_area+rew_area, max_grad=system.max_grad, system=system
    )

# Set the Slice rewinder balance gradients delay
t_spiral_rewind = max(len(g_rewind_x), len(g_rewind_y)) * GRT
gzrr.delay = calc_duration(gsp_x, gsp_y, adc) - min(calc_duration(gzrr), t_spiral_rewind)

# set the rotations.
gsp_xs = []
gsp_ys = []
print(f"Spiral arm ordering is {params['spiral']['arm_ordering']}.")
if params["spiral"]["arm_ordering"] == "linear":
    if (n_int % 2) == 1 and (params["acquisition"]["repetitions"] % 2) == 1:
        warnings.warn(
            "Number of interleaves is odd. To solve this, we increased it by 1. If this is undesired, please set repetitions to an even number instead."
        )
        n_int += 1
    for i in range(0, n_int):
        gsp_x_rot, gsp_y_rot = rotate(
            gsp_x, gsp_y, axis="z", angle=2 * np.pi * i / n_int
        )
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        params["spiral"]["GA_angle"] = 360 / n_int
    n_TRs = int(n_int * params["acquisition"]["repetitions"])
elif params["spiral"]["arm_ordering"] == "ga":
    n_int = params["spiral"]["GA_steps"]
    if (n_int % 2) == 1 and (params["acquisition"]["repetitions"] % 2) == 1:
        warnings.warn(
            """
                      =================================================================================
                      Number of arms in the sequence is odd. This may create steady state artifacts
                      during the imaging with multiple runs, due to RF phase not alternating properly.
                      To avoid this issue, set repetitions to an even number.
                      =================================================================================
                      """
        )

    ang = 0
    for i in range(0, n_int):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=ang)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        ang += params["spiral"]["GA_angle"] * np.pi / 180
        ang = ang % (2 * np.pi)
        # print(f"Deg: {ang*180/np.pi}")
    n_TRs = int(n_int * params["acquisition"]["repetitions"])
elif params["spiral"]["arm_ordering"] == "linear_custom":
    assert n_int == len(params["spiral"]["custom_order"]), (
        "number of interleaves does not match custom order!"
    )
    view_order = params["spiral"]["custom_order"]
    for i in range(0, n_int):
        gsp_x_rot, gsp_y_rot = rotate(
            gsp_x, gsp_y, axis="z", angle=2 * np.pi * i / n_int
        )
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        params["spiral"]["GA_angle"] = 360 / n_int

    # re-order using the custom view order
    gsp_xs[:] = [gsp_xs[d] for d in view_order]
    gsp_ys[:] = [gsp_ys[d] for d in view_order]
    n_TRs = n_int * params["acquisition"]["repetitions"]
else:
    raise Exception("Unknown arm ordering")

# Set the delays
# TE
if params["acquisition"]["TE"] == 0:
    TEd = 0
    TE = (
        rf.shape_dur
        - calc_rf_center(rf)[0]
        + calc_duration(gzr)
        - gzr.delay
        + gsp_x.delay
    )
    print(f"Min TE is set: {TE * 1e3:.3f} ms.")
    params["acquisition"]["TE"] = TE
else:
    TE = params["acquisition"]["TE"] * 1e-3
    TEd = TE - (rf.shape_dur - calc_rf_center(rf)[0] + calc_duration(gzr) + gsp_x.delay)
    assert TEd >= 0, "Required TE can not be achieved."

# TR
if params["acquisition"]["TR"] == 0:
    TRd = 0
    TR = calc_duration(rf, gzz) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc, gzrr)
    print(f"Min TR is set: {TR * 1e3:.3f} ms.")
    params["acquisition"]["TR"] = TR
else:
    TR = params["acquisition"]["TR"] * 1e-3
    TRd = TR - (calc_duration(rf, gzz) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc, gzrr))
    assert TRd >= 0, "Required TR can not be achieved."

TE_delay = make_delay(TEd)
TR_delay = make_delay(TRd)

# Sequence looping
seq = Sequence(system)

# handle any preparation pulses.
prep_str = kernel_handle_preparations(seq, params, system, rf=rf, gz=gzz)

# useful for end_peparation pulses.
params["flip_angle_last"] = params["acquisition"]["flip_angle"]

# tagging pulse pre-prep (only if fa_schedule exists)
rf_amplitudes, FA_schedule_str = schedule_FA(params, n_TRs)

# used for FLASH only: set rf spoiling increment.
rf_phase = 0
rf_inc = 0

if params["spiral"]["contrast"] == "FLASH":
    linear_phase_increment = 0
    quadratic_phase_increment = np.deg2rad(117)
elif params["spiral"]["contrast"] in ("trueFISP", "FISP"):
    linear_phase_increment = np.deg2rad(180)
    quadratic_phase_increment = 0
else:
    print("Unknown contrast type. Assuming trueFISP.")
    linear_phase_increment = np.deg2rad(180)
    quadratic_phase_increment = 0
    params["spiral"]["contrast"] = "trueFISP"

enable_trigger = True

trig = make_digital_output_pulse(channel="ext1", duration=0.001, system=system)

# loop the multi-slice sequence.
hz_per_thickness = gz.amplitude * params["acquisition"]["slice_shift"] * 1e-3

if "num_slices" in params["acquisition"]:
    n_slices = params["acquisition"]["num_slices"]
    slice_range = range(-n_slices // 2, n_slices // 2)
else:
    n_slices = 1
    slice_range = [0]


arm_i = 0
for slice_i in slice_range:
    rf.freq_offset = hz_per_thickness * slice_i
    slice_phase_corr = np.mod(-2 * np.pi * rf.freq_offset * rf.center, 2*np.pi) # compensate for the slice-offset induced phase
    _, rf.shape_IDs = seq.register_rf_event(rf)
    # seq.add_block(make_label('SLC', 'SET', slice_i + abs(min(slice_range))))
    for _ in range(0, n_TRs):
        curr_rf = copy.deepcopy(rf)

        # check if we are using a rammped FA scheme (rf_amplitudes is a list [])
        if len(rf_amplitudes) > 0:
            if arm_i >= len(rf_amplitudes):
                n_TRs = arm_i
                break
            curr_rf.signal = (
                rf.signal
                * rf_amplitudes[arm_i]
                / np.deg2rad(params["acquisition"]["flip_angle"])
            )

        curr_rf.phase_offset = np.mod(rf_phase + slice_phase_corr, 2 * np.pi)
        adc.phase_offset = rf_phase

        rf_inc = np.mod(rf_inc + quadratic_phase_increment, np.pi * 2)
        rf_phase = np.mod(rf_phase + linear_phase_increment + rf_inc, np.pi * 2)

        if enable_trigger is True:
            seq.add_block(trig, curr_rf, gzz)
        else:
            seq.add_block(curr_rf, gzz)

        seq.add_block(TE_delay)
        if params["spiral"]["arm_ordering"] == "ga":
            seq.add_block(
                make_label("LIN", "SET", arm_i % params["spiral"]["GA_steps"])
            )
            seq.add_block(
                make_label("REP", "SET", arm_i // params["spiral"]["GA_steps"])
            )
            seq.add_block(
                gsp_xs[arm_i % params["spiral"]["GA_steps"]],
                gsp_ys[arm_i % params["spiral"]["GA_steps"]],
                adc,
                gzrr,
            )
        else:
            seq.add_block(make_label("LIN", "SET", arm_i % n_int))
            seq.add_block(gsp_xs[arm_i % n_int], gsp_ys[arm_i % n_int], adc, gzrr)

        seq.add_block(TR_delay)

        if arm_i == n_int - 1:
            arm_i = 0
        else:
            arm_i += 1

# handle any end_preparation pulses.
end_prep_str = kernel_handle_end_preparations(seq, params, system, rf=rf, gz=gzz)

# Quick timing check
ok, error_report = seq.check_timing()

if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed. Error listing follows:")
    [print(e) for e in error_report]

# Plot the sequence
if params["user_settings"]["show_plots"]:
    seq.plot(
        show_blocks=True,
        grad_disp="mT/m",
        plot_now=False,
        time_disp="ms",
        time_range=(0, 20 * TR),
        stacked=True,
    )
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    plt.figure()
    plt.plot(
        k_traj[0, : int(num_samples * n_TRs)], k_traj[1, : int(num_samples * n_TRs)]
    )

    # make axis suqaure
    plt.gca().set_aspect("equal", adjustable="box")
    # double fontsize
    plt.rcParams.update({"font.size": 14})

    # plt.plot(k_traj_adc[0,:], k_traj_adc[1,:], 'rx')
    plt.xlabel("$k_x [mm^{-1}]$")
    plt.ylabel("$k_y [mm^{-1}]$")
    plt.title("k-Space Trajectory")

    if (
        "acoustic_resonances" in params
        and "frequencies" in params["acoustic_resonances"]
    ):
        resonances = []
        for idx in range(len(params["acoustic_resonances"]["frequencies"])):
            resonances.append(
                {
                    "frequency": params["acoustic_resonances"]["frequencies"][idx],
                    "bandwidth": params["acoustic_resonances"]["bandwidths"][idx],
                }
            )
        seq.calculate_gradient_spectrum(acoustic_resonances=resonances)
        plt.title("Gradient spectrum")
    plt.show()


# Detailed report if requested
if params["user_settings"]["detailed_rep"]:
    print("\n===== Detailed Test Report =====\n")
    rep_str = seq.test_report()
    print(rep_str)

# Write the sequence to file
if params["user_settings"]["write_seq"]:
    if "num_slices" in params["acquisition"]:
        slab_thickness = (
            params["acquisition"]["slice_shift"]
            * 1e-3
            * int(params["acquisition"]["num_slices"])
        ) + (
            params["acquisition"]["slice_thickness"]
            - params["acquisition"]["slice_shift"]
        ) * 1e-3
        print(f"Slab thickness: {slab_thickness}")
        seq.set_definition(
            key="FOV", value=[fov[0] * 1e-2, fov[0] * 1e-2, slab_thickness]
        )

        # Siemens does not allow overlapping slices, so slice thickness is set to min(slice_thickness, slice_shift)
        # seq.set_definition(key="SliceThickness", value=min(params['acquisition']['slice_thickness']*1e-3, params['acquisition']['slice_shift']*1e-3))
        seq.set_definition(key="SliceThickness", value=slab_thickness)

        # Similarly, slice gap is not allowed to be negative.
        # seq.set_definition(key='SliceGap', value=max((params['acquisition']['slice_shift']-params['acquisition']['slice_thickness'])*1e-3, 0.0))
        # seq.set_definition(key='SlicePositions', value=[i*params['acquisition']['slice_shift']*1e-3 for i in slice_range])
    else:
        seq.set_definition(
            key="SliceThickness", value=params["acquisition"]["slice_thickness"] * 1e-3
        )
        seq.set_definition(
            key="FOV",
            value=[
                fov[0] * 1e-2,
                fov[0] * 1e-2,
                params["acquisition"]["slice_thickness"] * 1e-3,
            ],
        )

    seq.set_definition(key="Name", value="sprssfp")
    seq.set_definition(key="TE", value=TE)
    seq.set_definition(key="TR", value=TR)
    seq.set_definition(key="FA", value=params["acquisition"]["flip_angle"])
    seq.set_definition(key="Resolution_mm", value=res)
    seq_filename = f"spiral_{params['spiral']['contrast']}{FA_schedule_str}{prep_str}{end_prep_str}_{params['spiral']['arm_ordering']}{params['spiral']['GA_angle']:.4f}_nTR{n_TRs}_Tread{params['spiral']['ro_duration'] * 1e3:.2f}_TR{TR * 1e3:.2f}ms_FA{params['acquisition']['flip_angle']}_tbwp{params['acquisition']['tbwp']}_shift{params['acquisition']['slice_shift']}_{params['user_settings']['filename_ext']}"

    # remove double, triple, quadruple underscores, and trailing underscores
    seq_filename = (
        seq_filename.replace("__", "_").replace("__", "_").replace("__", "_").strip("_")
    )

    seq_path = os.path.join("out_seq", f"{seq_filename}.seq")

    # ensure the out_seq directory exists before writing.
    os.makedirs("out_seq", exist_ok=True)

    seq.write(seq_path)  # Save to disk

    # Export k-space trajectory
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    # save_traj_dcf(seq.signature_value, k_traj_adc, n_TRs, n_int, fov, res, ndiscard, params['user_settings']['show_plots'])
    params_save = {
        "adc_dwell": spiral_sys["adc_dwell"],
        "ndiscard": ndiscard,
        "n_TRs": n_int,
        "n_int": n_int,
        "ga_rotation": params["spiral"]["GA_angle"],
        "fov": fov,
        "spatial_resolution": res,
        "arm_ordering": params["spiral"]["arm_ordering"],
        "n_slices": int(params["acquisition"]["num_slices"]),
        "slice_shift": params["acquisition"]["slice_shift"],
    }

    # truncate trajectory to only include unique arms
    # n_slices = int(params['acquisition']['num_slices'])
    k_traj_adc = k_traj_adc[:, 0 : int(num_samples * n_int)]

    save_metadata(
        seq.signature_value,
        k_traj_adc,
        params_save,
        params["user_settings"]["show_plots"],
        dcf_method="hoge",
        out_dir="out_trajectory",
    )

    print(
        f"Metadata file for {seq_filename} is saved as {seq.signature_value} in out_trajectory/."
    )


# %%
