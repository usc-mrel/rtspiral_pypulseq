import numpy as np
import matplotlib.pyplot as plt
from pypulseq import Opts
from pypulseq import (make_adc, make_sinc_pulse, make_digital_output_pulse, make_delay, 
                      make_arbitrary_grad, make_trapezoid, make_extended_trapezoid_area, 
                      calc_duration, calc_rf_center, 
                      rotate, add_gradients, make_label)
from pypulseq.Sequence.sequence import Sequence
from utils import schedule_FA, load_params
from utils.traj_utils import save_metadata
from libspiral import vds_fixed_ro, plotgradinfo, raster_to_grad
from libspiralutils import pts_to_waveform, design_rewinder_exact_time
from kernels.kernel_handle_preparations import kernel_handle_preparations, kernel_handle_end_preparations
from math import ceil
import copy
import argparse
import os
import warnings

# Cmd args
parser = argparse.ArgumentParser(
                    prog='Write2DRTSpiral',
                    description='Generates a 2D real-time spiral Pulseq sequence for given parameters.')

parser.add_argument('-c', '--config', type=str, default='config', help='Config file path.')

args = parser.parse_args()


print(f'Using config file: {args.config}.')
# Load and prep system and sequence parameters
params = load_params(f"{args.config}", './')

system = Opts(
    max_grad = params['system']['max_grad'], grad_unit="mT/m",
    max_slew = params['system']['max_slew'], slew_unit="T/m/s",
    grad_raster_time = params['system']['grad_raster_time'],  # [s] ( 10 us)
    rf_raster_time   = params['system']['rf_raster_time'],    # [s] (  1 us)
    rf_ringdown_time = params['system']['rf_ringdown_time'],  # [s] ( 10 us)
    rf_dead_time     = params['system']['rf_dead_time'],      # [s] (100 us)
    adc_dead_time    = params['system']['adc_dead_time'],     # [s] ( 10 us)
)

GRT = params['system']['grad_raster_time']

spiral_sys = {
    'max_slew'          :  params['system']['max_slew']*params['spiral']['slew_ratio'],   # [T/m/s] 
    'max_grad'          :  params['system']['max_grad']*0.99,   # [mT/m] 
    'adc_dwell'         :  params['spiral']['adc_dwell'],  # [s]
    'grad_raster_time'  :  GRT, # [s]
    'os'                :  8
    }

fov   = params['acquisition']['fov'] # [cm]
res   = params['acquisition']['resolution'] # [mm]
Tread = params['spiral']['ro_duration'] # [s]


# Design the spiral trajectory
k, g, t, n_int = vds_fixed_ro(spiral_sys, fov, res, Tread)

print(f'Number of interleaves for fully sampled trajectory: {n_int}.')

t_grad, g_grad = raster_to_grad(g, spiral_sys['adc_dwell'], GRT)

# === design rewinder ===
M = np.cumsum(g_grad, axis=0) * GRT

grad_rew_method = params['spiral']['grad_rew_method']
T_rew = params['spiral']['rewinder_time']
# Design rew with gropt
if grad_rew_method == 'gropt':
    from gropt import get_min_TE_gfix

    # Method 1: GrOpt, separate optimization
    gropt_params = {}
    gropt_params['mode'] = 'free'
    gropt_params['gmax'] = params['system']['max_grad']*0.77*1e-3 # [mT/m] -> [T/m]
    gropt_params['smax'] = params['system']['max_slew']*params['spiral']['slew_ratio']
    gropt_params['dt']   = GRT

    gropt_params['moment_params']  = [[0, 0, 0, -1, -1, -M[-1,0]*1e3, 1.0e-3]]
    gropt_params['gfix']  = np.array([g_grad[-1, 0]*1e-3, -99999, 0])

    g_rewind_x, T = get_min_TE_gfix(gropt_params, T_rew*1e3, True)
    g_rewind_x = g_rewind_x.T[:,0]*1e3

    gropt_params['moment_params']  = [[0, 0, 0, -1, -1, -M[-1,1]*1e3, 1.0e-3]]
    gropt_params['gfix']  = np.array([g_grad[-1, 1]*1e-3, -99999, 0])

    g_rewind_y, T = get_min_TE_gfix(gropt_params, T_rew*1e3, True)
    g_rewind_y = g_rewind_y.T[:,0]*1e3

elif grad_rew_method == 'ext_trap_area':
    from pypulseq.make_extended_trapezoid_area import make_extended_trapezoid_area

    # Copy the system to modify slew rate to obey reduced SR of the spirals.
    system2 = copy.deepcopy(system)
    system2.max_slew = system.max_slew*params['spiral']['slew_ratio']
    _,times_x,amplitudes_x = make_extended_trapezoid_area(channel='x', area=-M[-1,0]*system2.gamma*1e-3, grad_start=g_grad[-1, 0]*system2.gamma*1e-3, grad_end=0, system=system2)
    _,times_y,amplitudes_y = make_extended_trapezoid_area(channel='y', area=-M[-1,1]*system2.gamma*1e-3, grad_start=g_grad[-1, 1]*system2.gamma*1e-3, grad_end=0, system=system2)

    g_rewind_x = 1e3*pts_to_waveform(times_x, amplitudes_x, GRT)/system2.gamma
    g_rewind_y = 1e3*pts_to_waveform(times_y, amplitudes_y, GRT)/system2.gamma

elif grad_rew_method == 'exact_time':

    [times_x, amplitudes_x] = design_rewinder_exact_time(g_grad[-1, 0], 0, T_rew, -M[-1,0], spiral_sys)
    [times_y, amplitudes_y] = design_rewinder_exact_time(g_grad[-1, 1], 0, T_rew, -M[-1,1], spiral_sys)

    g_rewind_x = pts_to_waveform(times_x, amplitudes_x, GRT)
    g_rewind_y = pts_to_waveform(times_y, amplitudes_y, GRT)

# add zeros to the end of g_rewind_x or g_rewind_y to make them the same length (in case they are not).
if len(g_rewind_x) > len(g_rewind_y):
    g_rewind_y = np.concatenate((g_rewind_y, np.zeros(len(g_rewind_x) - len(g_rewind_y))))
else:
    g_rewind_x = np.concatenate((g_rewind_x, np.zeros(len(g_rewind_y) - len(g_rewind_x))))

# concatenate g and g_rewind, and plot.
g_grad = np.concatenate((g_grad, np.stack([g_rewind_x[0:], g_rewind_y[0:]]).T))

if params['user_settings']['show_plots']:
    plotgradinfo(g_grad, GRT)
    plt.show()

# Excitation
rf, gz, gzr = make_sinc_pulse(flip_angle=params['acquisition']['flip_angle']/180*np.pi, 
                                duration=params['acquisition']['rf_duration'],
                                slice_thickness=params['acquisition']['slice_thickness']*1e-3, # [mm] -> [m]
                                time_bw_product=2,
                                return_gz=True,
                                use='excitation', system=system)

gzrr = copy.deepcopy(gzr)
gzrr.delay = 0 #gz.delay
rf.delay = calc_duration(gzrr) + gz.rise_time
gz.delay = calc_duration(gzrr)
gzr.delay = calc_duration(gzrr, gz)
gzz = add_gradients([gzrr, gz, gzr], system=system)

# ADC
ndiscard = 10 # Number of samples to discard from beginning
num_samples = np.floor(Tread/spiral_sys['adc_dwell']) + ndiscard
adc = make_adc(num_samples, dwell=spiral_sys['adc_dwell'], delay=0, system=system)

# NOTE: we shift by GRT/2 and round to GRT because the grads will be shifted by GRT/2, and if we don't, last GRT/2 ADC samples discarded will be non-zero k-space.
# Basically we will miss the center of k-space. Caveat this way is, now we have GRT/2 ADC samples that are at 0, and we potentially lost 10 us, both are no biggie.
discard_delay_t = ceil((ndiscard*spiral_sys['adc_dwell']+GRT/2)/GRT)*GRT # [s] Time to delay grads.

# Readout gradients
gsp_x = make_arbitrary_grad(channel='x', waveform=g_grad[:,0]*42.58e3, delay=discard_delay_t, system=system) # [mT/m] -> [Hz/m]
gsp_x.first = 0
gsp_x.last = 0

gsp_y = make_arbitrary_grad(channel='y', waveform=g_grad[:,1]*42.58e3, delay=discard_delay_t, system=system) # [mT/m] -> [Hz/m]
gsp_y.first = 0
gsp_y.last = 0

# Set the Slice rewinder balance gradients delay
gzrr.delay = calc_duration(gsp_x, gsp_y, adc)

# create a crusher gradient (only for FLASH)
if params['spiral']['contrast'] == 'FLASH' or params['spiral']['contrast'] == 'FISP':
    crush_area = (4 / (params['acquisition']['slice_thickness'] * 1e-3)) + (-1 * gzr.area)
    gz_crush = make_trapezoid(channel='z', 
                              area=crush_area, 
                              max_grad=system.max_grad, 
                              system=system)

# set the rotations.
gsp_xs = []
gsp_ys = []
print(f"Spiral arm ordering is {params['spiral']['arm_ordering']}.")
if params['spiral']['arm_ordering'] == 'linear':
    if (n_int%2) == 1 and (params['acquisition']['repetitions']%2) == 1:
        warnings.warn("Number of interleaves is odd. To solve this, we increased it by 1. If this is undesired, please set repetitions to an even number instead.")
        n_int += 1
    for i in range(0, n_int):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=2*np.pi*i/n_int)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        params['spiral']['GA_angle'] = 360/n_int
    n_TRs = n_int * params['acquisition']['repetitions']
elif params['spiral']['arm_ordering'] == 'ga':
    n_TRs = params['spiral']['GA_steps']
    if (n_TRs%2) == 1 and (params['acquisition']['repetitions']%2) == 1:
        warnings.warn(
                    '''
                      ========================================
                      Number of arms in the sequence is odd.
                      This may create steady state artifacts during the imaging with multiple runs, due to RF phase not alternating properly.
                      To avoid this issue, set repetitions to an even number.
                      ========================================
                      ''')

    n_int = n_TRs
    ang = 0
    for i in range(0, n_TRs):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=ang)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        ang += params['spiral']['GA_angle']*np.pi/180
        ang = ang % (2*np.pi)
        # print(f"Deg: {ang*180/np.pi}")
    n_TRs = n_int * params['acquisition']['repetitions']
elif params['spiral']['arm_ordering'] == 'linear_custom':
    assert n_int == len(params['spiral']['custom_order']), "number of interleaves does not match custom order!"
    view_order = params['spiral']['custom_order']
    for i in range(0, n_int):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=2*np.pi*i/n_int)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        params['spiral']['GA_angle'] = 360/n_int

    # re-order using the custom view order
    gsp_xs[:] = [gsp_xs[d] for d in view_order]
    gsp_ys[:] = [gsp_ys[d] for d in view_order]
    n_TRs = n_int * params['acquisition']['repetitions']
else:
    raise Exception("Unknown arm ordering") 

# Set the delays
# TE 
if params['acquisition']['TE'] == 0:
    TEd = 0
    TE = rf.shape_dur - calc_rf_center(rf)[0] + calc_duration(gzr) - gzr.delay + gsp_x.delay
    print(f'Min TE is set: {TE*1e3:.3f} ms.')
    params['acquisition']['TE'] = TE
else:
    TE = params['acquisition']['TE']*1e-3
    TEd = TE - (rf.shape_dur - calc_rf_center(rf)[0] + calc_duration(gzr) + gsp_x.delay)
    assert TEd >= 0, "Required TE can not be achieved."

# TR
if params['acquisition']['TR'] == 0:
    TRd = 0
    TR = calc_duration(rf, gzz) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc)
    if params['spiral']['contrast'] in ('FLASH', 'FISP'):
        TR = TR + calc_duration(gz_crush)
    print(f'Min TR is set: {TR*1e3:.3f} ms.')
    params['acquisition']['TR'] = TR
else:
    TR = params['acquisition']['TR']*1e-3
    TRd = TR - (calc_duration(rf, gzz) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc))
    if params['spiral']['contrast'] in ('FLASH', 'FISP'):
        TRd = TRd - calc_duration(gz_crush)
    assert TRd >= 0, "Required TR can not be achieved."

TE_delay = make_delay(TEd)
TR_delay = make_delay(TRd)
# Sequence looping

seq = Sequence(system)

# grab CINE parameters
# check if acquisition.cine exists
if 'cine' in params['acquisition']:
    n_phases = params['acquisition']['cine']['cardiac_phases']
    n_tr_per_phase = int(params['acquisition']['cine']['n_tr_per_phase'])

# useful for end_peparation pulses.
params['flip_angle_last'] = params['acquisition']['flip_angle']

# tagging pulse pre-prep (only if fa_schedule exists)
rf_amplitudes, FA_schedule_str = schedule_FA(params, n_TRs)

# used for FLASH only: set rf spoiling increment.
rf_phase = 0
rf_inc = 0

if params['spiral']['contrast'] == 'FLASH':
    linear_phase_increment = 0
    quadratic_phase_increment = np.deg2rad(117)
elif params['spiral']['contrast'] in ('trueFISP', 'FISP'):
    linear_phase_increment = np.deg2rad(180)
    quadratic_phase_increment = 0
else:
    print("Unknown contrast type. Assuming trueFISP.")
    linear_phase_increment = np.deg2rad(180)
    quadratic_phase_increment = 0
    params['spiral']['contrast'] = 'trueFISP'


# CINE trigger
from pypulseq.make_trigger import make_trigger
trigger = make_trigger('physio1', duration=2000e-6, system=system)

# Start the CINE acquisition
n_TRs = 0

n_rep = params['acquisition']['repetitions'] 

for rep in range(n_rep):

    _, rf.shape_IDs = seq.register_rf_event(rf)
    for idx in range(np.ceil(len(gsp_xs)/n_tr_per_phase).astype(int)):
        # add trigger FIRST
        seq.add_block(trigger) 

        # handle any preparation pulses.
        prep_str = kernel_handle_preparations(seq, params, system, rf=rf, gz=gzz, rep=rep)

        for phase_i in range(0, n_phases):
            for tr_i in range(0, n_tr_per_phase):

                rf_idx = tr_i + (phase_i * n_tr_per_phase)
                idx_ = idx*n_tr_per_phase + tr_i

                # LABEL EXTENSIONS
                seq.add_block(make_label('LIN', 'SET', idx_ % n_int))

                curr_rf = copy.deepcopy(rf)
                # check if we are using a rammped FA scheme (rf_amplitudes is a list []) 
                if len(rf_amplitudes) > 0:
                    if idx_ >= len(rf_amplitudes):
                        n_TRs += idx_ 
                        break
                    curr_rf.signal = rf.signal * rf_amplitudes[rf_idx] / np.deg2rad(params['acquisition']['flip_angle'])
                n_TRs += 1
                curr_rf.phase_offset = rf_phase
                adc.phase_offset = rf_phase
                rf_inc = np.mod(rf_inc + quadratic_phase_increment, np.pi * 2)
                rf_phase = np.mod(rf_phase + linear_phase_increment + rf_inc, np.pi * 2)

                seq.add_block(curr_rf, gzz)
                seq.add_block(TE_delay)
                if params['spiral']['arm_ordering'] == 'ga':
                    seq.add_block(gsp_xs[idx_ % params['spiral']['GA_steps']], gsp_ys[idx_ % params['spiral']['GA_steps']], adc) 
                else:
                    seq.add_block(gsp_xs[idx_ % n_int], gsp_ys[idx_% n_int], adc)
                if params['spiral']['contrast'] in ('FLASH', 'FISP'):
                    seq.add_block(gz_crush)
                seq.add_block(TR_delay) 

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
if params['user_settings']['show_plots']:
    seq.plot(show_blocks=True, grad_disp='mT/m', plot_now=False, time_disp='ms')
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    plt.figure()
    plt.plot(k_traj[0,:], k_traj[1, :])

    # make axis suqaure
    plt.gca().set_aspect('equal', adjustable='box')
    # double fontsize
    plt.rcParams.update({'font.size': 14})

    #plt.plot(k_traj_adc[0,:], k_traj_adc[1,:], 'rx')
    plt.xlabel('$k_x [mm^{-1}]$')
    plt.ylabel('$k_y [mm^{-1}]$')
    plt.title('k-Space Trajectory')

    seq.calculate_gradient_spectrum(acoustic_resonances=[{'frequency': 700, 'bandwidth': 100}, {'frequency': 1164, 'bandwidth': 250}])
    plt.title('Gradient spectrum')
    plt.show()

 
# Detailed report if requested
if params['user_settings']['detailed_rep']:
    print("\n===== Detailed Test Report =====\n")
    rep_str = seq.test_report()
    print(rep_str)

# Write the sequence to file
if params['user_settings']['write_seq']:

    seq.set_definition(key="FOV", value=[fov[0]*1e-2, fov[0]*1e-2, params['acquisition']['slice_thickness']*1e-3])
    seq.set_definition(key="Slice_Thickness", value=params['acquisition']['slice_thickness']*1e-3)
    seq.set_definition(key="Name", value="sprssfp")
    seq.set_definition(key="TE", value=TE)
    seq.set_definition(key="TR", value=TR)
    seq.set_definition(key="FA", value=params['acquisition']['flip_angle'])
    seq.set_definition(key="Resolution_mm", value=res)
    seq_filename = f"spiral_{params['spiral']['contrast']}{FA_schedule_str}{prep_str}{end_prep_str}_{params['spiral']['arm_ordering']}{params['spiral']['GA_angle']:.4f}_nTR{n_TRs}_Tread{params['spiral']['ro_duration']*1e3:.2f}_TR{TR*1e3:.2f}ms_FA{params['acquisition']['flip_angle']}_{params['user_settings']['filename_ext']}"

    # remove double, triple, quadruple underscores, and trailing underscores
    seq_filename = seq_filename.replace("__", "_").replace("__", "_").replace("__", "_").strip("_")

    seq_path = os.path.join('out_seq', f"{seq_filename}.seq")

    # ensure the out_seq directory exists before writing.
    os.makedirs("out_seq", exist_ok=True)

    seq.write(seq_path)  # Save to disk

    # Export k-space trajectory
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    # save_traj_dcf(seq.signature_value, k_traj_adc, n_TRs, n_int, fov, res, ndiscard, params['user_settings']['show_plots'])
    params_save = {
        'adc_dwell': spiral_sys['adc_dwell'],
        'ndiscard': ndiscard,
        'n_TRs': n_TRs,
        'n_int': n_int,
        'ga_rotation': params['spiral']['GA_angle'],
        'fov': fov,
        'spatial_resolution': res,
        'arm_ordering': params['spiral']['arm_ordering'],
    }
    save_metadata(seq.signature_value, k_traj_adc, params_save, params['user_settings']['show_plots'], dcf_method="hoge", out_dir="out_trajectory")

    print(f'Metadata file for {seq_filename} is saved as {seq.signature_value} in out_trajectory/.')