import numpy as np
import matplotlib.pyplot as plt
from pypulseq import (
    make_sinc_pulse, make_arbitrary_rf, make_arbitrary_grad, 
    make_trapezoid, make_delay, make_label, make_adc,
    calc_duration, calc_rf_center, 
    Opts, Sequence, rotate, add_gradients
)
from utils.schedule_FA import schedule_FA
from utils.load_params import load_params
from libspiral import vds_fixed_ro, plotgradinfo, raster_to_grad
from libspiralutils import design_rewinder_exact_time, pts_to_waveform

from kernels.kernel_handle_preparations import kernel_handle_preparations, kernel_handle_end_preparations
from math import ceil
from sigpy.mri.rf import slr 
import copy
from utils.traj_utils import generate_encoding_indices

# Load and prep system and sequence parameters
params = load_params('config_3d', './')

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
    'max_slew'          :  params['system']['max_slew']*params['acquisition']['spiral']['slew_ratio'],   # [T/m/s] 
    'max_grad'          :  params['system']['max_grad']*0.99,   # [mT/m] 
    'adc_dwell'         :  params['acquisition']['spiral']['adc_dwell'],  # [s]
    'grad_raster_time'  :  GRT, # [s]
    'os'                :  8
    }

fov   = params['acquisition']['fov'] # [cm]
res   = params['acquisition']['resolution'] # [mm]
Tread = params['acquisition']['spiral']['ro_duration'] # [s]
n_kz = params['acquisition']['kz_encoding']['repetitions']
n_dummy = params['acquisition']['n_dummy']

if type(params['acquisition']['TE']) is list:
    TEs = params['acquisition']['TE']
    n_eco = len(params['acquisition']['TE'])
else:
    TEs = [params['acquisition']['TE']]
    n_eco = 1

# Design the spiral trajectory
k, g, t, n_int = vds_fixed_ro(spiral_sys, fov, res, Tread)
n_int = int(n_int * params['acquisition']['spiral']['inplane_os'])
print(f'Number of interleaves for fully sampled trajectory: {n_int}.')

t_grad, g_grad = raster_to_grad(g, spiral_sys['adc_dwell'], GRT)


# === design rewinder ===
T_rew = 1.2e-3
M = np.cumsum(g_grad, axis=0) * GRT

# TODO: add rewinders to metadata.
grad_rew_method = 1
# Design rew with gropt
if grad_rew_method == 1:
    from gropt.helper_utils import get_min_TE_gfix

    # Method 1: GrOpt, separate optimization
    gropt_params = {}
    gropt_params['mode'] = 'free'
    gropt_params['gmax'] = params['system']['max_grad']*1e-3 # [mT/m] -> [T/m]
    gropt_params['smax'] = params['system']['max_slew']*params['acquisition']['spiral']['slew_ratio']
    gropt_params['dt']   = GRT

    gropt_params['moment_params']  = [[0, 0, 0, -1, -1, -M[-1,0]*1e3, 1.0e-3]]
    gropt_params['gfix']  = np.array([g_grad[-1, 0]*1e-3, -99999, 0])

    g_rewind_x, T = get_min_TE_gfix(gropt_params, T_rew*1e3, True)
    g_rewind_x = g_rewind_x.T[:,0]*1e3

    gropt_params['moment_params']  = [[0, 0, 0, -1, -1, -M[-1,1]*1e3, 1.0e-3]]
    gropt_params['gfix']  = np.array([g_grad[-1, 1]*1e-3, -99999, 0])

    g_rewind_y, T = get_min_TE_gfix(gropt_params, T_rew*1e3, True)
    g_rewind_y = g_rewind_y.T[:,0]*1e3

elif grad_rew_method == 2:
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
if params['acquisition']['excitation'] == 'sinc':
    rf, gz, gzr = make_sinc_pulse(flip_angle=params['acquisition']['flip_angle']/180*np.pi, 
                                duration=params['acquisition']['rf_duration'],
                                slice_thickness=params['acquisition']['slice_thickness']*1e-3, # [mm] -> [m]
                                time_bw_product=6,
                                return_gz=True,
                                use='excitation', system=system)
else: #params['acquisition']['excitation'] == 'slr':
    alpha = params['acquisition']['flip_angle']
    dt = system.rf_raster_time
    raster_ratio = int(system.grad_raster_time / system.rf_raster_time)
    Trf = params['acquisition']['rf_duration']
    tbwp = params['acquisition']['tbwp']

    n = ceil((Trf/dt)/(4*raster_ratio))*4*raster_ratio
    Trf = n*dt
    bw = tbwp/Trf
    signal = slr.dzrf(n=n, tb=tbwp, ptype='st', ftype='ls', d1=0.01, d2=0.01, cancel_alpha_phs=False)

    rf, gz = make_arbitrary_rf(signal=signal, slice_thickness=params['acquisition']['slice_thickness']*1e-3,
                               bandwidth=bw,  flip_angle=alpha * np.pi / 180,
                               system=system, return_gz=True, use="excitation")
    gzr = make_trapezoid(channel='z', area=-gz.area/2, system=system)

# gzrr = copy.deepcopy(gzr)
# gzrr.delay = 0 #gz.delay
# rf.delay = calc_duration(gzrr) + gz.rise_time

additional_delay = 0
if rf.delay < params['system']['rf_dead_time']:
    # unfortunately, we have to artificially increase the delay.
    additional_delay = params['system']['rf_dead_time'] - rf.delay

# rf.delay = rf.delay + additional_delay
# gz.delay = gz.delay
# gz.delay = calc_duration(gzrr) + additional_delay
# gzr.delay = calc_duration(gzrr, gz)
# gzz = add_gradients([gzrr, gz], system=system)
gzz = gz

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
# gzrr.delay = calc_duration(gsp_x, gsp_y, adc)

# create a crusher gradient (only for FLASH)
# TODO: for 3D area should depend on partition index. That is why it can be advantageous to put it Gx and Gy.
gz_crush = 0
crush_area = (4 / (params['acquisition']['slice_thickness'] * 1e-3)) + (-1 * gzr.area)
gz_crush = make_trapezoid(channel='z', 
                          area=crush_area, 
                          max_grad=system.max_grad, 
                          system=system)

# set the rotations.
gsp_xs = []
gsp_ys = []
print(f"Spiral arm ordering is {params['acquisition']['spiral']['arm_ordering']}.")
if params['acquisition']['spiral']['arm_ordering'] == 'linear':
    for i in range(0, n_int):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=2*np.pi*i/n_int)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        params['acquisition']['spiral']['GA_angle'] = 360/n_int
    n_TRs = n_int * n_kz * params['acquisition']['repetitions']
elif params['acquisition']['spiral']['arm_ordering'] == 'ga':
    n_TRs = params['acquisition']['spiral']['GA_steps']
    n_int = n_TRs
    ang = 0
    for i in range(0, n_TRs):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=ang)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        ang += params['acquisition']['spiral']['GA_angle']*np.pi/180
        ang = ang % (2*np.pi)
        # print(f"Deg: {ang*180/np.pi}")
    n_TRs = n_int * params['acquisition']['repetitions'] * n_kz

else:
    raise Exception("Unknown arm ordering") 

# Generate encoding indices
kz_ordering = params['acquisition']['kz_encoding']['kz_ordering']
kspace_ordering = params['acquisition']['kz_encoding']['kspace_ordering']
idx = generate_encoding_indices(n_int, n_kz, n_eco=n_eco, n_rep=params['acquisition']['repetitions'], kz_ordering=kz_ordering, kspace_ordering=kspace_ordering)

# TODO: Combine slice rewinder and partition encoding to reduce min TE.
# set the kz encoding if it is a 3D acquisition.
acquisition_type = '3D' if 'kz_encoding' in params['acquisition'] else '2D'
gzs = []
gz_rewind = []

dummy_trap = make_trapezoid(channel="z", area=0, system=system)
kz_encoding_str = ''

kz_fov = params['acquisition']['kz_encoding']['FOV'] * 1e-3
nkz = params['acquisition']['kz_encoding']['repetitions']
phase_areas = (np.arange(nkz) - (nkz / 2)) * (1 / kz_fov) + gzr.area
max_phase_area_idx = np.argmax(np.abs(phase_areas))

phase_rewind_areas = -(np.arange(nkz) - (nkz / 2)) * (1 / kz_fov) + gzr.area
max_phase_rewind_area_idx = np.argmax(np.abs(phase_rewind_areas))

# make the largest trapezoid, and use it's duration for all of them.
dummy_trap = make_trapezoid(channel="z", area=phase_areas[max_phase_area_idx], system=system)
dummy_trap_rewind = make_trapezoid(channel="z", area=phase_rewind_areas[max_phase_rewind_area_idx], system=system)

print(f"Kz encoding ordering is {kz_ordering}.")

for i in range(0, nkz):
    gzs.append(make_trapezoid(channel='z', area=phase_areas[i], duration=calc_duration(dummy_trap), system=system))
    gz_rewind.append(make_trapezoid(channel='z', area=phase_rewind_areas[i], duration=calc_duration(dummy_trap_rewind), system=system))

# add the stack-type to the kz encoding string
kz_encoding_str = kz_encoding_str + '_' + kspace_ordering

# Set the delays
TEd = []
TE = [te_*1e-3 for te_ in TEs]

if TE[0] == 0:
    TE[0] = (rf.shape_dur - calc_rf_center(rf)[0] 
                    + calc_duration(gzs[0]) 
                    + gsp_x.delay)

TEd.append(TE[0] - (rf.shape_dur - calc_rf_center(rf)[0] 
                    + calc_duration(gzs[0]) 
                    + gsp_x.delay))

assert TEd[0] >= 0, "Required TE can not be achieved."

for eco_i in range(1, n_eco):
    TEd.append(TE[eco_i] - (TE[eco_i-1] + calc_duration(gsp_xs[0], gsp_ys[0], adc) + gsp_x.delay))

    assert TEd[eco_i] >= 0, f"Required TE for eco {eco_i} can not be achieved."

# TR
if params['acquisition']['TR'] == 0:
    TRd = 0
    TR = calc_rf_center(rf)[0] + TE[-1] + calc_duration(gsp_xs[0], gsp_ys[0], adc)
    if params['acquisition']['spiral']['contrast'] in ('FLASH', 'FISP'):
        TR = TR + calc_duration(gz_crush)
    if acquisition_type == '3D':
        TR = TR + (calc_duration(gzs[0]) * 2)
    print(f'Min TR is set: {TR*1e3:.3f} ms.')
    params['acquisition']['TR'] = TR
else:
    TR = params['acquisition']['TR']*1e-3
    TRd = TR - (calc_rf_center(rf)[0] + TE[-1]  + calc_duration(gsp_xs[0], gsp_ys[0], adc))
    if params['acquisition']['spiral']['contrast'] in ('FLASH', 'FISP'):
        TRd = TRd - calc_duration(gz_crush)
    if acquisition_type == '3D':
        TRd = TRd - (calc_duration(gzs[0]) * 2)
    assert TRd >= 0, "Required TR can not be achieved."

# TE_delay = make_delay(TEd)
TR_delay = make_delay(TRd)
# Sequence looping

seq = Sequence(system)

# handle any preparation pulses.
prep_str = kernel_handle_preparations(seq, params, system, rf=rf, gz=gzz)

# useful for end_peparation pulses.
params['flip_angle_last'] = params['acquisition']['flip_angle']

# tagging pulse pre-prep (only if fa_schedule exists)
rf_amplitudes, FA_schedule_str = schedule_FA(params, n_TRs)

# used for FLASH only: set rf spoiling increment.
rf_phase = 0
rf_inc = 0

if params['acquisition']['spiral']['contrast'] == 'FLASH':
    linear_phase_increment = 0
    quadratic_phase_increment = np.deg2rad(117)
elif params['acquisition']['spiral']['contrast'] in ('trueFISP', 'FISP'):
    linear_phase_increment = np.deg2rad(180)
    quadratic_phase_increment = 0
else:
    print("Unknown contrast type. Assuming trueFISP.")
    linear_phase_increment = np.deg2rad(180)
    quadratic_phase_increment = 0
    params['acquisition']['spiral']['contrast'] = 'trueFISP'

_, rf.shape_IDs = seq.register_rf_event(rf)

# dummy scans
for dumm_i in range(0,n_dummy):
    curr_rf = copy.deepcopy(rf)

    curr_rf.phase_offset = rf_phase
    adc.phase_offset = rf_phase

    rf_inc = np.mod(rf_inc + quadratic_phase_increment, np.pi * 2)
    rf_phase = np.mod(rf_phase + linear_phase_increment + rf_inc, np.pi * 2)

    seq.add_block(curr_rf, gzz)

    # if we are doing 3D, then add the blips.
    seq.add_block(gzs[0])

    for eco_i in range(0, n_eco):
        seq.add_block(make_delay(TEd[eco_i]))
        seq.add_block(gsp_xs[0], gsp_ys[0])

    # if we are doing 3D, add 3D rewinder.
    seq.add_block(gz_rewind[0])

    # additional crushing if necessary.
    if params['acquisition']['spiral']['contrast'] in ('FLASH', 'FISP'):
        seq.add_block(gz_crush)

    # find the TR to check the timing
    seq.add_block(TR_delay)
 
# Actual event loop
for arm_i in range(0,n_TRs):
    curr_rf = copy.deepcopy(rf)

    # check if we are using a rammped FA scheme (rf_amplitudes is a list []) 
    if len(rf_amplitudes) > 0:
        if arm_i >= len(rf_amplitudes):
            n_TRs = arm_i
            break
        curr_rf.signal = rf.signal * rf_amplitudes[arm_i] / np.deg2rad(params['acquisition']['flip_angle'])
    
    curr_rf.phase_offset = rf_phase
    adc.phase_offset = rf_phase

    rf_inc = np.mod(rf_inc + quadratic_phase_increment, np.pi * 2)
    rf_phase = np.mod(rf_phase + linear_phase_increment + rf_inc, np.pi * 2)

    seq.add_block(curr_rf, gzz)

    # Add labels
    seq.add_block(make_label(label='LIN', type='SET', value=idx['kspace_step_1'][::n_eco][arm_i]), 
                  make_label(label='PAR', type='SET', value=idx['kspace_step_2'][::n_eco][arm_i]))
    
    seq.add_block(gzs[idx['kspace_step_2'][::n_eco][arm_i]])

    for eco_i in range(0, n_eco):
        seq.add_block(make_label(label='ECO', type='SET', value=eco_i))
        seq.add_block(make_delay(TEd[eco_i]))
        seq.add_block(gsp_xs[idx['kspace_step_1'][::n_eco][arm_i]], gsp_ys[idx['kspace_step_1'][::n_eco][arm_i]], adc)

    seq.add_block(gz_rewind[idx['kspace_step_2'][::n_eco][arm_i]])

    # additional crushing if necessary.
    if params['acquisition']['spiral']['contrast'] in ('FLASH', 'FISP'):
        seq.add_block(gz_crush)

    # find the TR to check the timing
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
    from matplotlib.widgets import Cursor
    seq.plot(show_blocks=False, grad_disp='mT/m', plot_now=False, time_disp='ms')
    ax = plt.gca()
    cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    if acquisition_type=='3D': 
        plt.figure()
        plt.axes(projection='3d')
    else:
        plt.figure()

    plt.plot(k_traj_adc[0,:], k_traj_adc[1, :], k_traj_adc[2, :])

    # make axis suqaure
    plt.gca().set_aspect('equal', adjustable='box')
    # double fontsize
    plt.rcParams.update({'font.size': 14})

    #plt.plot(k_traj_adc[0,:], k_traj_adc[1,:], 'rx')
    plt.xlabel('$k_x [mm^{-1}]$')
    plt.ylabel('$k_y [mm^{-1}]$')
    plt.title('k-Space Trajectory')
    plt.show()
 
# Detailed report if requested
if params['user_settings']['detailed_rep']:
    print("\n===== Detailed Test Report =====\n")
    rep_str = seq.test_report()
    print(rep_str)

# Write the sequence to file
if params['user_settings']['write_seq']:

    import os
    from utils.traj_utils import save_3Dtraj, save_3Dtraj_2Donly

    seq.set_definition(key="FOV", value=[fov[0]*1e-2, fov[0]*1e-2, res*1e-3])
    seq.set_definition(key="SliceThickness", value=res*n_kz*1e-3)
    seq.set_definition(key="Name", value="sprssfp")
    seq.set_definition(key="TE", value=TE)
    seq.set_definition(key="TR", value=TR)
    seq.set_definition(key="FA", value=params['acquisition']['flip_angle'])
    seq.set_definition(key="Resolution_mm", value=res)
    seq.set_definition(key="kSpaceCenterSample", value=0)
    seq.set_definition(key="kSpaceCenterLine", value=0)
    seq.set_definition(key="kSpaceCenterPartition", value=n_eco*n_int*(n_kz//2))

    seq_filename = f"spiral_{acquisition_type}_{kz_encoding_str}_{params['acquisition']['spiral']['contrast']}{FA_schedule_str}{prep_str}{end_prep_str}_{params['acquisition']['spiral']['arm_ordering']}{params['acquisition']['spiral']['GA_angle']}_nTR{n_TRs}_neco{n_eco}_Tread{params['acquisition']['spiral']['ro_duration']*1e3:.2f}_TR{TR*1e3:.2f}ms_FA{params['acquisition']['flip_angle']}_{params['user_settings']['filename_ext']}"

    # remove double, triple, quadruple underscores, and trailing underscores
    seq_filename = seq_filename.replace("__", "_").replace("__", "_").replace("__", "_").strip("_")

    seq_path = os.path.join('out_seq', f"{seq_filename}.seq")
    seq.write(seq_path)  # Save to disk

    # Export k-space trajectory
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    # save_traj_dcf(seq.signature_value, k_traj_adc, n_TRs, n_int, fov, res, ndiscard, params['user_settings']['show_plots'])

#     save_3Dtraj(seq.signature_value, k_traj_adc, 
                # n_TRs, n_eco, n_int, params['acquisition']['spiral']['GA_angle'], 
                # idx, fov, res, spiral_sys['adc_dwell'], ndiscard, params['user_settings']['show_plots'])
    
    save_3Dtraj_2Donly(seq.signature_value, k_traj_adc, 
                n_TRs, n_eco, n_int, params['acquisition']['spiral']['GA_angle'], 
                idx, fov, res, spiral_sys['adc_dwell'], ndiscard, params['user_settings']['show_plots'])


    print(f'Metadata file for {seq_filename} is saved as {seq.signature_value} in out_trajectory/.')
    
