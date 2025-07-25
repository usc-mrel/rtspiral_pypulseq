import numpy as np
import matplotlib.pyplot as plt
from pypulseq import (
    Opts, make_sinc_pulse, make_arbitrary_rf, make_arbitrary_grad, 
    make_trapezoid, make_delay, calc_duration, calc_rf_center, 
    make_adc, Sequence, rotate, add_gradients
)
from utils.schedule_FA import schedule_FA
from utils.load_params import load_params
from libspiral import vds_fixed_ro, plotgradinfo, raster_to_grad
from librewinder.design_rewinder import design_rewinder
from kernels.kernel_handle_preparations import kernel_handle_preparations, kernel_handle_end_preparations
from math import ceil
from sigpy.mri.rf import slr 
import copy

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
    'max_grad'          :  params['system']['max_grad']*0.95,   # [mT/m] 
    'adc_dwell'         :  params['acquisition']['spiral']['adc_dwell'],  # [s]
    'grad_raster_time'  :  GRT, # [s]
    'os'                :  8
    }

fov   = params['acquisition']['fov'] # [cm]
res   = params['acquisition']['resolution'] # [mm]
Tread = params['acquisition']['spiral']['ro_duration'] # [s]


# Design the spiral trajectory
k, g, t, n_int = vds_fixed_ro(spiral_sys, fov, res, Tread)

print(f'Number of interleaves for fully sampled trajectory: {n_int}.')

t_grad, g_grad = raster_to_grad(g, spiral_sys['adc_dwell'], GRT)

g_rewind_x, g_rewind_y = design_rewinder(g_grad, params['spiral']['rewinder_time'], system, \
                                         slew_ratio=params['spiral']['slew_ratio'], \
                                         grad_rew_method=params['spiral']['grad_rew_method'], \
                                         M1_nulling=params['spiral']['M1_nulling'])

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

gzrr = copy.deepcopy(gzr)
gzrr.delay = 0 #gz.delay
rf.delay = calc_duration(gzrr) + gz.rise_time

additional_delay = 0
if rf.delay < params['system']['rf_dead_time']:
    # unfortunately, we have to artificially increase the delay.
    additional_delay = params['system']['rf_dead_time'] - rf.delay

rf.delay = rf.delay + additional_delay
gz.delay = calc_duration(gzrr) + additional_delay
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
    n_TRs = n_int * params['acquisition']['repetitions']
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
    n_TRs = n_int * params['acquisition']['repetitions']

else:
    raise Exception("Unknown arm ordering") 


# set the kz encoding if it is a 3D acquisition.
acquisition_type = '3D' if 'kz_encoding' in params['acquisition'] else '2D'
gzs = []
dummy_trap = make_trapezoid(channel="z", area=0, system=system)
kz_encoding_str = ''
if acquisition_type == '3D':
    kz_fov = params['acquisition']['kz_encoding']['FOV'] * 1e-3
    nkz = params['acquisition']['kz_encoding']['repetitions']
    phase_areas = (np.arange(nkz) - (nkz / 2)) * (1 / kz_fov)

    # make the largest trapezoid, and use it's duration for all of them.
    dummy_trap = make_trapezoid(channel="z", area=phase_areas[0], system=system)

    kz_encoding_str = params['acquisition']['kz_encoding']['ordering'] 
    print(f"Kz encoding ordering is {kz_encoding_str}.")
    if kz_encoding_str == 'ping-pong':
        kz_idx = np.hstack((np.arange(nkz), np.flip(np.arange(nkz))))
        for i in range(0, kz_idx.shape[0]):
            gzs.append(make_trapezoid(channel='z', area=phase_areas[kz_idx[i]], duration=calc_duration(dummy_trap), system=system))
    elif kz_encoding_str == 'gaussian':
        kz_idx = np.asarray([1,3,5,6,7,8,8,9,9,9,10,10,11,12,13,15,\
                             16,14,12,11,10,10,9,9,9,8,8,7,6,5,4,2])
        # make it 0-indexed.
        kz_idx = kz_idx - 1
        for i in range(0, kz_idx.shape[0]):
            gzs.append(make_trapezoid(channel='z', area=phase_areas[kz_idx[i]],\
                                      duration=calc_duration(dummy_trap), system=system))

    # add the stack-type to the kz encoding string
    kz_encoding_str = kz_encoding_str + '_' +\
        params['acquisition']['kz_encoding']['rotation_type']

# Set the delays

# TE 
if params['acquisition']['TE'] == 0:
    TEd = 0
    TE = rf.shape_dur - calc_rf_center(rf)[0] + calc_duration(gzr) - gzr.delay + gsp_x.delay
    if acquisition_type == '3D':
        TE = TE + calc_duration(gzs[0])
    print(f'Min TE is set: {TE*1e3:.3f} ms.')
    params['acquisition']['TE'] = TE
else:
    TE = params['acquisition']['TE']*1e-3
    TEd = TE - (rf.shape_dur - calc_rf_center(rf)[0] + calc_duration(gzr) + gsp_x.delay)
    if acquisition_type == '3D': 
        TEd = TEd - calc_duration(gzs[0])
    assert TEd >= 0, "Required TE can not be achieved."

# TR
if params['acquisition']['TR'] == 0:
    TRd = 0
    TR = calc_duration(rf, gzz) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc)
    if params['acquisition']['spiral']['contrast'] in ('FLASH', 'FISP'):
        TR = TR + calc_duration(gz_crush)
    if acquisition_type == '3D':
        TR = TR + (calc_duration(gzs[0]) * 2)
    print(f'Min TR is set: {TR*1e3:.3f} ms.')
    params['acquisition']['TR'] = TR
else:
    TR = params['acquisition']['TR']*1e-3
    TRd = TR - (calc_duration(rf, gzz) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc))
    if params['acquisition']['spiral']['contrast'] in ('FLASH', 'FISP'):
        TRd = TRd - calc_duration(gz_crush)
    if acquisition_type == '3D':
        TRd = TRd - (calc_duration(gzs[0]) * 2)
    assert TRd >= 0, "Required TR can not be achieved."

TE_delay = make_delay(TEd)
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

# decide to loop kz on the 'outside' or 'inside.
loop_type = params['acquisition']['kz_encoding']['rotation_type']

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

    # if we are doing 3D, then add the blips.
    if acquisition_type == '3D':
        seq.add_block(gzs[arm_i % len(gzs)])

    seq.add_block(TE_delay)

    in_plane_rot = params['acquisition']['spiral']['GA_steps'] if \
         params['acquisition']['spiral']['arm_ordering'] == 'ga' \
         else n_int

    arm_idx = np.floor(arm_i / params['acquisition']['kz_encoding']['repetitions']) \
        .astype(int) % in_plane_rot \
        if loop_type == 'stack' \
        else arm_i % in_plane_rot
    
    # LABEL Extensions
    if acquisition_type == '3D':
        seq.add_block(make_label('PAR','SET', int(kz_idx[arm_i % len(kz_idx)]))) # should it be SLC?
    seq.add_block(make_label('LIN', 'SET', int(arm_idx)))

    seq.add_block(gsp_xs[arm_idx], gsp_ys[arm_idx], adc)

    # if we are doing 3D, add 3D rewinder.
    if acquisition_type == '3D':
        gz_rewind = gzs[arm_i % len(gzs)]
        gz_rewind.amplitude = -gz_rewind.amplitude
        seq.add_block(gz_rewind)
        gz_rewind.amplitude = -gz_rewind.amplitude

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
    seq.plot(show_blocks=False, grad_disp='mT/m', plot_now=False, time_disp='ms')
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
    from utils.traj_utils import save_traj_dcf, save_traj_analyticaldcf

    seq.set_definition(key="FOV", value=[fov[0]*1e-2, fov[0]*1e-2, params['acquisition']['slice_thickness']*1e-3])
    #if acquisition_type == '2D':
    seq.set_definition(key="Slice_Thickness", value=params['acquisition']['slice_thickness']*1e-3)
    # else:
    #    seq.set_definition(key="Slice_Thickness", value=params['acquisition']['kz_encoding']['FOV']/params['acquisition']['kz_encoding']['repetitions']*1e-3)
    seq.set_definition(key="Name", value="sprssfp")
    seq.set_definition(key="TE", value=TE)
    seq.set_definition(key="TR", value=TR)
    seq.set_definition(key="FA", value=params['acquisition']['flip_angle'])
    seq.set_definition(key="Resolution_mm", value=res)
    seq_filename = f"spiral_{acquisition_type}_{kz_encoding_str}_{params['acquisition']['spiral']['contrast']}{FA_schedule_str}{prep_str}{end_prep_str}_{params['acquisition']['spiral']['arm_ordering']}{params['acquisition']['spiral']['GA_angle']}_nTR{n_TRs}_Tread{params['acquisition']['spiral']['ro_duration']*1e3:.2f}_TR{TR*1e3:.2f}ms_FA{params['acquisition']['flip_angle']}_{params['user_settings']['filename_ext']}"

    # remove double, triple, quadruple underscores, and trailing underscores
    seq_filename = seq_filename.replace("__", "_").replace("__", "_").replace("__", "_").strip("_")

    seq_path = os.path.join('out_seq', f"{seq_filename}.seq")
    seq.write(seq_path)  # Save to disk

    # Export k-space trajectory
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    # save_traj_dcf(seq.signature_value, k_traj_adc, n_TRs, n_int, fov, res, ndiscard, params['user_settings']['show_plots'])
    save_traj_analyticaldcf(seq.signature_value, k_traj_adc, n_TRs, n_int, fov, res, spiral_sys['adc_dwell'], ndiscard, params['user_settings']['show_plots'])

    print(f'Metadata file for {seq_filename} is saved as {seq.signature_value} in out_trajectory/.')
    
