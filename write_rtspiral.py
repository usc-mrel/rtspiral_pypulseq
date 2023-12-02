import numpy as np
import matplotlib.pyplot as plt
from pypulseq import Opts
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.Sequence.sequence import Sequence
from pypulseq.rotate import rotate
from utils.load_params import load_params
from libvds.vds import vds_fixed_ro, plotgradinfo, raster_to_grad
from libvds_rewind.design_rewinder_exact_time import design_rewinder_exact_time
from libvds_rewind.pts_to_waveform import pts_to_waveform
import copy
from scipy.io import savemat
from sigpy.mri.dcf import pipe_menon_dcf

# Load and prep system and sequence parameters
params = load_params('config', './')

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
    'max_slew'          :  params['system']['max_slew'],   # [T/m/s] 
    'max_grad'          :  params['system']['max_grad'],   # [mT/m] 
    'adc_dwell'         :  params['spiral']['adc_dwell'],  # [s]
    'grad_raster_time'  :  GRT, # [s]
    'os'                :  8
    }

fov   = [params['acquisition']['fov']] # [cm]
res   =  params['acquisition']['resolution'] # [mm]
Tread =  params['spiral']['ro_duration'] # [s]


# Design the spiral trajectory
k, g, t, n_int = vds_fixed_ro(spiral_sys, fov, res, Tread)

print(f'Number of interleaves for fully sampled trajectory: {n_int}.')

t_grad, g_grad = raster_to_grad(g, spiral_sys['adc_dwell'], GRT)
# plotgradinfo(g, spiral_sys['adc_dwell'])
# plotgradinfo(g_grad, GRT)
# plt.show()
# === design rewinder ===
# calculate moment of each gradient for rewinder M0-nulled.
M = np.cumsum(g_grad, axis=0) * GRT
[times_x, amplitudes_x] = design_rewinder_exact_time(g_grad[-1, 0], 0, 1e-3, -M[-1,0], spiral_sys)
[times_y, amplitudes_y] = design_rewinder_exact_time(g_grad[-1, 1], 0, 1e-3, -M[-1,1], spiral_sys)

g_rewind_x = pts_to_waveform(times_x, amplitudes_x, GRT)
g_rewind_y = pts_to_waveform(times_y, amplitudes_y, GRT)

# # add zeros to the end of g_rewind_x or g_rewind_y to make them the same length (in case they are not).
if len(g_rewind_x) > len(g_rewind_y):
    g_rewind_y = np.concatenate((g_rewind_y, np.zeros(len(g_rewind_x) - len(g_rewind_y))))
else:
    g_rewind_x = np.concatenate((g_rewind_x, np.zeros(len(g_rewind_y) - len(g_rewind_x))))


# concatenate g and g_rewind, and plot.
g_grad = np.concatenate((g_grad, np.stack([g_rewind_x[0:], g_rewind_y[0:]]).T))
plotgradinfo(g_grad, GRT)
# gro_raster = g[4::10,:] # TODO: Generalize to other adc_dwell?

plt.show()

# Excitation

rf, gz, gzr = make_sinc_pulse(flip_angle=params['acquisition']['flip_angle']/180*np.pi, 
                                duration=params['acquisition']['rf_duration'],
                                slice_thickness=params['acquisition']['slice_thickness']*1e-3, # [mm] -> [m]
                                time_bw_product=2,
                                return_gz=True,
                                use='excitation', system=system)

gzrr = copy.deepcopy(gzr)
# gzrr.amplitude = -gzr.amplitude

# ADC
ndiscard = 10 # Number of samples to discard from beginning
num_samples = np.floor(Tread/spiral_sys['adc_dwell']) + ndiscard
adc = make_adc(num_samples, dwell=spiral_sys['adc_dwell'], delay=0, system=system)

discard_delay_t = ndiscard*spiral_sys['adc_dwell'] # [s] Time to delay grads.

# Readout gradients

gsp_x = make_arbitrary_grad(channel='x', waveform=g_grad[:,0]*42.58e3, delay=discard_delay_t, system=system) # [mT/m] -> [Hz/m]
gsp_x.first = 0
gsp_x.last = 0

gsp_y = make_arbitrary_grad(channel='y', waveform=g_grad[:,1]*42.58e3, delay=discard_delay_t, system=system) # [mT/m] -> [Hz/m]
gsp_y.first = 0
gsp_y.last = 0

# gsprew_x = make_extended_trapezoid(channel='x', amplitudes=amplitudes_x, times=times_x, system=system)
# gsprew_y = make_extended_trapezoid(channel='y', amplitudes=amplitudes_y, times=times_y, system=system)

# Set the Slice rewinder balance gradients delay

gzrr.delay = calc_duration(gsp_x, gsp_y, adc) - calc_duration(gzrr)

# set the rotations.

gsp_xs = []
gsp_ys = []
for i in range(0, n_int):
    gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=2*np.pi*i/n_int)
    gsp_xs.append(gsp_x_rot)
    gsp_ys.append(gsp_y_rot)

# Sequence looping

seq = Sequence(system)


# if n_int is odd, double num TRs because of phase cycling requirements.
n_TRs = n_int if n_int % 2 == 0 else 2 * n_int

for arm_i in range(0,n_TRs):
    rf.phase_offset = np.pi*np.mod(arm_i, 2)
    adc.phase_offset = rf.phase_offset
    seq.add_block(rf, gz)
    seq.add_block(gzr)

    seq.add_block(gsp_xs[arm_i % n_int], gsp_ys[arm_i % n_int], adc, gzrr)


# plt.figure()

# Quick timing check
ok, error_report = seq.check_timing()

if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed. Error listing follows:")
    [print(e) for e in error_report]

# Plot the sequence
if params['user_settings']['show_plots']:
    seq.plot(show_blocks=True, grad_disp='mT/m', plot_now=True)
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    plt.figure()
    plt.plot(k_traj[0,:], k_traj[1, :])
    plt.plot(k_traj_adc[0,:], k_traj_adc[1,:], 'rx')
    # plt.plot(k_traj.T)
    plt.show()
# 
# Detailed report if requested
if params['user_settings']['detailed_rep']:
    print("\n===== Detailed Test Report =====\n")
    rep_str = seq.test_report()
    print(rep_str)

# Write the sequence to file
if params['user_settings']['write_seq']:
    seq.set_definition(key="FOV", value=[fov[0]*1e-2, fov[0]*1e-2])
    seq.set_definition(key="SliceThickness", value=params['acquisition']['slice_thickness']*1e-3)
    seq.set_definition(key="Name", value="sprssfp")
    # seq.set_definition(key="TE", value=TE)
    # seq.set_definition(key="TR", value=TR)
    seq.set_definition(key="FA", value=params['acquisition']['flip_angle'])

    seq_filename = f"spiral_bssfp_{params['user_settings']['filename_ext']}"
    import os
    seq_path = os.path.join('out_seq', f'{seq_filename}.seq')
    seq.write(seq_path)  # Save to disk

    # Export k-space trajectory
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    Nsample = int(k_traj_adc.shape[1]/n_TRs)
    kx = k_traj_adc[0,:]
    ky = k_traj_adc[1,:]
    k_max = np.max(np.abs(kx + 1j * ky))
    k = (kx / k_max) + (1j * ky / k_max)

    # calculate density compensation weights using Pipe and Menon's method
    Nsample = int(k_traj_adc.shape[1]/n_TRs)
    w = pipe_menon_dcf(k_traj_adc[0:2, :].T)
    w = w[Nsample+1:2*Nsample+1]
    w = w / (np.max(w));
    w[w > 0.4] = 0.4;
    w = w / np.max(w);   
    w[int(w.shape[0]*2/3):w.shape[0]] = 1
    plt.plot(w)
    plt.show()

    param = {
        'fov': fov[0],
        'spatial_resolution': params['acquisition']['resolution'],
        'repetitions': n_TRs,
    }

    traj_path = os.path.join('out_trajectory', f'{seq_filename}.mat')
    savemat(traj_path, {'kx': kx, 'ky': ky, 'w' : w, 'param': param})