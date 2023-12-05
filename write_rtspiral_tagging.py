import numpy as np
import matplotlib.pyplot as plt
from pypulseq import Opts
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_delay import make_delay
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_adc import make_adc
from pypulseq.Sequence.sequence import Sequence
from pypulseq.rotate import rotate
from utils.load_params import load_params
from libvds.vds import vds_fixed_ro, plotgradinfo, raster_to_grad
from libvds_rewind.design_rewinder_exact_time import design_rewinder_exact_time
from libvds_rewind.pts_to_waveform import pts_to_waveform
from kernels.kernel_tag_CSPAMM import kernel_tag_CSPAMM
import copy

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

# === design rewinder ===
# calculate moment of each gradient for rewinder M0-nulled.
T_rew = 1e-3
M = np.cumsum(g_grad, axis=0) * GRT
[times_x, amplitudes_x] = design_rewinder_exact_time(g_grad[-1, 0], 0, T_rew, -M[-1,0], spiral_sys)
[times_y, amplitudes_y] = design_rewinder_exact_time(g_grad[-1, 1], 0, T_rew, -M[-1,1], spiral_sys)

g_rewind_x = pts_to_waveform(times_x, amplitudes_x, GRT)
g_rewind_y = pts_to_waveform(times_y, amplitudes_y, GRT)

# # add zeros to the end of g_rewind_x or g_rewind_y to make them the same length (in case they are not).
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

# Set the Slice rewinder balance gradients delay

gzrr.delay = calc_duration(gsp_x, gsp_y, adc) - calc_duration(gzrr)

# set the rotations.

gsp_xs = []
gsp_ys = []
for i in range(0, n_int):
    gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=2*np.pi*i/n_int)
    gsp_xs.append(gsp_x_rot)
    gsp_ys.append(gsp_y_rot)

# Set the delays

# TE 
if params['acquisition']['TE'] == 0:
    TEd = 0
    TE = calc_duration(rf, gz) - calc_rf_center(rf)[0] - gz.fall_time + calc_duration(gzr) + gsp_x.delay
    print(f'Min TE is set: {TE*1e3:.3f} ms.')
else:
    TEd = params['acquisition']['TE']*1e-3 - (calc_duration(rf, gz) - calc_rf_center(rf)[0] - gz.fall_time + calc_duration(gzr) + gsp_x.delay)
    assert TEd >= 0, "Required TE can not be achieved."

# TR
if params['acquisition']['TR'] == 0:
    TRd = 0
    TR = calc_duration(rf, gz) + calc_duration(gzr) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc, gzrr)
    print(f'Min TR is set: {TR*1e3:.3f} ms.')
else:
    TRd = params['acquisition']['TR']*1e-3 - (calc_duration(rf, gz) + calc_duration(gzr) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc, gzrr))
    assert TRd >= 0, "Required TE can not be achieved."

TE_delay = make_delay(TEd)
TR_delay = make_delay(TRd)
# Sequence looping

seq = Sequence(system)

# add tagging pulses:
kernel_tag_CSPAMM(seq, system)

# if n_int is odd, double num TRs because of phase cycling requirements.
n_TRs = n_int if n_int % 2 == 0 else 2 * n_int

n_TRs = n_TRs * 10 # add more TRs for T1 tagging recovery.

for arm_i in range(0,n_TRs):
    rf.phase_offset = np.pi*np.mod(arm_i, 2)
    adc.phase_offset = rf.phase_offset
    seq.add_block(rf, gz)
    seq.add_block(gzr)
    seq.add_block(TE_delay)
    seq.add_block(gsp_xs[arm_i % n_int], gsp_ys[arm_i % n_int], adc, gzrr)
    seq.add_block(TR_delay)

# Quick timing check
ok, error_report = seq.check_timing()

if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed. Error listing follows:")
    [print(e) for e in error_report]

# Plot the sequence
if params['user_settings']['show_plots']:
    seq.plot(show_blocks=True, grad_disp='mT/m', plot_now=False)
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    plt.figure()
    plt.plot(k_traj[0,:], k_traj[1, :])
    # plt.plot(k_traj_adc[0,:], k_traj_adc[1,:], 'rx')
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
    from utils.traj_utils import save_traj_dcf

    seq.set_definition(key="FOV", value=[fov[0]*1e-2, fov[0]*1e-2, params['acquisition']['slice_thickness']*1e-3])
    seq.set_definition(key="Slice_Thickness", value=params['acquisition']['slice_thickness']*1e-3)
    seq.set_definition(key="Name", value="sprssfp")
    seq.set_definition(key="TE", value=params['acquisition']['TE']*1e-3)
    seq.set_definition(key="TR", value=params['acquisition']['TR']*1e-3)
    seq.set_definition(key="FA", value=params['acquisition']['flip_angle'])
    seq.set_definition(key="Resolution_mm", value=res)

    seq_filename = f"spiral_bssfp_{params['user_settings']['filename_ext']}"
    seq_path = os.path.join('out_seq', f'{seq_filename}.seq')
    seq.write(seq_path)  # Save to disk

    # Export k-space trajectory
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    save_traj_dcf(seq.signature_value, k_traj_adc, n_TRs, n_int, fov, res, ndiscard, params['user_settings']['show_plots'])
    print(f'Metadata file for {seq_filename} is saved as {seq.signature_value} in out_trajectory/.')