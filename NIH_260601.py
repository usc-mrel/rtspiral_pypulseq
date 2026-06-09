#%%
import numpy as np
import matplotlib.pyplot as plt
from pypulseq import Opts
from pypulseq import (make_adc, make_sinc_pulse, make_digital_output_pulse, make_delay, 
					  make_arbitrary_grad, make_trapezoid, 
					  calc_duration, calc_rf_center, 
					  rotate, add_gradients, make_label)
from pypulseq.make_sigpy_pulse import sigpy_n_seq as make_sigpy_pulse
from pypulseq.Sequence.sequence import Sequence
from utils import schedule_FA, load_params
from utils.traj_utils import save_metadata_spi
from libspiral import vds_fixed_ro, plotgradinfo, raster_to_grad
from librewinder.design_rewinder import design_rewinder
from kernels.kernel_handle_preparations import kernel_handle_preparations, kernel_handle_end_preparations
from math import ceil
import copy
import argparse
import os
import warnings

from utils.vis_utils import plot_rf, spi_vis_check
from utils.spi_utils import create_disk_inplane_rot, rotate_disks, assign_fovs_and_res_based_on_rotation_axis
from utils.spi_utils import get_seq_name, apply_view_order

# ------------------------------------------------------------------------------------
# Cmd args
# ------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
					prog='WriteSPI',
					description='Generates a 3D Spiral Projection Imaging Pulseq sequence for given parameters.')

parser.add_argument('-c', '--config', type=str, default='config', help='Config file path.')
args = parser.parse_args()

# Load and prep system and sequence parameters
print(f'Using config file: {args.config}.')
params = load_params(args.config, './')

show_plots 		= params['user_settings']['show_plots']
flag_view_order = True if 'view_order' in params.keys() else False	# 26-06-06. Adding the parameters for the view order sweep

# ------------------------------------------------------------------------------------
# Set system limits
# ------------------------------------------------------------------------------------
system = Opts(
	max_grad = params['system']['max_grad'], grad_unit="mT/m",
	max_slew = params['system']['max_slew'], slew_unit="T/m/s",
	grad_raster_time = params['system']['grad_raster_time'],  # [s] ( 10 us)
	rf_raster_time   = params['system']['rf_raster_time'],    # [s] (  1 us)
	rf_ringdown_time = params['system']['rf_ringdown_time'],  # [s] ( 10 us)
	rf_dead_time     = params['system']['rf_dead_time'],      # [s] (100 us)
	adc_dead_time    = params['system']['adc_dead_time'],     # [s] ( 10 us)
)

# ------------------------------------------------------------------------------------
# Folders
# ------------------------------------------------------------------------------------
seq_pp_folder  = params['user_settings'].get('seq_pp_folder', 'seq_pp')
seq_meta_folder= params['user_settings'].get('seq_meta_folder', 'seq_meta')

# ------------------------------------------------------------------------------------
# Spiral design params with constant RO
# ------------------------------------------------------------------------------------
GRT = params['system']['grad_raster_time']
spiral_sys = {
	'max_slew'          :  params['system']['max_slew'] * params['spiral']['slew_ratio'],   # [T/m/s] 
	'max_grad'          :  params['system']['max_grad'] * params['spiral']['grad_ratio'],   # [mT/m] 
	'adc_dwell'         :  params['spiral']['adc_dwell'],  									# [s]
	'grad_raster_time'  :  GRT, 															# [s]
	'os'                :  8
	}

# ------------------------------------------------------------------------------------
# nr_planes & nr_interleaves
# Design base spiral
# ------------------------------------------------------------------------------------

# Dimensions etc
sfov   	= params['acquisition']['fov'][0] 				# [cm], Spiral FOV
rfov	= params['acquisition']['fov'][1] 				# [cm], Radial FOV
slab   	= params['acquisition']['slab_thickness']		# [mm], Slab thickness
res   	= params['acquisition']['resolution'] 			# [mm]
Tread 	= params['spiral']['ro_duration'] 				# [s]
disk_os = params['spiral'].get('disk_os', 1.0)			# Through-plane oversampling factor.

# ------------------------------------------------------------------------------------
# FOV List for metadata based on the rotation axis in [m]
# ------------------------------------------------------------------------------------
fovs, res_m = assign_fovs_and_res_based_on_rotation_axis(params, sfov, rfov, slab, res)

# ------------------------------------------------------------------------------------
# nr_planes
# ------------------------------------------------------------------------------------
nr_planes = int( (rfov*10 / res) * disk_os * np.pi/2)
if nr_planes % 2 == 1:
	nr_planes += 1  # make it even
	print("Number of planes increased by 1 to make it even.")

params['radial']['nr_planes'] = nr_planes

# ------------------------------------------------------------------------------------
# Design the spiral trajectory (nr_interleaves)
# ------------------------------------------------------------------------------------
k, g, t, nr_interleaves = vds_fixed_ro(spiral_sys, [sfov], res, Tread)
params['spiral']['nr_interleaves'] = nr_interleaves

print(f'Radial FOV: {rfov}, Number of planes for fully sampled trajectory: {nr_planes}.')
print(f'Spiral FOV: {sfov}, Number of interleaves for fully sampled trajectory: {nr_interleaves}.')

if g is None:
	raise RuntimeError("Failed to design spiral trajectory. Please check your parameters.")

# Raster gradients
t_grad, g_grad = raster_to_grad(g, spiral_sys['adc_dwell'], GRT)

# ------------------------------------------------------------------------------------
# Design the spiral rewinder
# ------------------------------------------------------------------------------------
if params['spiral']['rotate_grads']:
	g_rewind_x, g_rewind_y, g_grad = design_rewinder(g_grad, params['spiral']['rewinder_time'], system, # type: ignore
											 slew_ratio=params['spiral']['slew_ratio'],
											 grad_rew_method=params['spiral']['grad_rew_method'],
											 M1_nulling=params['spiral']['M1_nulling'], rotate_grads=params['spiral']['rotate_grads'])
else:
	g_rewind_x, g_rewind_y = design_rewinder(g_grad, params['spiral']['rewinder_time'], system, # type: ignore
											 slew_ratio=params['spiral']['slew_ratio'],
											 grad_rew_method=params['spiral']['grad_rew_method'],
											 M1_nulling=params['spiral']['M1_nulling'])

# concatenate g and g_rewind, and plot.
g_grad = np.concatenate((g_grad, np.stack([g_rewind_x[0:], g_rewind_y[0:]]).T))

if show_plots:
	plotgradinfo(g_grad, GRT)
	plt.show()

#%%
# ------------------------------------------------------------------------------------
# Excitation
# ------------------------------------------------------------------------------------
tbwp = params['acquisition']['tbwp']

if params['acquisition']['excitation'] == 'sinc':
	rf, gz, gzr = make_sinc_pulse(flip_angle=params['acquisition']['flip_angle']/180*np.pi, 
								duration=params['acquisition']['rf_duration'],
								slice_thickness=slab*1e-3, 		# [mm] -> [m]
								time_bw_product=tbwp,
								return_gz=True,
								use='excitation', system=system) # type: ignore
elif params['acquisition']['excitation'] == 'slr':
	rf, gz, gzr = make_sigpy_pulse(flip_angle=params['acquisition']['flip_angle']/180*np.pi, 
								duration=params['acquisition']['rf_duration'],
								slice_thickness=slab*1e-3, 		# [mm] -> [m]
								time_bw_product=tbwp,
								pulse_cfg=None,
								return_gz=True,
								use='excitation', system=system,
								plot=False)
# ------------------------------------------------------------------------------------
# Assign the channel for the RF
# ------------------------------------------------------------------------------------
rf_ch = params['Orientation']['rf_axis']
gz.channel  = rf_ch
gzr.channel = rf_ch

# TODO: Slice rewinder balancer can be played with Gz rewinder.
gzrr = copy.deepcopy(gzr)
gzrr.delay = 0 #gz.delay
rf.delay = calc_duration(gzrr) + gz.rise_time
gz.delay = calc_duration(gzrr)
gzr.delay = calc_duration(gzrr, gz)
gzz = add_gradients([gzrr, gz, gzr], system=system)

# Additional plotting, turn off
plot_rf(rf, gz, rf_raster_scale = 10, show_plots=show_plots)

# Add delays to RF and Gz to account for dead times
additional_delay = 0
if rf.delay < params['system']['rf_dead_time']:
	# unfortunately, we have to artificially increase the delay.
	additional_delay = params['system']['rf_dead_time'] - rf.delay

rf.delay = rf.delay + additional_delay
gz.delay = calc_duration(gzrr) + additional_delay
gzr.delay = calc_duration(gzrr, gz)
gzz = add_gradients([gzrr, gz, gzr], system=system)

# ------------------------------------------------------------------------------------
# ADC
# ------------------------------------------------------------------------------------
ndiscard 	= 10 				# Number of samples to discard from beginning
num_samples = np.floor(Tread/spiral_sys['adc_dwell']) + ndiscard
adc 		= make_adc(num_samples, dwell=spiral_sys['adc_dwell'], delay=0, system=system)

# NOTE: we shift by GRT/2 and round to GRT because the grads will be shifted by GRT/2, and if we don't, last GRT/2 ADC samples discarded will be non-zero k-space.
# Basically we will miss the center of k-space. Caveat this way is, now we have GRT/2 ADC samples that are at 0, and we potentially lost 10 us, both are no biggie.
discard_delay_t = ceil((ndiscard*spiral_sys['adc_dwell']+GRT/2)/GRT)*GRT # [s] Time to delay grads.

# ------------------------------------------------------------------------------------
# Readout gradients for the base spiral (no rotations)
# ------------------------------------------------------------------------------------
plane = params['Orientation']['pulseq_ort_map']

if plane == 'XYZ-TRA': # Base plane is on the axial
	rot_axis = "z"
	gsp_x = make_arbitrary_grad(channel='x', waveform=g_grad[:,0]*42.58e3, 	     first=0, last=0, delay=discard_delay_t, system=system) # [mT/m] -> [Hz/m]
	gsp_y = make_arbitrary_grad(channel='y', waveform=g_grad[:,1]*42.58e3, 	     first=0, last=0, delay=discard_delay_t, system=system) # [mT/m] -> [Hz/m]
elif plane == 'XYZ-SAG':
	raise Exception("XYZ-SAG orientation not implemented yet.")
elif plane == 'XYZ-COR':
	raise Exception("XYZ-COR orientation not implemented yet.")
else:
	raise Exception("Unknown orientation mapping.")

# Set the Slice rewinder balance gradients delay
# TODO: Do I need to pass on anything
gzrr.delay = calc_duration(gsp_x, gsp_y, adc)

# ------------------------------------------------------------------------------------
# (only for FLASH) create a crusher gradient 
# ------------------------------------------------------------------------------------
if params['acquisition']['contrast'] == 'FLASH' or params['acquisition']['contrast'] == 'FISP':
	crush_area = (4 / (params['acquisition']['slice_thickness'] * 1e-3)) + (-1 * gzr.area)
	gz_crush = make_trapezoid(channel=rf_ch, 
							  area=crush_area, 
							  max_grad=system.max_grad, 
							  system=system)

# ------------------------------------------------------------------------------------
# Define the base Plane (Plane0) gradients. set the rotations In-plane.
# ------------------------------------------------------------------------------------
gsp_xy = create_disk_inplane_rot(gsp_x, gsp_y, params, rot_axis=rot_axis)

# ------------------------------------------------------------------------------------
# set the rotations for the following planes. Generates a proj-mode fully-sampled SPI 
# ------------------------------------------------------------------------------------
gsp_xyz 	= rotate_disks(gsp_xy, params)
n_TRs_total = len(gsp_xyz)

# ------------------------------------------------------------------------------------
# Set the delays
# ------------------------------------------------------------------------------------
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
	TR = calc_duration(rf, gzz) + TEd + calc_duration(*gsp_xyz[0], adc)
	if params['acquisition']['contrast'] in ('FLASH', 'FISP'):
		TR = TR + calc_duration(gz_crush) # pyright: ignore[reportPossiblyUnboundVariable]
	print(f'Min TR is set: {TR*1e3:.3f} ms.')
	params['acquisition']['TR'] = TR
else:
	TR = params['acquisition']['TR']*1e-3
	TRd = TR - (calc_duration(rf, gzz) + TEd + calc_duration(*gsp_xyz[0], adc))
	if params['acquisition']['contrast'] in ('FLASH', 'FISP'):
		TRd = TRd - calc_duration(gz_crush) # pyright: ignore[reportPossiblyUnboundVariable]
	assert TRd >= 0, "Required TR can not be achieved."

TE_delay = make_delay(TEd)
TR_delay = make_delay(TRd)

# ------------------------------------------------------------------------------------
# 26-06-06
# [If view order] shuffle the TRs before the loop according to the view order and parameters
# ------------------------------------------------------------------------------------
if flag_view_order:
	n_TRs_total, gsp_xyz = apply_view_order(gsp_xyz, params)

# ------------------------------------------------------------------------------------
# Sequence looping
# ------------------------------------------------------------------------------------
seq = Sequence(system)

# handle any preparation pulses.
prep_str = kernel_handle_preparations(seq, params, system, rf=rf, gz=gzz)

# useful for end_peparation pulses.
params['flip_angle_last'] = params['acquisition']['flip_angle']

# tagging pulse pre-prep (only if fa_schedule exists)
rf_amplitudes, FA_schedule_str = schedule_FA(params, n_TRs_total)

# used for FLASH only: set rf spoiling increment.
rf_phase = 0
rf_inc = 0

if params['acquisition']['contrast'] == 'FLASH':
	linear_phase_increment = 0
	quadratic_phase_increment = np.deg2rad(117)
elif params['acquisition']['contrast'] in ('trueFISP', 'FISP'):
	linear_phase_increment = np.deg2rad(180)
	quadratic_phase_increment = 0
else:
	print("Unknown contrast type. Assuming trueFISP.")
	linear_phase_increment = np.deg2rad(180)
	quadratic_phase_increment = 0
	params['acquisition']['contrast'] = 'trueFISP'

enable_trigger = True	# NOTE: Dont know implication of True or False
trig = make_digital_output_pulse(channel='ext1', duration=0.001, system=system)

_, rf.shape_IDs = seq.register_rf_event(rf)

# ------------------------------------------------------------------------------------
# Dummy pulses
# TODO: Change in the future to alpha/2 TR/2 pulses
# ------------------------------------------------------------------------------------
for arm_i in range(0,params['acquisition']['n_dummy']):
	curr_rf = copy.deepcopy(rf)
	
	curr_rf.phase_offset = rf_phase
	adc.phase_offset 	 = rf_phase

	rf_inc 	 = np.mod(rf_inc + quadratic_phase_increment, np.pi * 2)
	rf_phase = np.mod(rf_phase + linear_phase_increment + rf_inc, np.pi * 2)

	seq.add_block(curr_rf, gzz)
		
	seq.add_block(TE_delay)
	
	# LIN -> Interleave, Arm. arm_i mod nr_interleaves
	# PAR -> Disk.			  arm_i // nr_interleaves 
	if params['spiral']['arm_ordering'] == 'ga':
		raise ValueError("Haven't extended to SPI.")
		seq.add_block(make_label('LIN', 'SET', arm_i % params['spiral']['GA_steps']))
		seq.add_block(*gsp_xyz[arm_i % params['spiral']['GA_steps']], adc) 
	else:  # Linear
		seq.add_block(make_label('ONCE', 'SET', 1))
		seq.add_block(*gsp_xyz[0])
	if params['acquisition']['contrast'] in ('FLASH', 'FISP'):
		seq.add_block(gz_crush) # pyright: ignore[reportPossiblyUnboundVariable]
	seq.add_block(TR_delay)
seq.add_block(make_label('ONCE', 'SET', 0))

# ------------------------------------------------------------------------------------
# MAIN: Loop over TRs
# ------------------------------------------------------------------------------------
for arm_i in range(0,n_TRs_total):
	curr_rf = copy.deepcopy(rf)

	# check if we are using a rammped FA scheme (rf_amplitudes is a list []) 
	if len(rf_amplitudes) > 0:
		if arm_i >= len(rf_amplitudes):
			n_TRs = arm_i
			break
		curr_rf.signal = rf.signal * rf_amplitudes[arm_i] / np.deg2rad(params['acquisition']['flip_angle'])
	
	curr_rf.phase_offset = rf_phase
	adc.phase_offset 	 = rf_phase

	rf_inc 	 = np.mod(rf_inc + quadratic_phase_increment, np.pi * 2)
	rf_phase = np.mod(rf_phase + linear_phase_increment + rf_inc, np.pi * 2)

	if enable_trigger is True:
		seq.add_block(trig, curr_rf, gzz)
	else:
		seq.add_block(curr_rf, gzz)
		
	seq.add_block(TE_delay)
	# LIN -> Interleave, Arm. arm_i mod nr_interleaves
	# PAR -> Disk.			  arm_i // nr_interleaves 
	# Adding TRs one-by-one.
	# NOTE: The labels are added for proj-mode fully-sampled and not exactly used by the others...
	seq.add_block(make_label('LIN', 'SET', arm_i % nr_interleaves))
	seq.add_block(make_label('PAR', 'SET', arm_i // nr_interleaves))
	seq.add_block(*gsp_xyz[arm_i], adc)
	
	if params['acquisition']['contrast'] in ('FLASH', 'FISP'):
		seq.add_block(gz_crush) # pyright: ignore[reportPossiblyUnboundVariable]
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

# ------------------------------------------------------------------------------------
# Plot the sequence
# ------------------------------------------------------------------------------------
# spi_vis_check(seq, params, nr_interleaves, nr_planes, show_plots = show_plots)

if 'acoustic_resonances' in params and 'frequencies' in params['acoustic_resonances']:
	resonances = []
	for idx in range(len(params['acoustic_resonances']['frequencies'])):
		resonances.append({'frequency': params['acoustic_resonances']['frequencies'][idx], 'bandwidth': params['acoustic_resonances']['bandwidths'][idx]})
	seq.calculate_gradient_spectrum(acoustic_resonances=resonances)
	plt.title('Gradient spectrum')
	plt.show()

# ------------------------------------------------------------------------------------
# Detailed report if requested
# ------------------------------------------------------------------------------------
if params['user_settings']['detailed_rep']:
	print("\n===== Detailed Test Report =====\n")
	rep_str = seq.test_report()
	print(rep_str)

# ------------------------------------------------------------------------------------
# Write the sequence to file
# ------------------------------------------------------------------------------------
if params['user_settings']['write_seq']:
	seq.set_definition(key="FOV", value=fovs)  # [m]
	seq.set_definition(key="Slab_Thickness", value=params['acquisition']['slab_thickness']*1e-3)
	seq.set_definition(key="Name", value="sprssfp")
	seq.set_definition(key="TE", value=TE)
	seq.set_definition(key="TR", value=TR)
	seq.set_definition(key="FA", value=params['acquisition']['flip_angle'])
	seq.set_definition(key="Resolution_mm", value=res)

	m1_str = "M1" if params['spiral']['M1_nulling'] else ""

	# --------------------------------------------------------------------------------
	# Write sequence and save metadata
	# --------------------------------------------------------------------------------
	seq_fname = get_seq_name(seq_pp_folder, params, FA_schedule_str = FA_schedule_str,
						  	prep_str=prep_str, end_prep_str=end_prep_str, m1_str=m1_str)		# 26-06-07. Get sequence name
	# ensure the out_seq directory exists before writing.
	os.makedirs(seq_pp_folder, exist_ok=True)
	seq.write(seq_fname)  # Save to disk

	# Export k-space trajectory
	k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

	# save_traj_dcf(seq.signature_value, k_traj_adc, n_TRs, n_int, fov, res, ndiscard, params['user_settings']['show_plots'])
	try:
		disk_ordering = params['spiral']['disk_ordering']
	except:
		disk_ordering = params['radial']['plane_ordering']
	params_save = {
		'adc_dwell': 		 spiral_sys['adc_dwell'],
		'ndiscard': 		 ndiscard,
		'n_TRs': 			 n_TRs_total,
		'nr_interleaves': 	 nr_interleaves,
		'nr_planes': 		 nr_planes,
		'fov': 				 fovs, 					# [m]
		'spatial_resolution':res_m,					# [m]
		'FA':  			 	 params['acquisition']['flip_angle'],
		'TR': 				 TR * 1e3,				# [ms]
		'TE': 				 TE * 1e3,				# [ms]
		'readout_duration':  params['spiral']['ro_duration'] * 1e3,	# [ms]
		'arm_ordering': 	 params['spiral']['arm_ordering'],
		'disk_ordering':	 disk_ordering,
		'rotation_angles':   params['spiral']['rotation_angles'] # [spiral in plane, hex theta, radial rotation]
	}
	save_metadata_spi(seq.signature_value, k_traj_adc, params_save, params['user_settings']['show_plots'], dcf_method="pipe_menon", out_dir=seq_meta_folder)

	print(f'Metadata file for {seq_fname} is saved as {seq.signature_value} in {seq_meta_folder}/.')
	

# %%