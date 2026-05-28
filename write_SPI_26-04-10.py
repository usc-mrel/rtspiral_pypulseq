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

'''
README

This script is for an initial exploration of the View Order. Specifically, only following case is considered.

- Aera specs
- Target spatial resolution : 3.5 mm3. 
- Linear in-plane rotation, GA spoke rotation.
- First N spiral arms in a disk are acquired, then disk is rotated. 
- N is calculated from the Undersampling factor given in the config file.

Use with config file 'config_SPI_vieworder.toml'.
'''

# ------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------
def plot_rf(rf, gz, rf_raster_scale = 10, show_plots=False):

	if not show_plots:
		return
	
	gamabar = 42.58  # MHz/T
	dt 		= 1e-6 * rf_raster_scale ; Fs = 1/dt  # [Hz]
	
	t_rf = rf.t[::rf_raster_scale] * 1e3  # [ms]
	rf_  = rf.signal[::rf_raster_scale] / gamabar
	rfe  = np.zeros(len(rf_) * rf_raster_scale)
	idx  = [(len(rfe) - len(rf_))//2, (len(rfe) + len(rf_))//2]
	rfe[idx[0] : idx[1]] = rf_ 
	
	N 	 = len(rfe)
	flag = True if N % 2 else False
	f 	 = np.linspace(-Fs/2, Fs/2, N, endpoint=flag)
	z 	 = f / gz.amplitude  # [m]

	if N % 2:
		RF = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rfe)))
	else:
		RF = np.fft.ifftshift(np.fft.fft(np.fft.ifftshift(rfe)))

	plt.figure()

	plt.subplot(2,1,1)
	plt.plot(rf.t * 1e3, rf.signal / gamabar, 'b')
	plt.plot(t_rf, rf_, '.r')
	plt.title(f'RF Pulse, peak B1: {np.max(np.abs(rf.signal))/gamabar:.2f} uT')
	plt.xlabel('Time [ms]')
	plt.ylabel('B1 Amplitude [uT]')
	plt.grid()

		#
	plt.subplot(2,1,2)
	plt.plot(z, np.abs(RF))
	plt.xlabel('[m]')
	plt.ylabel('m_xy')
	plt.grid()
	plt.title(f'FWHM {(np.sum(np.abs(RF) > np.max(np.abs(RF))/2) * (f[1]-f[0]) / gz.amplitude) * 1e2 :.1f} m')
	plt.show()
	return

def create_disk_inplane_rot(gsp_x, gsp_y, params, rot_axis="z"):
	'''
	Input: 
		gsp_x, gsp_y: base spiral gradients
		params: dictionary of parameters
	Output:
		gsp: [(gsp_x_0, gsp_y_0), (gsp_x_1, gsp_y_1), ...]
	'''
	
	gsp_xy = []

	print(f"Spiral arm ordering is {params['spiral']['arm_ordering']}.")
	print(f"Rotating around axis {rot_axis}")

	# ---------------------------------------------------------------------------------
	# Linear order
	if params['spiral']['arm_ordering'] == 'linear':

		nr_interleaves = params['spiral']['nr_interleaves']
		theta = 2*np.pi/nr_interleaves
		print(f"Rotation Angle in-plane: {np.rad2deg(theta):.2f} deg.")

		for arm in range(nr_interleaves):
			gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis=rot_axis, angle=theta*arm)
			gsp_xy.append((gsp_x_rot, gsp_y_rot))
	# ---------------------------------------------------------------------------------
	# Linear order with undersampling
	elif params['spiral']['arm_ordering'] == 'linear-skip':

		nr_interleaves = params['spiral']['nr_interleaves']
		sampling_disk   = params['spiral']['sampling_disk']

		if sampling_disk <= 0 or sampling_disk > 1:
			raise ValueError("sampling_disk should be in the range (0, 1].")

		nr_interleaves = ceil(nr_interleaves * sampling_disk) 
		theta = 2*np.pi/nr_interleaves
		print(f"Number of interleaves is changed to: {nr_interleaves} for {sampling_disk}x sampling of the full spiral.")
		print(f"Rotation Angle in-plane: {np.rad2deg(theta):.2f} deg.")

		for arm in range(nr_interleaves):
			gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis=rot_axis, angle=theta*arm)
			gsp_xy.append((gsp_x_rot, gsp_y_rot))
	# ---------------------------------------------------------------------------------
	# Golden Angle order
	elif params['spiral']['arm_ordering'] == 'ga':

		nr_interleaves = params['spiral']['GA_steps']
		theta = np.deg2rad(params['spiral']['GA_angle'])
		print(f"Rotation Angle in-plane: {params['spiral']['GA_angle']:.2f} deg.")

		ang = 0
		for arm in range(nr_interleaves):
			gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=ang)
			gsp_xy.append((gsp_x_rot, gsp_y_rot))

			ang += theta
			ang = ang % (2*np.pi)
	# ---------------------------------------------------------------------------------
	# Linear order with custom view order
	elif params['spiral']['arm_ordering'] == 'linear_custom':
		assert nr_interleaves == len(params['spiral']['custom_order']), "number of interleaves does not match custom order!"
		view_order = params['spiral']['custom_order']
		theta = 2*np.pi/nr_interleaves
		print(f"Rotation Angle in-plane: {np.rad2deg(theta):.2f} deg.")
		gsp_xs, gsp_ys = [], []
		for arm in range(nr_interleaves):
			gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=theta*arm)
			gsp_xs.append(gsp_x_rot)
			gsp_ys.append(gsp_y_rot)
			
			params['spiral']['GA_angle'] = 360/nr_interleaves
		# re-order using the custom view order
		gsp_xs[:] = [gsp_xs[d] for d in view_order]
		gsp_ys[:] = [gsp_ys[d] for d in view_order]
		for arm in range(nr_interleaves):
			gsp_xy.append((gsp_xs[arm], gsp_ys[arm]))
	else:
		raise Exception("Unknown arm ordering")

	# Record rotation angle (In-plane1)
	params['spiral']['rotation_angles'] = [np.rad2deg(theta), 0, 0] # In-plane1, In-plane2, Disk
	return gsp_xy

def rotate_disks(gsp_xy, params, base_rot_axis="z"):

	# User parameters
	nr_planes 		= params['spiral']['nr_planes']
	nr_interleaves 	= len(gsp_xy) #params['spiral']['nr_interleaves']
	disk_ordering 	= params['spiral']['disk_ordering']
	rot_axis 		= params['Orientation']['disk_rot_axis']
	# for hexagonal sampling in the (PE,SL) palne
	hex_sampling 	= params['spiral']['hex_sampling']
	hex_theta 		= 1 * np.pi / nr_interleaves if hex_sampling else 0

	# Output lists
	gsp_xyz = []

	print(f'Number of disks is set to {nr_planes} for Nyquist sampling.')
	print(f"Disk rotation axis is {rot_axis}.")

	if disk_ordering == 'linear':
		theta = 1 * np.pi / nr_planes # linear angle increment [rad]
	elif disk_ordering == 'ga':
		theta = params['spiral']['disk_GA_angle'] * np.pi / 180 # GA angle increment [rad]	
	elif disk_ordering == 'tiny_ga':
		N = params['spiral']['disk_tinyGA_N']
		GR = (1 + np.sqrt(5)) / 2  # Golden Ratio
		theta = 1 * np.pi / (GR + N - 1)# tiny GA angle increment [rad] 
	elif disk_ordering == 'phylotaxis':
		raise ValueError("Not implemented.") #TODO

	# First arms in a disk, then new disk
	for plane_number in range(nr_planes):
		hex_theta_curr = hex_theta * plane_number
		
		for arm in range(0, nr_interleaves):
			gsp_xy_ = gsp_xy[arm]
			if hex_sampling:
				gsp_xy_ = rotate(*gsp_xy[arm], axis=base_rot_axis, angle=hex_theta_curr)
			
			if params['spiral']['disk_GA_angle'] == 90:
				angle_ = np.deg2rad(0) if plane_number % 2 == 0 else np.deg2rad(90)
				if arm == 0:
					print(f"Disk {plane_number}: Rotating by {np.rad2deg(angle_)} deg.")
				gsp_xyz_ = tuple(rotate(*gsp_xy_, axis=rot_axis, angle=angle_))
			else:
				gsp_xyz_ = tuple(rotate(*gsp_xy_, axis=rot_axis, angle=theta*plane_number))
			gsp_xyz.append(gsp_xyz_)

	# Record rotation angle (Inplane2, Disk)
	params['spiral']['rotation_angles'][1:] = [np.rad2deg(hex_theta), np.rad2deg(theta)]
	return gsp_xyz

# Bit reversed order
def bit_reversed_view_order(nr_arms):
	idx = np.arange(nr_arms)
	nr_bits = int(np.ceil(np.log2(nr_arms)))
	idx_reversed = np.zeros_like(idx)

	for i in range(nr_arms):
		bits = np.array(list(np.binary_repr(i, width=nr_bits)), dtype=int)
		bits_reversed = bits[::-1]
		idx_reversed[i] = int(''.join(bits_reversed.astype(str)), 2)
	
	idx_reversed_sorted_indices = np.argsort(idx_reversed)
	idx_reversed_sorted = idx_reversed[idx_reversed_sorted_indices]	
	idx_sorted = idx[idx_reversed_sorted_indices]
	return idx_sorted

def get_arm_index(plane_idx, arm_idx, nr_arms_per_plane):
	return plane_idx * nr_arms_per_plane + arm_idx

def get_curr_arm_idx(nr_planes, nr_int_per_plane, curr_plane_idx, interleave_order, nr_total_planes, nr_total_interleaves_per_plane):
	
	curr_arm_indices = []

	if nr_int_per_plane > nr_total_interleaves_per_plane or nr_planes > nr_total_planes:
		raise ValueError("Undersampled arms cannot be more than fully sampled arms.")
	
	# plane indices ['row']. Planes are already GA ordered, so just take the first 'nr_planes' planes.
	plane_indices = np.arange(curr_plane_idx, curr_plane_idx + nr_planes) 
	if plane_indices[-1] >= nr_total_planes:
		plane_indices = plane_indices % nr_total_planes
		interleave_order = np.roll(interleave_order, -1)
	curr_plane_idx = plane_indices[-1] + 1

	# interleave indices ['column']. Interleaves are ordered according to the specified view order. 
	# = get_interleave_indices(nr_total_interleaves_per_plane, interleave_view_order)
	# 26-04-09. Adding inner For loop for adding interleaves first
	interleave_indices = []
	for ii, plane_idx in enumerate(plane_indices):
		for _ in range(nr_int_per_plane):
			interleave_indices.append(interleave_order[0])
			curr_arm_indices.append(int(get_arm_index(plane_idx, interleave_indices[-1], nr_total_interleaves_per_plane)))
			interleave_order = np.roll(interleave_order, -1)

	print(f"Current plane number: {curr_plane_idx}")
	print(f"Current Interleave order: {interleave_order}")

	return curr_arm_indices, curr_plane_idx, interleave_order

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

# View Order
curr_nr_planes	 		= params['vieworder']['nr_planes']
curr_nr_interleaves 	= params['vieworder']['nr_interleaves']
curr_nr_arms_per_frame 	= curr_nr_planes * curr_nr_interleaves

# FOV List for metadata based on the rotation axis in [m]
if params['Orientation']['disk_rot_axis'] == 'x':
	fovs = [sfov*1e-2, rfov*1e-2, rfov*1e-2]
elif params['Orientation']['disk_rot_axis'] == 'y':
	fovs = [rfov*1e-2, sfov*1e-2, rfov*1e-2]
elif params['Orientation']['disk_rot_axis'] == 'z':
	fovs = [rfov*1e-2, rfov*1e-2, sfov*1e-2]
else:
	raise ValueError("Unknown disk rotation axis.")
res_m = res * 1e-3  	# [m]

# nr_planes
if params['spiral']['disk_ordering'] == 'ga':
	nr_planes = params['spiral']['disk_GA_steps']
elif params['spiral']['disk_ordering'] == 'linear' or params['spiral']['disk_ordering'] == 'tiny_ga':
	nr_planes = ceil( (rfov*10 / res) * disk_os * np.pi/2)
	if nr_planes % 2 == 1:
		nr_planes += 1  # make it even
		print("Number of planes increased by 1 to make it even.") 
else:
	raise ValueError("Unknown or not implemented disk ordering.")

params['spiral']['nr_planes'] = nr_planes

# Design the spiral trajectory (nr_interleaves)
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

if params['user_settings']['show_plots']:
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
if params['user_settings']['show_plots']:
	plot_rf(rf, gz, rf_raster_scale = 10, show_plots=True)

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
# Define the initial Disk (Disk0) gradients. set the rotations In-plane.
# ------------------------------------------------------------------------------------
gsp_xy = create_disk_inplane_rot(gsp_x, gsp_y, params, rot_axis=rot_axis)

# ------------------------------------------------------------------------------------
# Create bit reversed order to be used in rotate_disks
# ------------------------------------------------------------------------------------
interleave_order = np.arange(len(gsp_xy)) #bit_reversed_view_order(len(gsp_xy))

# ------------------------------------------------------------------------------------
# set the rotations for the disks
# ------------------------------------------------------------------------------------
gsp_xyz 	= rotate_disks(gsp_xy, params)
n_TRs_total = len(gsp_xyz)

# ------------------------------------------------------------------------------------
# Change view order
# ------------------------------------------------------------------------------------
nr_image_frames = n_TRs_total // curr_nr_arms_per_frame
sampling_mask 	= [None] * nr_image_frames 
gsp_xyz_view 	= []

# Get the arm indices (linear index) for this frame. Update [row, column] = [curr_plane_idx, curr_int_per_plane_idx] indices for the next frame.
curr_plane_idx = 0

for curr_image_frame in range(nr_image_frames):

	curr_arm_indices, curr_plane_idx, interleave_order = get_curr_arm_idx(nr_planes=curr_nr_planes, nr_int_per_plane=curr_nr_interleaves,
																	  curr_plane_idx=curr_plane_idx, interleave_order=interleave_order,
																	  nr_total_planes=nr_planes, nr_total_interleaves_per_plane=len(gsp_xy))
	if curr_arm_indices in sampling_mask:
			raise ValueError("Duplicate sampling mask. Check the view order and the way you update the plane and interleave indices.")

	# for later debug
	sampling_mask[curr_image_frame] = curr_arm_indices

	for ii in range(curr_nr_arms_per_frame):
		gsp_xyz_view.append(gsp_xyz[curr_arm_indices[ii]])

gsp_xyz 	= gsp_xyz_view
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

# TODO: I dont need trigger??
enable_trigger = True
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
# Loop over TRs
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
	if params['spiral']['arm_ordering'] == 'ga':
		raise ValueError("Haven't extended to SPI.")
		seq.add_block(make_label('LIN', 'SET', arm_i % params['spiral']['GA_steps']))
		seq.add_block(*gsp_xyz[arm_i % params['spiral']['GA_steps']], adc) 
	else:  # Linear
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
if params['user_settings']['show_plots']:
	seq.plot(show_blocks=True, grad_disp='mT/m', time_range=[2,2.1], plot_now=False, time_disp='ms')
	k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	# Draw a few spiral in different disks
	Ns = int(k_traj.shape[1] / nr_interleaves / nr_planes)
	get_idx = lambda plane, arm: range( (plane*nr_interleaves + arm)*Ns, (plane*nr_interleaves + arm+1)*Ns)
	color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
	for plane in np.arange(nr_planes, step=20):
		for arm in np.arange(nr_interleaves, step=6):
			idx = get_idx(plane, arm)
			ax.scatter(k_traj[0, idx], k_traj[1, idx], k_traj[2, idx], marker='.', c=color[plane % len(color)], s=1)

	# make axis suqaure
	plt.gca().set_aspect('equal', adjustable='box')
	# double fontsize
	plt.rcParams.update({'font.size': 14})

	#plt.plot(k_traj_adc[0,:], k_traj_adc[1,:], 'rx')
	plt.xlabel('$k_x [mm^{-1}]$')
	plt.ylabel('$k_y [mm^{-1}]$')
	plt.title('k-Space Trajectory')


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
	
	#seq_filename = f"spiral_{params['acquisition']['contrast']}{FA_schedule_str}{prep_str}{end_prep_str}_inplane_{params['spiral']['arm_ordering']}_nrinter_{nr_interleaves:d}_rotplanes_{params['spiral']['disk_ordering']}_nrplanes_{nr_planes}_Tread{params['spiral']['ro_duration']*1e3:.2f}_TR{TR*1e3:.2f}ms_FA{params['acquisition']['flip_angle']}_{m1_str}_{params['user_settings']['filename_ext']}"
	#seq_filename = f"inplane_{params['spiral']['arm_ordering']}_nrinter_{nr_interleaves:d}_rotplanes_{params['spiral']['disk_ordering']}_nrplanes_{nr_planes}_Tread{params['spiral']['ro_duration']*1e3:.2f}_TR{TR*1e3:.2f}ms_FA{params['acquisition']['flip_angle']}_{m1_str}_{params['user_settings']['filename_ext']}"
	# remove double, triple, quadruple underscores, and trailing underscores
	#seq_filename = seq_filename.replace("__", "_").replace("__", "_").replace("__", "_").strip("_")

	seq_filename = f"spiral_res_{res}_vieworder{params['user_settings']['filename_ext']}"

	seq_path = os.path.join(seq_pp_folder, f"{seq_filename}.seq")

	if len(seq_filename) > 255:
		warnings.warn(f"Filename is too long ({len(seq_filename)} characters). Truncating to 255 characters.")
		seq_filename = seq_filename[:255]

	# ensure the out_seq directory exists before writing.
	os.makedirs(seq_pp_folder, exist_ok=True)
	seq.write(seq_path)  # Save to disk

	# Export k-space trajectory
	k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

	# save_traj_dcf(seq.signature_value, k_traj_adc, n_TRs, n_int, fov, res, ndiscard, params['user_settings']['show_plots'])
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
		'disk_ordering': 	 params['spiral']['disk_ordering'],
		'rotation_angles':   params['spiral']['rotation_angles']
	}
	save_metadata_spi(seq.signature_value, k_traj_adc, params_save, params['user_settings']['show_plots'], dcf_method="pipe_menon", out_dir=seq_meta_folder)

	print(f'Metadata file for {seq_filename} is saved as {seq.signature_value} in {seq_meta_folder}/.')
	

# %%