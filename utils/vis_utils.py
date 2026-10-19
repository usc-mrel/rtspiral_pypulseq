
import numpy as np
import matplotlib.pyplot as plt

def plot_rf(rf, gz, rf_raster_scale = 10, show_plots=False):

	if not show_plots:
		return
	
	def _interp_crossing(x0, x1, y0, y1, level):
		if y1 == y0:
			return 0.5 * (x0 + x1)
		return x0 + (level - y0) * (x1 - x0) / (y1 - y0)

	def _nearest_level_crossing(x, y, peak_idx, direction, level):
		i = peak_idx
		while 0 <= i + direction < len(y):
			j = i + direction
			y_i = y[i] - level
			y_j = y[j] - level
			if y_i == 0:
				return x[i]
			if y_i * y_j <= 0:
				return _interp_crossing(x[i], x[j], y[i], y[j], level)
			i = j
		return np.nan

	def _format_width(width_m):
		return f'{width_m * 1e2:.1f} cm' if np.isfinite(width_m) else 'n/a'

	if gz.amplitude == 0:
		raise ValueError('gz.amplitude must be non-zero to calculate the RF profile.')

	gamabar = 42.58  # MHz/T
	
	t_rf_s = rf.t[::rf_raster_scale]
	dt 	  = np.mean(np.diff(t_rf_s)) if len(t_rf_s) > 1 else 1e-6 * rf_raster_scale  # [s]
	t_rf = t_rf_s * 1e3  # [ms]
	rf_  = rf.signal[::rf_raster_scale] / gamabar
	rfe  = np.zeros(len(rf_) * rf_raster_scale, dtype=rf_.dtype)
	idx  = [(len(rfe) - len(rf_))//2, (len(rfe) + len(rf_))//2]
	rfe[idx[0] : idx[1]] = rf_ 
	
	N 	 = len(rfe)
	f 	 = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
	z 	 = f / gz.amplitude  # [m]

	RF = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(rfe)))

	profile = np.abs(RF)
	peak_idx = np.argmax(profile)
	half_max = profile[peak_idx] / 2

	fwhm_left = _nearest_level_crossing(z, profile, peak_idx, -1, half_max)
	fwhm_right = _nearest_level_crossing(z, profile, peak_idx, 1, half_max)
	FWHM = np.abs(fwhm_right - fwhm_left)

	RF_phased = RF * np.exp(-1j * np.angle(RF[peak_idx]))
	profile_signed = np.real(RF_phased)
	fwzc_left = _nearest_level_crossing(z, profile_signed, peak_idx, -1, 0)
	fwzc_right = _nearest_level_crossing(z, profile_signed, peak_idx, 1, 0)
	FWZC = np.abs(fwzc_right - fwzc_left)

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
	plt.plot(z, profile)
	if np.isfinite(FWHM):
		plt.plot([fwhm_left, fwhm_right], [half_max, half_max], 'o-', color='tab:orange',
				 linewidth=2, label=f'FWHM {_format_width(FWHM)}')
		plt.vlines([fwhm_left, fwhm_right], 0, half_max, colors='tab:orange',
				   linestyles='--', alpha=0.6)
	if np.isfinite(FWZC):
		plt.plot([fwzc_left, fwzc_right], [0, 0], 'o-', color='tab:green',
				 linewidth=2, label=f'FWZC {_format_width(FWZC)}')
		plt.vlines([fwzc_left, fwzc_right], 0, profile[peak_idx], colors='tab:green',
				   linestyles=':', alpha=0.5)
	if profile[peak_idx] > 0:
		plt.ylim(-0.05 * profile[peak_idx], 1.05 * profile[peak_idx])
	plt.xlabel('Position [m]')
	plt.ylabel('m_xy')
	plt.grid()
	#plt.title(f'FWHM {_format_width(FWHM)}, FWZC {_format_width(FWZC)}')
	if np.isfinite(FWHM) or np.isfinite(FWZC):
		plt.legend()
	plt.show()
	return


def spi_vis_check(seq, params, nr_interleaves, nr_planes, show_plots = False):
	if show_plots:
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
		plt.show()
	return