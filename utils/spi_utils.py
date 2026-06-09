
from turtle import width

import os
import warnings
import numpy as np
from pypulseq import rotate


def modified_golden_angles(
    base_ga,
    n_angles,
    aspect_ratio,
    fov_shape="rectangle",
    angle_period=2*np.pi,
    input_unit="rad",
    output_unit="rad",
    angle_offset=0.0,
    n_map=20000,
    return_regular=False,
):
    """Generate anisotropic-FOV-adapted golden-angle samples.

    The ordinary golden-angle sequence is generated in a normalized angular
    space and then mapped into physical k-space angles using the inverse CDF
    of the angular density implied by the target FOV shape.

    Parameters
    ----------
    base_ga : float
        Golden-angle increment. Use ``input_unit`` to choose degrees/radians.
        Common values are 137.5078 deg for full-circle rotations and
        111.2461 deg for half-spoke radial ordering.
    n_angles : int
        Number of angles to generate.
    aspect_ratio : float
        Target FOV aspect ratio FOVx/FOVy. For isotropic resolution and a
        fixed number of angles, the absolute FOV scale cancels out.
    fov_shape : {"rectangle", "ellipse"}, optional
        Target support shape used to define angular sampling density.
    angle_period : float, optional
        Angular period in radians. Use 2*pi for full-circle spiral rotations
        and pi for full-spoke radial projection angles.
    input_unit : {"deg", "rad"}, optional
        Unit of ``base_ga`` and ``angle_offset``.
    output_unit : {"rad", "deg"}, optional
        Unit of the returned angles.
    angle_offset : float, optional
        Constant physical angle offset applied after remapping.
    n_map : int, optional
        Number of samples used for the numerical inverse-CDF map.
    return_regular : bool, optional
        If True, return ``(theta_modified, theta_regular)``.

    Returns
    -------
    theta_modified : np.ndarray
        FOV-adapted physical angles.
    theta_regular : np.ndarray, optional
        Regular golden-angle samples before FOV remapping.
    """

    if n_angles < 1:
        raise ValueError("n_angles must be at least 1.")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive.")
    if angle_period <= 0:
        raise ValueError("angle_period must be positive.")
    if n_map < 16:
        raise ValueError("n_map must be at least 16.")

    fov_shape = fov_shape.lower()
    if fov_shape not in ("rectangle", "ellipse"):
        raise ValueError("fov_shape must be 'rectangle' or 'ellipse'.")

    input_unit = input_unit.lower()
    output_unit = output_unit.lower()
    if input_unit not in ("deg", "rad"):
        raise ValueError("input_unit must be 'deg' or 'rad'.")
    if output_unit not in ("deg", "rad"):
        raise ValueError("output_unit must be 'deg' or 'rad'.")

    if input_unit == "deg":
        base_ga_rad = np.deg2rad(base_ga)
        angle_offset_rad = np.deg2rad(angle_offset)
    else:
        base_ga_rad = base_ga
        angle_offset_rad = angle_offset

    theta_regular = np.mod(np.arange(n_angles) * base_ga_rad, angle_period)
    u_regular = theta_regular / angle_period

    theta_grid = np.linspace(0, angle_period, n_map + 1)
    density = _anisotropic_fov_spoke_density(theta_grid, aspect_ratio, fov_shape)
    cdf = _normalized_cumulative_trapezoid(theta_grid, density)

    theta_modified = np.interp(u_regular, cdf, theta_grid)
    theta_modified = np.mod(theta_modified + angle_offset_rad, angle_period)

    if output_unit == "deg":
        theta_modified = np.rad2deg(theta_modified)
        theta_regular = np.rad2deg(theta_regular)

    if return_regular:
        return theta_modified, theta_regular
    return theta_modified


def plot_modified_golden_angles_full_sampling(
    base_ga,
    n_angles,
    aspect_ratio,
    fov_shape="rectangle",
    angle_period=2*np.pi,
    input_unit="rad",
    angle_offset=0.0,
    n_map=20000,
    n_bins=36,
    include_density=True,
    show=True,
):
    """Visualize regular and anisotropic-FOV golden-angle spokes.

    This is intended as a quick fully sampled sanity check. It plots all
    ``n_angles`` regular golden-angle spokes next to all modified spokes and,
    optionally, a histogram against the target angular density.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    axes : np.ndarray
        Axes handles.
    theta_modified : np.ndarray
        Modified angles in radians.
    theta_regular : np.ndarray
        Regular golden-angle samples in radians.
    """

    import matplotlib.pyplot as plt

    theta_modified, theta_regular = modified_golden_angles(
        base_ga=base_ga,
        n_angles=n_angles,
        aspect_ratio=aspect_ratio,
        fov_shape=fov_shape,
        angle_period=angle_period,
        input_unit=input_unit,
        output_unit="rad",
        angle_offset=angle_offset,
        n_map=n_map,
        return_regular=True,
    )

    n_cols = 3 if include_density else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    axes = np.atleast_1d(axes)

    _plot_angle_spokes(
        axes[0],
        theta_regular,
        color=(0.20, 0.45, 0.85),
        title=f"Regular GA, N = {n_angles}",
    )
    _plot_angle_spokes(
        axes[1],
        theta_modified,
        color=(0.85, 0.30, 0.20),
        title=f"{fov_shape.capitalize()} FOV GA, AR = {aspect_ratio:.3g}",
    )

    if include_density:
        _plot_angle_density_check(
            axes[2],
            theta_regular,
            theta_modified,
            aspect_ratio,
            fov_shape,
            angle_period,
            n_map,
            n_bins,
        )

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes, theta_modified, theta_regular


def _anisotropic_fov_spoke_density(theta, aspect_ratio, fov_shape):
    """Angular spoke density for an anisotropic FOV and isotropic resolution."""
    fov_x = aspect_ratio
    fov_y = 1.0
    perpendicular_angle = theta + np.pi/2

    if fov_shape == "rectangle":
        return _rectangle_diameter(perpendicular_angle, fov_x, fov_y)
    if fov_shape == "ellipse":
        return _ellipse_diameter(perpendicular_angle, fov_x, fov_y)
    raise ValueError("fov_shape must be 'rectangle' or 'ellipse'.")


def _rectangle_diameter(angle, width, height):
    """Diameter through the center of a rectangle along a polar angle."""
    c = np.abs(np.cos(angle))
    s = np.abs(np.sin(angle))
    eps_val = np.finfo(float).eps

    d_x = width / np.maximum(c, eps_val)
    d_y = height / np.maximum(s, eps_val)
    return np.minimum(d_x, d_y)


def _ellipse_diameter(angle, width, height):
    """Diameter through the center of an ellipse along a polar angle."""
    c = np.cos(angle)
    s = np.sin(angle)
    eps_val = np.finfo(float).eps
    denom = np.sqrt((c / width)**2 + (s / height)**2)
    return 1.0 / np.maximum(denom, eps_val)


def _normalized_cumulative_trapezoid(x, y):
    """Return a normalized CDF computed with the trapezoidal rule."""
    cdf = np.concatenate(([0.0], np.cumsum((y[:-1] + y[1:]) * np.diff(x) / 2)))
    if cdf[-1] <= 0:
        raise ValueError("density integral must be positive.")
    return cdf / cdf[-1]


def _plot_angle_spokes(ax, theta, color, title, kmax=1.0):
    """Plot radial spokes through the origin on an existing axis."""
    circle_theta = np.linspace(0, 2*np.pi, 512)
    ax.plot(
        kmax*np.cos(circle_theta),
        kmax*np.sin(circle_theta),
        color=(0.80, 0.80, 0.80),
        linewidth=1.0,
    )

    for angle in theta:
        x = kmax * np.array([-np.cos(angle), np.cos(angle)])
        y = kmax * np.array([-np.sin(angle), np.sin(angle)])
        ax.plot(x, y, color=color, linewidth=0.6)

    ax.plot(0, 0, "k.", markersize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.05*kmax, 1.05*kmax)
    ax.set_ylim(-1.05*kmax, 1.05*kmax)
    ax.set_xlabel(r"$k_x / k_{max}$")
    ax.set_ylabel(r"$k_y / k_{max}$")
    ax.set_title(title)
    ax.grid(True)


def _plot_angle_density_check(
    ax,
    theta_regular,
    theta_modified,
    aspect_ratio,
    fov_shape,
    angle_period,
    n_map,
    n_bins,
):
    """Plot angle histograms against the target FOV-dependent density."""
    edges = np.linspace(0, angle_period, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width_deg = np.rad2deg(edges[1] - edges[0])

    regular_counts, _ = np.histogram(theta_regular, bins=edges, density=True)
    modified_counts, _ = np.histogram(theta_modified, bins=edges, density=True)

    theta_grid = np.linspace(0, angle_period, n_map + 1)
    target_density = _anisotropic_fov_spoke_density(theta_grid, aspect_ratio, fov_shape)
    target_density = target_density / np.trapz(target_density, theta_grid)

    centers_deg = np.rad2deg(centers)
    ax.bar(
        centers_deg,
        regular_counts,
        width=0.90*bin_width_deg,
        alpha=0.35,
        color=(0.20, 0.45, 0.85),
        edgecolor="none",
        label="Regular GA",
    )
    ax.bar(
        centers_deg,
        modified_counts,
        width=0.55*bin_width_deg,
        alpha=0.45,
        color=(0.85, 0.30, 0.20),
        edgecolor="none",
        label="Modified GA",
    )
    ax.plot(
        np.rad2deg(theta_grid),
        target_density,
        color="k",
        linewidth=2.0,
        label="Target density",
    )
    ax.set_xlim(0, np.rad2deg(angle_period))
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Angular density")
    ax.set_title("Angular Density Check")
    ax.grid(True)
    ax.legend(loc="best")

# 26-06-07. Added functionality to switch to complement GA.
def _get_fullspoke_ga(params):
	try:
		theta = params['radial']['plane_GA_angle'] * np.pi / 180 # GA angle increment [rad]	
	except KeyError:
		theta = params['spiral']['disk_GA_angle'] * np.pi / 180 # GA angle increment [rad]	

	if theta > (np.pi - theta):
		print(f'Plane GA is given {np.rad2deg(theta):.2f} deg; switching to {np.rad2deg(np.pi - theta):.2f} deg.')
		theta = np.pi - theta
		params['spiral']['GA_angle'] = np.rad2deg(theta)

	return theta

# ------------------------------------------------------------------------------------
# This creates the base spiral plane.
# Later, use this list to create all SPI by rotating.
# ------------------------------------------------------------------------------------
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

	if params['spiral']['arm_ordering'] == 'linear':

		nr_interleaves = params['spiral']['nr_interleaves']
		theta = 2*np.pi/nr_interleaves
		print(f"Rotation Angle in-plane: {np.rad2deg(theta):.2f} deg.")

		for arm in range(nr_interleaves):
			gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis=rot_axis, angle=theta*arm)
			gsp_xy.append((gsp_x_rot, gsp_y_rot))
	
	elif params['spiral']['arm_ordering'] == 'bit-reverse':
		nr_interleaves = params['spiral']['nr_interleaves']
		theta = 2*np.pi/nr_interleaves
		print(f"Rotation Angle in-plane: {np.rad2deg(theta):.2f} deg.")

		arms = list(range(nr_interleaves))
		keys = ['{:0{width}b}'.format(arm, width=int(np.ceil(np.log2(nr_interleaves)))) for arm in arms]
		keys_br = [key[::-1] for key in keys]
		arms_br = [int(key_br, 2) for key_br in keys_br]

		arms_shuffled = np.argsort(arms_br)

		for arm in arms_shuffled:
			gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis=rot_axis, angle=theta*arm)
			gsp_xy.append((gsp_x_rot, gsp_y_rot))

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

# ------------------------------------------------------------------------------------
# This does the rotation of the base spiral. 
# Use another function to shuffle the acquisition order of the arms and disks.
# ------------------------------------------------------------------------------------
def rotate_disks(gsp_xy, params, base_rot_axis="z", DEBUG=False, fov_shape="rectangle"):

	# User parameters
	try:
		nr_planes 	  = params['radial']['nr_planes']
		disk_ordering = params['radial']['plane_ordering']
		rot_axis 	  = params['Orientation']['plane_rot_axis']
		hex_sampling  = params['radial']['hex_sampling']
	except:
		nr_planes 	  = params['spiral']['nr_planes'] 
		disk_ordering = params['spiral']['disk_ordering']
		rot_axis 	  = params['Orientation']['disk_rot_axis']
		hex_sampling  = params['spiral']['hex_sampling']

	nr_interleaves 	= params['spiral']['nr_interleaves']
	# for hexagonal sampling in the (PE,SL) palne
	hex_theta 		= 1 * np.pi / nr_interleaves if hex_sampling else 0

	# Output lists
	gsp_xyz = []

	print(f'Number of disks is set to {nr_planes} for Nyquist sampling.')
	print(f"Disk rotation axis is {rot_axis}.")

	# ----------------------------------------------------------------------------
	# Angle schedule 
	# ----------------------------------------------------------------------------
	flag = False
	if disk_ordering == 'linear':

		theta = 1 * np.pi / nr_planes # linear angle increment [rad]
	
	elif disk_ordering == 'ga':
	
		theta = _get_fullspoke_ga(params) # golden angle increment [rad]
	
	elif disk_ordering == 'tinyga':
		
		GR = (1 + np.sqrt(5)) / 2
		N  = int(params['radial']['plane_tinyGA_N'])
		if N < 3 or N < 0:
			raise ValueError("N for tiny golden angle must be an integer greater than or equal to 3.")
		
		theta = np.deg2rad(180 / (N + GR - 1)) # tiny golden angle increment [rad]
		print(f"Tiny Golden Angle increment: {np.rad2deg(theta):.4f} deg for N={N}.")
		
	elif disk_ordering == 'modga':
		
		base_theta = _get_fullspoke_ga(params)
		flag = True

		# Modified golden angle for anisotropic FOVs (e.g. rectangular)
		aspect_ratio_param = params['radial'].get('plane_aspect_ratio', -1.0)
		if aspect_ratio_param < 0:
			aspect_ratio = (params['acquisition']['fov'][-1] * 1e-2) / (params['acquisition']['slab_thickness'] * 1e-3)
		else:
			aspect_ratio = aspect_ratio_param
		params['radial']['plane_aspect_ratio'] = aspect_ratio
		#aspect_ratio = 1./ aspect_ratio
		theta, theta_reg = modified_golden_angles(
			base_ga=base_theta,
			n_angles=nr_planes,
			aspect_ratio=aspect_ratio,
			fov_shape=fov_shape,
			return_regular=True,
		)

		if DEBUG:
			fig, axes, theta_mod, theta_reg = plot_modified_golden_angles_full_sampling(
				base_ga=base_theta,
				n_angles=nr_planes,
				aspect_ratio=aspect_ratio,
				fov_shape=fov_shape,
			)		
	
	elif disk_ordering == 'phylotaxis':
		raise ValueError("Not implemented.") #TODO

	# ----------------------------------------------------------------------------
	# First arms in a disk, then new disk
	# ----------------------------------------------------------------------------
	plane_theta_list = []
	for plane_number in range(nr_planes):
		
		hex_theta_curr = hex_theta * plane_number
		theta_curr = theta[plane_number] if flag else theta * plane_number

		for arm in range(0, nr_interleaves):
			gsp_xy_ = gsp_xy[arm]
			if hex_sampling:
				gsp_xy_ = rotate(*gsp_xy[arm], axis=base_rot_axis, angle=hex_theta_curr)
			
			gsp_xyz_ = tuple(rotate(*gsp_xy_, axis=rot_axis, angle=theta_curr))
			gsp_xyz.append(gsp_xyz_)
			
		plane_theta_list.append( np.rad2deg(theta_curr))	

	# Record rotation angle (Inplane2, Disk)
	params['spiral']['rotation_angles'][1:] = [np.rad2deg(hex_theta), plane_theta_list]
	return gsp_xyz

# -----------------------------------------------------------------------------
# TODO WIP
# -----------------------------------------------------------------------------
def apply_view_order(gsp_xyz, params):
	n_TRs_total_orig = len(gsp_xyz)

	# Get the view order parameters
	view_order_mode 		= params['view_order']['view_order_mode']
	view_order_submode 		= params['view_order']['view_order_submode']
	undersampling_factor 	= params['view_order']['undersampling_factor']

	# Get plane and in-plane spiral rotation schemes
	plane_ordering, _ = get_plane_ordering_str(params)
	spiral_arm_ordering = params['spiral']['arm_ordering'] 

	# Get the number of interleaves in a plane and the number of planes
	nr_interleaves  = params['spiral']['nr_interleaves']
	nr_planes 		= params['radial']['nr_planes']

	if not (view_order_mode in ['projection-mode', '3d-mode']):
		raise ValueError("view_order_mode must be 'projection-mode' or '3d-mode'.")

	# ---------------------------------------------------------------------------------
	if view_order_mode == 'projection-mode':
		
		if not (view_order_submode in ['fully-sampled', 'undersampled']):
			raise ValueError("view_order_submode must be 'fully-sampled' or 'undersampled'.")
		
		# -----------------------------------------------------------------------------
		# Already designed for projection-mode, fully-sampled.
		# ------------------------------------------------------------------------------
		if view_order_submode == 'fully-sampled':
			print("Selected view order: projection-mode, fully-sampled.") 
			
			if not spiral_arm_ordering == 'linear':
				raise ValueError("For projection-mode view ordering, the in-plane spiral arm ordering must be 'linear' ")

			print("No need for shuffling, keeping the original order.")
		# -----------------------------------------------------------------------------
		# * spiral_arm_ordering
		# - linear: skip some arms
		# - any other scheme just use first N arms that fit the undersampling factor
		# -----------------------------------------------------------------------------
		else:

			if undersampling_factor < 0 or undersampling_factor >= nr_interleaves:
				raise ValueError("Undersampling factor must be between 0 and the number of interleaves (exclusive).")
			
			print(f"Selected view order: projection-mode, undersampled with undersampling factor {undersampling_factor}.")

			if undersampling_factor == 1:
				print("Undersampling factor is 1, keeping the original order.")
			else:
				gsp_xyz = get_projection_mode_undersampled_view_order(gsp_xyz, params, undersampling_factor)
		# -----------------------------------------------------------------------------
	# ---------------------------------------------------------------------------------
	# ---------------------------------------------------------------------------------
	elif view_order_mode == '3d-mode':
		
		if not (view_order_submode in ['fully-sampled', 'undersampled']):
			raise ValueError("view_order_submode must be 'fully-sampled' or 'undersampled'.")
		
		if view_order_submode == 'fully-sampled':
			gsp_xyz = get_3dmode_fullysampled_view_order(gsp_xyz, params)
		else:
			raise NotImplementedError("3d-mode view ordering with undersampling is not implemented yet.")
	
	n_TRs_total = len(gsp_xyz)
	return n_TRs_total, gsp_xyz

# -------------------------------------------------------------------------------------
# 
# -------------------------------------------------------------------------------------
def get_3dmode_fullysampled_view_order(gsp_xyz, params):
	
	# Get the number of planes and interleaves
	nr_planes 		= params['radial']['nr_planes']
	nr_interleaves 	= params['spiral']['nr_interleaves']
	nr_TRs_total 	= len(gsp_xyz)

	# Create the new view order list by iterating through planes and selecting the appropriate arms based on the in-plane ordering scheme.
	gsp_xyz_ = []
	inplane_indices = np.arange(nr_interleaves)		# This will circshift

	while len(gsp_xyz_) < nr_TRs_total:
		
		shift_ = (nr_planes-1) % nr_interleaves if len(gsp_xyz_) > 0 else 0
		inplane_indices = np.roll(inplane_indices, shift=shift_)

		for plane in range(nr_planes):
			
			idx = plane * nr_interleaves + inplane_indices[0]
			gsp_xyz_.append(gsp_xyz[idx])

			#print(f"Plane {plane}: selecting arm with index {inplane_indices[0]} out of {nr_interleaves} total arms.")

			if len(gsp_xyz_) >= nr_TRs_total:
				return gsp_xyz_
			
			inplane_indices = np.roll(inplane_indices, shift=-1)	# circshift in-plane arm indices by 1 for each subsequent plane to achieve a more uniform sampling across planes

	return gsp_xyz_

# -------------------------------------------------------------------------------------
# 
# -------------------------------------------------------------------------------------
def get_projection_mode_undersampled_view_order(gsp_xyz, params, undersampling_factor):

	# Get the number of planes and interleaves
	nr_planes 		= params['radial']['nr_planes']
	nr_interleaves 	= params['spiral']['nr_interleaves']

	# Acceleration factor
	nr_spiral_arms_per_plane = int(np.round(nr_interleaves / undersampling_factor))
	nr_interleaves_rounded   = nr_spiral_arms_per_plane * undersampling_factor
	print(f"Number of spiral arms per plane after undersampling: {nr_spiral_arms_per_plane}.")

	nr_TRs_total_rounded = nr_planes * nr_interleaves_rounded

	# Get the rotation scheme of the in-plane spirals
	spiral_arm_ordering = params['spiral']['arm_ordering']

	# Create the new view order list by iterating through planes and selecting the appropriate arms based on the in-plane ordering scheme.
	gsp_xyz_ = []
	inplane_indices = np.arange(nr_interleaves)		# This will circshift

	if spiral_arm_ordering == 'linear': 
		
		while len(gsp_xyz_) < nr_TRs_total_rounded:
			
			# to make sure I get the uncollected arms for each plane
			shift_ = (nr_planes-1) % nr_interleaves if len(gsp_xyz_) > 0 else 0
			inplane_indices = np.roll(inplane_indices, shift=shift_)

			for plane in range(nr_planes):
				linear_indices_ = plane * nr_interleaves + inplane_indices[::undersampling_factor]
				#print(f"Plane {plane}: selecting arms with indices {inplane_indices[::undersampling_factor]} out of {nr_interleaves} total arms.")
				for idx in linear_indices_:	
					
					if len(gsp_xyz_) >= nr_TRs_total_rounded:
						return gsp_xyz_
					
					gsp_xyz_.append(gsp_xyz[idx])

				# circshift in-plane arm indices by nr_spiral_arms_per_plane for each subsequent plane to achieve a more uniform sampling across planes
				inplane_indices = np.roll(inplane_indices, shift=-1)	

	else:
		
		while len(gsp_xyz_) < nr_TRs_total_rounded:
			
			shift_ = (nr_spiral_arms_per_plane - (nr_planes * nr_spiral_arms_per_plane) % nr_interleaves) if len(gsp_xyz_) > 0 else 0
			inplane_indices = np.roll(inplane_indices, shift=-1*shift_)

			for plane in range(nr_planes):
				linear_indices_ = plane * nr_interleaves + inplane_indices[:nr_spiral_arms_per_plane]
				print(f"Plane {plane}: selecting arms with indices {inplane_indices[:nr_spiral_arms_per_plane]} out of {nr_interleaves} total arms.")
				for idx in linear_indices_:	
					
					if len(gsp_xyz_) >= nr_TRs_total_rounded:
						return gsp_xyz_
					
					gsp_xyz_.append(gsp_xyz[idx])

				# circshift in-plane arm indices by nr_spiral_arms_per_plane for each subsequent plane to achieve a more uniform sampling across planes
				inplane_indices = np.roll(inplane_indices, shift=-1*nr_spiral_arms_per_plane)

	return gsp_xyz_ 

# Note: 
# in XYZ-TRA mode this is the direction mapping, not sure about the signs 
# TRA: * x -> y [AP] ; * y -> x [LR] ; * z -> z [FH]
# SAG: * x -> z [FH] ; * y -> y [AP] ; * z -> x [LR]
# COR: * x -> z [HF] ; * y -> x [LR] ; * z -> y [AP]
# NOTE: Select XYZ-TRA in the scanner and then select COR for the correct SLAB.
def  assign_fovs_and_res_based_on_rotation_axis(params, sfov, rfov, slab, res):

	# Return FOVs in [m]
	try:
		rot_axis = params['Orientation']['plane_rot_axis']
	except:
		rot_axis = params['Orientation']['disk_rot_axis']
	rf_axis  = params['Orientation']['rf_axis']

	rfov = rfov * 1e-2 
	fovs = [rfov, rfov, rfov] # [m]

	if rot_axis == 'x' and rf_axis == 'y': # Designed for the COR setting
		fovs[0] = sfov * 1e-2
		fovs[1] = slab * 1e-3
	else:
		raise ValueError("Rotation axis and RF axis combination not recognized. Check the configs and the assign_fovs_and_res function.")
	
	res_m = res * 1e-3  	# [m]
	return fovs, res_m


# -----------------------------------------------------------------------------
# Naming extensions for sweeps
# -----------------------------------------------------------------------------
def get_plane_ordering_str(params):

	try:
		plane_ordering = params['spiral']['disk_ordering']
	except:
		plane_ordering = params['radial']['plane_ordering']

	plane_ordering_str = ''
	if 'modga' in plane_ordering:
		aspect_ratio = params['radial'].get('plane_aspect_ratio', -1.0)
		if aspect_ratio < 0:
			aspect_ratio =  (params['acquisition']['slab_thickness'] * 1e-3) / (params['acquisition']['fov'][-1] * 1e-2)
		plane_ordering_str += f'_modGA_AR_{aspect_ratio:0.1f}'
	elif 'tinyga' in plane_ordering:
		plane_ordering_str += f'_tinyGA_N_{params["radial"]["plane_tinyGA_N"]}'
	elif 'ga' in plane_ordering:
		plane_ordering_str += f'_GA_'
	else:
		plane_ordering_str += f'_{plane_ordering}'

	return plane_ordering, plane_ordering_str

# -----------------------------------------------------------------------------
# Get a shortish view order description
# -----------------------------------------------------------------------------
def get_view_order_str(params):
	try:
		view_order_mode 		= params['view_order']['view_order_mode']
		view_order_submode 		= params['view_order']['view_order_submode']
		undersampling_factor 	= params['view_order']['undersampling_factor']

		if view_order_mode == 'projection-mode':
			view_order_mode_str = 'projmode'
			view_order_submode_str = 'fs' if view_order_submode == 'fully-sampled' else f'us_{undersampling_factor}'
		
			return f'{view_order_mode_str}_{view_order_submode_str}'
		elif view_order_mode == '3d-mode':
			view_order_mode_str = '3dmode'
			return f'{view_order_mode_str}'
		else:
			raise ValueError("view_order_mode must be 'projection-mode' or '3d-mode'.")
	except KeyError:
		return ''

def get_seq_name(seq_pp_folder, params, FA_schedule_str=None, prep_str='', end_prep_str='', m1_str=''):
	
	hex_sampling_str 	= '_hex' if params['radial']['hex_sampling'] else ''
	_, plane_order_str 	= get_plane_ordering_str(params)
	inplane_order_str 	= params['spiral']['arm_ordering']
	vieworder_str 		= get_view_order_str(params)  
	FA_schedule_str 	= FA_schedule_str if FA_schedule_str is not None and FA_schedule_str != '_' else f'{params['acquisition']['flip_angle']}'

	seq_filename = f'SPI_{params['acquisition']['contrast']}_slab_{params['acquisition']['slab_thickness']}mm_vieworder_{vieworder_str}_radial_plane_{plane_order_str}{hex_sampling_str}_spiral_inplane_{inplane_order_str}_FA_{FA_schedule_str}_{prep_str}_{end_prep_str}_{m1_str}_{params['user_settings']['filename_ext']}'

	# remove double, triple, quadruple underscores, and trailing underscores
	seq_filename = seq_filename.replace("__", "_").replace("__", "_").replace("__", "_").strip("_")

	if len(seq_filename) > 255:
		raise ValueError(f"Generated filename is too long ({len(seq_filename)} characters). Consider shortening the filename components or the filename extension.")

	seq_fullname = os.path.join(seq_pp_folder, f"{seq_filename}.seq")

	return seq_fullname