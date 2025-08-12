import numpy as np
from libspiralutils import pts_to_waveform, design_rewinder_exact_time
import copy
from .rewinder_m1_nayak import spiral_rewinder_m1_nayak
from gropt import get_min_TE_gfix
from scipy.optimize import minimize_scalar
import warnings

def calculate_moments(g_grad, GRT):
    """
    Calculate the zeroth and first moments of the gradient waveform.
    """
    M0 = np.cumsum(g_grad, axis=0) * GRT * 1e3  # [mT.s/m] -> [mT.ms/m]
    t = (np.arange(1, g_grad.shape[0] + 1)) * GRT * 1e3  # [ms]
    tt = (t * np.ones((2, 1))).T
    M1 = np.cumsum(g_grad * tt, axis=0) * GRT * 1e3  # [mT.ms^2/m]
    return M0, M1

def rotate_gradients(g_grad, angle):
    """
    Rotate the gradient waveform by a given angle.
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle),  np.cos(angle)]])
    return g_grad @ rotation_matrix

def design_rewinders_gropt(g_grad, M0, M1, M1_nulling, T_rew, gropt_params, verbose=False):

    GRT = gropt_params['dt']  # Gradient raster time in seconds
    t = (np.arange(1, g_grad.shape[0] + 1)) * GRT * 1e3  # [ms]
    gropt_params['moment_params']  = [[0, 0, 0, -1, -1, -M0[-1,0], 1.0e-5]]
    
    if M1_nulling:
        gropt_params['moment_params'].append([0, 1, 0, -1, -1, -M1[-1,0]+M0[-1,0]*(t[-1]+GRT*1e3), 1.0e-5])

    gropt_params['gfix']  = np.array([g_grad[-1, 0]*1e-3, -99999, 0])

    g_rewind_x, _ = get_min_TE_gfix(gropt_params, T_rew*1e3, verbose)
    g_rewind_x = g_rewind_x.T[:,0]*1e3

    gropt_params['moment_params']  = [[0, 0, 0, -1, -1, -M0[-1,1], 1.0e-5]]

    if M1_nulling:
        gropt_params['moment_params'].append([0, 1, 0, -1, -1, -M1[-1,1]+M0[-1,1]*(t[-1]+GRT*1e3), 1.0e-5])

    gropt_params['gfix']  = np.array([g_grad[-1, 1]*1e-3, -99999, 0])

    g_rewind_y, _ = get_min_TE_gfix(gropt_params, T_rew*1e3, verbose)
    g_rewind_y = g_rewind_y.T[:,0]*1e3
    return g_rewind_x, g_rewind_y


def design_rewinder(g_grad, T_rew, system, slew_ratio=0.7, grad_rew_method='gropt', M1_nulling=False, rotate_grads=False) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Design_rewinder for spiral trajectories.

    Inputs:
        g_grad: array [readout x 2] in units mT/m 
        GRT: gradient raster time (s)
        T_rew: time for rewind (s)
        system: pulseq system object (containing max grad, slew, and other hardware limits)
        slew_ratio: slew ratio to limit spiral rewinder
        grad_rew_method: 'gropt', 'ext_trap_area', 'exact_time', 'm1_nayak'
        M1_nulling: true or false (only available with 'gropt')
        rotate_grads: true or false (to rotate gradients for optimal rewinding timing)

    Returns:
        g_rewind_x 
        g_rewind_y
        g_rotated: Rotated gradients. Returned if rotate_grads is True
    """
    GRT = system.grad_raster_time

    # === design rewinder ===
    M0, M1 = calculate_moments(g_grad, GRT)

    grad_scale = 42.58e3 # from mT/m -> Hz/m
    slew_scale = 42.58e6 # from T/m/s -> Hz/m/s

    max_grad = system.max_grad / grad_scale
    max_slew = system.max_slew / slew_scale
    g_rewind_x = np.zeros(0)
    g_rewind_y = np.zeros(0)

    # Design rew with gropt
    if grad_rew_method == 'gropt':
        gropt_params = {}
        gropt_params['mode'] = 'free'
        gropt_params['gmax'] = max_grad*1e-3 # [mT/m] -> [T/m]
        gropt_params['smax'] = max_slew * slew_ratio
        gropt_params['dt']   = GRT

        if rotate_grads:
            def rotate_and_design(theta):
                # Rotate the gradient by theta
                g_rotated = rotate_gradients(g_grad, theta)
                M0, M1 = calculate_moments(g_rotated, GRT)
                g_rewind_x, g_rewind_y = design_rewinders_gropt(g_rotated, M0, M1, M1_nulling, T_rew, gropt_params)

                return abs(g_rewind_x.shape[0] - g_rewind_y.shape[0])

            res = minimize_scalar(rotate_and_design, bounds=(-np.pi/2, np.pi/2), method='bounded')
            g_rotated = rotate_gradients(g_grad, res.x)
            M0, M1 = calculate_moments(g_rotated, GRT)
            g_rewind_x, g_rewind_y = design_rewinders_gropt(g_rotated, M0, M1, M1_nulling, T_rew, gropt_params, verbose=True)
            return g_rewind_x, g_rewind_y, g_rotated

        else:
            g_rewind_x, g_rewind_y = design_rewinders_gropt(g_grad, M0, M1, M1_nulling, T_rew, gropt_params, verbose=True)

    elif grad_rew_method == 'ext_trap_area':
        from pypulseq.make_extended_trapezoid_area import make_extended_trapezoid_area

        if M1_nulling:
            warnings.warn("M1 nulling is not supported with 'ext_trap_area' method. It will be ignored.")
            M1_nulling = False
        # Copy the system to modify slew rate to obey reduced SR of the spirals.
        system2 = copy.deepcopy(system)
        system2.max_slew = system.max_slew * slew_ratio
        _,times_x,amplitudes_x = make_extended_trapezoid_area(channel='x', area=-M0[-1,0]*system2.gamma*1e-6, grad_start=g_grad[-1, 0]*system2.gamma*1e-3, grad_end=0, system=system2)
        _,times_y,amplitudes_y = make_extended_trapezoid_area(channel='y', area=-M0[-1,1]*system2.gamma*1e-6, grad_start=g_grad[-1, 1]*system2.gamma*1e-3, grad_end=0, system=system2)

        g_rewind_x = 1e3*pts_to_waveform(times_x, amplitudes_x, GRT)/system2.gamma
        g_rewind_y = 1e3*pts_to_waveform(times_y, amplitudes_y, GRT)/system2.gamma

    elif grad_rew_method == 'exact_time':
        
        if M1_nulling:
            warnings.warn("M1 nulling is not supported with 'exact_time' method. It will be ignored.")
            M1_nulling = False

        spiral_sys = {
            'max_slew'          :  max_slew * slew_ratio,   # [T/m/s] 
            'max_grad'          :  max_grad*0.99,   # [mT/m] 
            'grad_raster_time'  :  GRT, # [s]
            }

        [times_x, amplitudes_x] = design_rewinder_exact_time(g_grad[-1, 0], 0, T_rew, -M0[-1,0]*1e-3, spiral_sys)
        [times_y, amplitudes_y] = design_rewinder_exact_time(g_grad[-1, 1], 0, T_rew, -M0[-1,1]*1e-3, spiral_sys)

        g_rewind_x = pts_to_waveform(times_x, amplitudes_x, GRT)
        g_rewind_y = pts_to_waveform(times_y, amplitudes_y, GRT)

    elif grad_rew_method == 'm1_nayak':
        # Krishna rewinder
        system2 = copy.deepcopy(system)
        system2.max_slew = system.max_slew * slew_ratio
        g_grad, g_rewind_x, g_rewind_y = spiral_rewinder_m1_nayak(g_grad, GRT, system2)

    # add zeros to the end of g_rewind_x or g_rewind_y to make them the same length (in case they are not).
    if len(g_rewind_x) > len(g_rewind_y):
        g_rewind_y = np.concatenate((g_rewind_y, np.zeros(len(g_rewind_x) - len(g_rewind_y))))
    else:
        g_rewind_x = np.concatenate((g_rewind_x, np.zeros(len(g_rewind_y) - len(g_rewind_x))))

    return g_rewind_x, g_rewind_y
