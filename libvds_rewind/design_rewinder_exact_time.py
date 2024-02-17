import numpy as np
from scipy.optimize import minimize_scalar, minimize, LinearConstraint, NonlinearConstraint, Bounds
import math

from .rounding import round_to_GRT, round_up_to_GRT
from .trap_moments import trap_moment, trap_moment_exact_time 

def design_rewinder_exact_time(Gs, Ge, T, M, spiral_sys):
    """" design_rewinder_exact_time Given moment, start, end gradients and spiral_system limits, design a trapezoid.
    Parameters:
    -----------
    Gs: Starting gradient (mT/m)
    Ge: Ending gradient (mT/m)
    T: Total rewinder time (s)
    M: Desired moment (s.mT/m)
    spiral_sys: struct with following fields;
        area_tol: Moment error tolerance (MSE sense)
        max_grad: Max gradient amplitude (mT/m)
        max_slew: Max gradient slew rate (T/m/s)
        grad_raster_time: Gradient raster time [s]

    Returns:
    ---------
    times:  Time points of the trapezoid [s]
    amplitudes: Corresponding amplitudes of the times [mT/m]
    """

    # convert grad to Hz/m and slew to Hz/m/s
    grad_scale = 42.58e3 # from mT/m -> Hz/m
    slew_scale = 42.58e6 # from T/m/s -> Hz/m/s
    Gs = Gs * grad_scale
    Ge = Ge * grad_scale
    M = M * grad_scale

    max_grad = spiral_sys['max_grad'] * grad_scale
    max_slew = spiral_sys['max_slew'] * slew_scale

    if 'area_tol' in spiral_sys:
        area_tol = spiral_sys['area_tol']  # (MSE)
    else:
        area_tol = 1e-5

    SR = max_slew * 0.99  # otherwise we run into rounding errors during resampling
    dT = spiral_sys['grad_raster_time']

    # Set up the optimization.
    gscale = 1.0 / (max_grad)

    # Determine if a triangle should suffice.
    lb = -max_grad * gscale
    ub = max_grad * gscale

    # Iterate flat top time upper bound till we find a reasonable tradeoff between duration and precision.
    obj2 = lambda x: (M - trap_moment_exact_time(x / gscale, T, SR, dT, Gs, Ge))**2
    result = minimize_scalar(obj2, bounds=(lb, ub), method='bounded', options = {'disp': False, 'maxiter': 1000})
    Gp = result.x / gscale

    # Get results and derive other points from it.
    Tr = round_up_to_GRT(abs(Gp - Gs) / SR, GRT=dT)
    Tf = round_up_to_GRT(abs(Gp - Ge) / SR, GRT=dT)
    Tp = round_to_GRT((T - Tr - Tf), GRT=dT)

    # Validate if the result is still reasonable after roundings
    ga1 = trap_moment(Gp, Tr, Tp, Tf, Gs, Ge)
    err = (M - ga1)**2
    if err > area_tol:
        print(f'Optimization yielded larger than tolerated error.\nTolerance={area_tol}, error={err}')

    if Tp == 0:
        times = np.cumsum([0, Tr, Tf])
        amplitudes = np.array([Gs, Gp, Ge])
    else:
        times = np.cumsum([0, Tr, Tp, Tf])
        amplitudes = np.array([Gs, Gp, Gp, Ge])

    # convert amplitudes back to mT/m
    amplitudes = amplitudes / grad_scale

    return times, amplitudes

def design_rewinder_exact_flattop(Gs, Ge, Tr, Tp, Tf, M, spiral_sys):
    """design_rewinder_exact_flattop Given moment, start, end gradients and spiral_system limits, design a trapezoid, .
    Parameters:
    -----------
    Gs: Starting gradient [mT/m]
    Ge: Ending gradient [mT/m]
    Tr: Rewinder rise time [s]
    Tp: Rewinder flat top time [s]
    Tf: Rewinder fall time [s]
    M: Desired moment [s.mT/m]
    spiral_sys: dict with following fields;
        area_tol: Moment error tolerance (MSE sense)
        max_grad: Max gradient amplitude (mT/m)
        max_slew: Max gradient slew rate (T/m/s)
        grad_raster_time: Gradient raster time [s]

    Returns:
    ---------
    times:  Time points of the trapezoid [s]
    amplitudes: Corresponding amplitudes of the times [mT/m]

    """

    # convert grad to Hz/m and slew to Hz/m/s
    grad_scale = 42.58e3 # from mT/m -> Hz/m
    slew_scale = 42.58e6 # from T/m/s -> Hz/m/s
    Gs = Gs * grad_scale
    Ge = Ge * grad_scale
    M = M * grad_scale

    max_grad = spiral_sys['max_grad'] * grad_scale
    max_slew = spiral_sys['max_slew'] * slew_scale
    if 'area_tol' in spiral_sys:
        area_tol = spiral_sys['area_tol']  # (MSE)
    else:
        area_tol = 1e-5

    SR = max_slew * 0.99  # otherwise we run into rounding errors during resampling
    dT = spiral_sys['grad_raster_time']

    # Set up the optimization.
    gscale = 1.0 / (max_grad)

    # Determine if a triangle should suffice.
    lb = -max_grad * gscale
    ub = max_grad * gscale

    # So, this time we have all the timing, only variable is the amplitude (or fall and rise slew rates). Amplitude is more intuitive to optimize.
    obj2 = lambda x: (M - trap_moment(x / gscale, Tr, Tp, Tf, Gs, Ge))**2
    result = minimize_scalar(obj2, bounds=(lb, ub), method='bounded', options = {'disp': False, 'maxiter': 1000})
    Gp = result.x / gscale

    # Validate if the result is still reasonable after roundings
    ga1 = trap_moment(Gp, Tr, Tp, Tf, Gs, Ge)
    err = (M - ga1)**2
    if err > area_tol:
        print(f'Optimization yielded larger than tolerated error.\nTolerance={area_tol}, error={err}')

    if Tp == 0:
        times = np.cumsum([0, Tr, Tf])
        amplitudes = np.array([Gs, Gp, Ge])
    else:
        times = np.cumsum([0, Tr, Tp, Tf])
        amplitudes = np.array([Gs, Gp, Gp, Ge])

    # convert amplitudes back to mT/m
    amplitudes = amplitudes / grad_scale

    return times, amplitudes

def design_joint_rewinder_exact_time(Gs, Ge, T, M, spiral_sys):
    """" design_joint_rewinder_exact_time Given x and y gradient moment, start, end gradients and spiral_system limits, design x and y trapezoids.
    Parameters:
    -----------
    Gs: 2 element array of starting gradients [mT/m]
    Ge: 2 element array of ending gradients [mT/m]
    T: Total rewinder time [s]
    M: 2 element array of Desired moment [s.mT/m]
    spiral_sys: struct with following fields;
        area_tol: Moment error tolerance [MSE sense]
        max_grad: Max gradient amplitude [mT/m]
        max_slew: Max gradient slew rate [T/m/s]
        grad_raster_time: Gradient raster time [s]

    Returns:
    ---------
    times:  Time points of the trapezoid [s]
    amplitudes: Corresponding amplitudes of the times [mT/m]
    """
    # convert grad to Hz/m and slew to Hz/m/s
    grad_scale = 42.58e3 # from mT/m -> Hz/m
    slew_scale = 42.58e6 # from T/m/s -> Hz/m/s
    Gs = Gs * grad_scale
    Ge = Ge * grad_scale
    M = M * grad_scale

    max_grad = spiral_sys['max_grad'] * grad_scale
    max_slew = spiral_sys['max_slew'] * slew_scale

    if 'area_tol' in spiral_sys:
        area_tol = spiral_sys['area_tol']  # (MSE)
    else:
        area_tol = 1e-5

    SR = max_slew * 0.99  # otherwise we run into rounding errors during resampling
    dT = spiral_sys['grad_raster_time']

    # Set up the optimization.
    gscale = 1e-3 / (max_grad)

    # Set bounds and initial points
    lb = np.array([0, 0, -max_grad * gscale, -max_grad * gscale])
    ub = np.array([T, T, max_grad * gscale, max_grad * gscale])
    bnds = Bounds(lb, ub)

    # Test initing with exact_time:
    [timesx_, ampsx_] = design_rewinder_exact_time(Gs[0]/grad_scale, Ge[0]/grad_scale, T, M[0]/grad_scale, spiral_sys)
    [timesy_, ampsy_] = design_rewinder_exact_time(Gs[1]/grad_scale, Ge[1]/grad_scale, T, M[1]/grad_scale, spiral_sys)
    Tr0 = ((timesx_[1] - timesx_[0]) + (timesy_[1] - timesy_[0]))/2
    Tf0 = ((timesx_[3] - timesx_[2]) + (timesy_[3] - timesy_[2]))/2
    x0 = np.array([Tr0, Tf0, ampsx_[1]*grad_scale*gscale, ampsy_[1]*grad_scale*gscale]) # TODO: calling desing_rewinder_exact_time might be a good init point.

    # Opt problem is min_{Tr, Tf, Gx_p, Gy_p}|(Mx, My) - M|_2^2, s.t.
    # Tr + Tp + Tf = Trew => This is embedded into goal fcn as Tp = Trew-Tr-Tf
    # Non-linear constraints
    # (Gx_p - Gx_s)/Tr < SR
    # (Gy_p - Gy_s)/Tr < SR
    # (Gx_p - Gx_e)/Tf < SR
    # (Gy_p - Gy_e)/Tf < SR

    # Write an fcn to specify non-linear constraint lb <= f(x) <= ub. n = 4 (no of ind. vars).
    # These are constraints on slew-rates
    def nconst(x):
        return np.array([
            (x[2]/gscale - Gs[0])/x[0],
            (x[3]/gscale - Gs[1])/x[0],
            (x[2]/gscale - Ge[0])/x[1],
            (x[3]/gscale - Ge[1])/x[1],
            # Test: can we give Tr, Tf GRT constraint as nlin const?
            (x[0]*1e5)%1,
            (x[1]*1e5)%1,
        ])

    nlconst = NonlinearConstraint(nconst, 
                                #   np.array([-SR, -SR, -SR, -SR, -np.finfo(np.float64).eps, -np.finfo(np.float64).eps]), # lb
                                #   np.array([ SR,  SR,  SR,  SR,  np.finfo(np.float64).eps,  np.finfo(np.float64).eps])) # ub
                                np.array([-SR, -SR, -SR, -SR, -1e-3, -1e-3]), # lb
                                np.array([ SR,  SR,  SR,  SR,  1e-3,  1e-3])) # ub
        
    # TODO: Linearity constraint forcing Tp > 0 => Trew-Tr-Tf > 0
    # n = 4, m = 1 (no of constrts).
    A = np.array([[1,1,0,0]])
    lconst = LinearConstraint(A, 0, T)

    # Iterate flat top time upper bound till we find a reasonable tradeoff between duration and precision.
    # x = (Tr, Tf, Gx_p, Gy_p)
    #                 M0_x - M_x(       Gx_p,  Tr,  Trew-Tr-Tf,  Tf,   Gx_s,  0)^2   +  M0_y - M_y(       Gy_p,  Tr,  Trew-Tr-Tf,  Tf,   Gy_s,  0)^2
    # TODO: figure out how to enforce Tr and Tf on GRT.
    def trap_moment_GRT(x, Gp_, Gs_):
        Tr_ = round_up_to_GRT(x[0], GRT=dT)
        Tf_ = round_up_to_GRT(x[1], GRT=dT)
        # Tr_ = x[0]
        # Tf_ = x[1]
        return trap_moment(Gp_, Tr_, T-Tr_-Tf_, Tf_, Gs_, 0)

    obj = lambda x: (M[0] - trap_moment_GRT(x, x[2]/gscale, Gs[0]))**2 + (M[1] - trap_moment_GRT(x, x[3]/gscale, Gs[1]))**2
    result = minimize(obj, x0=x0, bounds=bnds, constraints=(nlconst,lconst), method='trust-constr', 
                      options = {'disp': False, 'maxiter': 10000, 'xtol': 1e-12})
    print(f'Optim is {result.success}: {result.message}')
    Gp = result.x[2:]/gscale

    # Get results and derive other points from it.
    Tr = round_up_to_GRT(result.x[0], GRT=dT)
    Tf = round_up_to_GRT(result.x[1], GRT=dT)
    Tp = round_to_GRT((T - Tr - Tf), GRT=dT)

    # Tr = result.x[0]
    # Tf = result.x[1]
    # Tp = (T - Tr - Tf)

    # Validate if the result is still reasonable after roundings
    ga1 = np.array([trap_moment(Gp[0], Tr, Tp, Tf, Gs[0], Ge[0]), trap_moment(Gp[1], Tr, Tp, Tf, Gs[1], Ge[1])])
    err = (M - ga1)**2
    if (err[0] > area_tol) or (err[1] > area_tol):
        print(f'Optimization yielded larger than tolerated error.\nTolerance={area_tol}, error={err}')

    if Tp == 0:
        times = np.cumsum([0, Tr, Tf])
        amplitudes = np.array([Gs, Gp, Ge])
    else:
        times = np.cumsum([0, Tr, Tp, Tf])
        amplitudes = np.array([Gs, Gp, Gp, Ge])

    # convert amplitudes back to mT/m
    amplitudes = amplitudes / grad_scale

    return times, amplitudes


def joint_rewinder_relaxed_time(Gs, Ge, T, M, spiral_sys):
    """" design_joint_rewinder_exact_time Given x and y gradient moment, start, end gradients and spiral_system limits, design x and y trapezoids.
    Parameters:
    -----------
    Gs: 2 element array of starting gradients [mT/m]
    Ge: 2 element array of ending gradients [mT/m]
    T: Total rewinder time [s]
    M: 2 element array of Desired moment [s.mT/m]
    spiral_sys: struct with following fields;
        area_tol: Moment error tolerance [MSE sense]
        max_grad: Max gradient amplitude [mT/m]
        max_slew: Max gradient slew rate [T/m/s]
        grad_raster_time: Gradient raster time [s]

    Returns:
    ---------
    times:  Time points of the trapezoid [s]
    amplitudes: Corresponding amplitudes of the times [mT/m]
    """
    # convert grad to Hz/m and slew to Hz/m/s
    grad_scale = 42.58e3 # from mT/m -> Hz/m
    slew_scale = 42.58e6 # from T/m/s -> Hz/m/s
    Gs = Gs * grad_scale
    Ge = Ge * grad_scale
    M = M * grad_scale

    max_grad = spiral_sys['max_grad'] * grad_scale
    max_slew = spiral_sys['max_slew'] * slew_scale

    if 'area_tol' in spiral_sys:
        area_tol = spiral_sys['area_tol']  # (MSE)
    else:
        area_tol = 1e-5

    SR = max_slew * 0.99  # otherwise we run into rounding errors during resampling
    dT = spiral_sys['grad_raster_time']

    # Set up the optimization.
    gscale = 1e-3 / (max_grad)

    # Set bounds and initial points
    lb = np.array([0, 0, 0, -max_grad * gscale, -max_grad * gscale])
    ub = np.array([T, T, T, max_grad * gscale, max_grad * gscale])
    bnds = Bounds(lb, ub)

    # Test initing with exact_time:
    [timesx_, ampsx_] = design_rewinder_exact_time(Gs[0]/grad_scale, Ge[0]/grad_scale, T, M[0]/grad_scale, spiral_sys)
    [timesy_, ampsy_] = design_rewinder_exact_time(Gs[1]/grad_scale, Ge[1]/grad_scale, T, M[1]/grad_scale, spiral_sys)
    Tr0 = ((timesx_[1] - timesx_[0]) + (timesy_[1] - timesy_[0]))/2
    Tf0 = ((timesx_[3] - timesx_[2]) + (timesy_[3] - timesy_[2]))/2
    x0 = np.array([Tr0, T-Tr0-Tf0, Tf0, ampsx_[1]*grad_scale*gscale, ampsy_[1]*grad_scale*gscale]) 
    # x0 = np.array([dT, dT, round_up_to_GRT(max(abs(Gs[0]), abs(Gs[1]))/SR), Gs[0]*gscale, Gs[1]*gscale]) # TODO: calling desing_rewinder_exact_time might be a good init point.

    # Opt problem is min_{Tr, Tp, Tf, Gx_p, Gy_p}|(Mx, My) - M|_2^2, s.t.
    # Non-linear constraints
    # (Gx_p - Gx_s)/Tr < SR
    # (Gy_p - Gy_s)/Tr < SR
    # (Gx_p - Gx_e)/Tf < SR
    # (Gy_p - Gy_e)/Tf < SR

    # Write an fcn to specify non-linear constraint lb <= f(x) <= ub. n = 4 (no of ind. vars).
    # These are constraints on slew-rates
    def nconst(x):
        return np.array([
            (x[3]/gscale - Gs[0])/x[0],
            (x[4]/gscale - Gs[1])/x[0],
            (x[3]/gscale - Ge[0])/x[2],
            (x[4]/gscale - Ge[1])/x[2],
            # Test: can we give Tr, Tf GRT constraint as nlin const?
            round(x[0]/dT) - x[0]/dT,#(x[0]/dT)%1,
            round(x[1]/dT) - x[1]/dT,#(x[1]/dT)%1,
            round(x[2]/dT) - x[2]/dT,#(x[2]/dT)%1,
        ])
    GRT_tol = 1e-12
    nlconst = NonlinearConstraint(nconst, 
                                #   np.array([-SR, -SR, -SR, -SR, -np.finfo(np.float64).eps, -np.finfo(np.float64).eps]), # lb
                                #   np.array([ SR,  SR,  SR,  SR,  np.finfo(np.float64).eps,  np.finfo(np.float64).eps])) # ub
                                np.array([-SR, -SR, -SR, -SR, 0, 0, 0]), # lb
                                np.array([ SR,  SR,  SR,  SR, GRT_tol,  GRT_tol,  GRT_tol]),
                                keep_feasible=False) # ub
        
    # Tr + Tp + Tf < Trew
    # n = 5, m = 1 (no of constrts).
    A = np.array([[1, 1, 1, 0, 0]])
    lconst = LinearConstraint(A, 0, T)

    # Iterate flat top time upper bound till we find a reasonable tradeoff between duration and precision.
    # x = (Tr, Tp, Tf, Gx_p, Gy_p)
    #                 M0_x - M_x(       Gx_p,  Tr,  Trew-Tr-Tf,  Tf,   Gx_s,  0)^2   +  M0_y - M_y(       Gy_p,  Tr,  Trew-Tr-Tf,  Tf,   Gy_s,  0)^2
    # TODO: figure out how to enforce Tr and Tf on GRT.
    def trap_moment_GRT(x, Gp_, Gs_):
        # Tr_ = round_up_to_GRT(x[0], GRT=dT)
        # Tp_ = round_up_to_GRT(x[1], GRT=dT)
        # Tf_ = round_up_to_GRT(x[2], GRT=dT)
        Tr_ = x[0]
        Tp_ = x[1]
        Tf_ = x[2]
        return trap_moment(Gp_, Tr_, Tp_, Tf_, Gs_, 0)

    obj = lambda x: (M[0] - trap_moment_GRT(x, x[3]/gscale, Gs[0]))**2 + (M[1] - trap_moment_GRT(x, x[4]/gscale, Gs[1]))**2
    result = minimize(obj, x0=x0, bounds=bnds, constraints=(nlconst,lconst), method='trust-constr', 
                      options = {'disp': True, 'maxiter': 10000})
    print(f'Optim is {result.success}: {result.message}')
    Gp = result.x[3:]/gscale

    # Get results and derive other points from it.
    Tr = round_up_to_GRT(result.x[0], GRT=dT)
    Tf = round_up_to_GRT(result.x[2], GRT=dT)
    Tp = round_up_to_GRT(result.x[1], GRT=dT)

    # Tr = result.x[0]
    # Tf = result.x[1]
    # Tp = (T - Tr - Tf)

    # Validate if the result is still reasonable after roundings
    ga1 = np.array([trap_moment(Gp[0], Tr, Tp, Tf, Gs[0], Ge[0]), trap_moment(Gp[1], Tr, Tp, Tf, Gs[1], Ge[1])])
    err = (M - ga1)**2
    if (err[0] > area_tol) or (err[1] > area_tol):
        print(f'Optimization yielded larger than tolerated error.\nTolerance={area_tol}, error={err}')

    if Tp == 0:
        times = np.cumsum([0, Tr, Tf])
        amplitudes = np.array([Gs, Gp, Ge])
    else:
        times = np.cumsum([0, Tr, Tp, Tf])
        amplitudes = np.array([Gs, Gp, Gp, Ge])

    # convert amplitudes back to mT/m
    amplitudes = amplitudes / grad_scale

    return times, amplitudes

if __name__ == '__main__':
    spiral_sys = {'max_slew': 120, 'max_grad': 30, 'adc_dwell': 1e-06, 'grad_raster_time': 1e-05, 'os': 8}
    Gxe = -19.11041988831517
    Gye = 18.014631437132557
    Mx = 0.0036346637402215626
    My = 0.004473246710403885
    Tr = 0.1e-3
    Tf = 0.1e-3
    Tp = 0.8e-3
    T_rew = Tr + Tf + Tp

    [times_x, amplitudes_x] = design_rewinder_exact_time(Gxe, 0, T_rew, -Mx, spiral_sys)
    [times_y, amplitudes_y] = design_rewinder_exact_time(Gye, 0, T_rew, -My, spiral_sys)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(times_x, amplitudes_x)
    plt.plot(times_y, amplitudes_y)