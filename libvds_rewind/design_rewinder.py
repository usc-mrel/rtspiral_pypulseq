import numpy as np
from scipy.optimize import minimize_scalar

from .rounding import round_to_GRT, round_up_to_GRT
from .trap_moments import trap_moment, trap_moment_exact_time


def design_rewinder(Gs, Ge, M, sys):
    # fancier rewinder. TODO.
    pass


def design_rewinder_exact_time(Gs, Ge, T, M, sys):
    #DESIGNREWINDER Given moment, start, end gradients and system limits, design a trapezoid.
    # Gs: Starting gradient (mT/m)
    # Ge: Ending gradient (mT/m)
    # M: Desired moment (s.mT/m)
    # T: Total rewinder time (s)

    # sys: struct with following fields;
    #    area_tol: Moment error tolerance (MSE sense)
    #    max_grad: Max gradient amplitude (mT/m)
    #    max_slew: Max gradient slew rate (T/m/s)

    # convert grad to Hz/m and slew to Hz/m/s
    grad_scale = 42.58e3 # from mT/m -> Hz/m
    slew_scale = 42.58e6 # from T/m/s -> Hz/m/s
    Gs = Gs * grad_scale
    Ge = Ge * grad_scale
    M = M * grad_scale

    max_grad = sys['max_grad'] * grad_scale
    max_slew = sys['max_slew'] * slew_scale

    if 'area_tol' in sys:
        area_tol = sys['area_tol']  # (MSE)
    else:
        area_tol = 1e-4

    SR = max_slew * 0.99  # otherwise we run into rounding errors during resampling
    dT = sys['Tdwell']

    # Set up the optimization.
    gscale = 1.0 / (max_grad)

    # Determine if a triangle should suffice.
    lb = -max_grad * gscale
    ub = max_grad * gscale

    # Iterate flat top time upper bound till we find a reasonable tradeoff between duration and precision.
    obj2 = lambda x: (M - trap_moment_exact_time(x / gscale, T, SR, dT, Gs, Ge))**2
    result = minimize_scalar(obj2, bounds=(lb, ub), method='bounded', options = {'disp': True, 'maxiter': 1000})
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
