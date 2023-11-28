import numpy as np
from scipy.optimize import minimize_scalar, minimize, NonlinearConstraint

from .rounding import round_to_GRT, round_up_to_GRT
from .trap_moments import trap_moment, trap_moment_lims


def design_rewinder_time_optimal(Gs, Ge, M, sys):
# PK: work in progress. This function is not working yet.

#DESIGNREWINDER Given moment, start, end gradients and system limits, design a trapezoid.
# Gs: Starting gradient (mT/m)
# Ge: Ending gradient (mT/m)
# M: Desired moment (s.mT/m)
# sys: struct with following fields;
#    .area_tol: Moment error tolerance (MSE sense)
#    .Tpud: Upper bound for flat top duration (s)
#    .maxGrad: Max gradient amplitude (mT/m)
#    .maxSlew: Max gradient slew rate (T/m/s)

    # Given moment, start, end gradients, and system limits, design a trapezoid.
    if 'area_tol' in sys:
        area_tol = sys['area_tol']  # (MSE)
    else:
        area_tol = 1e-4

    if 'max_iter' in sys:
        max_iter = sys['max_iter']
    else:
        max_iter = 100

    Tpud = sys['Tpud']  # (s)

    grad_scale = 42.58e3 # from mT/m -> Hz/m
    slew_scale = 42.58e6 # from T/m/s -> Hz/m/s
    Gs = Gs * grad_scale
    Ge = Ge * grad_scale
    M = M * grad_scale

    max_grad = sys['max_grad'] * grad_scale
    max_slew = sys['max_slew'] * slew_scale

    SR = max_slew * 0.99
    dT = sys['adc_dwell']

    tscale = 1
    gscale = 1. / max_grad

    gM = trap_moment_lims(np.sign(M) * max_grad, 0, SR, dT, Gs, Ge)
    step2 = False
    g0 = np.sign(M) * max_grad
    glb = -max_grad
    gub = max_grad

    if abs(gM) >= abs(M):
        # It is likely that a triangle is sufficient, search only for Gp
        options = {'disp': False}
        xtol = 1e-7
        obj1 = lambda x: (M - trap_moment_lims(x, 0, SR, dT, Gs, Ge))**2
        result = minimize_scalar(obj1, bounds=(glb, gub), tol=xtol, options=options)
        x1 = result.x

        if result.fun > area_tol:
            step2 = True
            g0 = x1

    if abs(gM) < abs(M) or step2:
        obj2 = lambda x: abs(x[1])

        bound_grad = [-max_grad * gscale, max_grad * gscale]
        bound_time = [dT * tscale, Tpud * tscale]

        options = {'disp': False}
        xtol = 1e-7
        Tp0 = round_to_GRT((M - trap_moment_lims(g0, 0, SR, dT, Gs, Ge)) / g0, GRT=dT)

        while Tp0 <= 0:
            g0 = g0 * 0.98
            Tp0 = round_to_GRT((M - trap_moment_lims(g0, 0, SR, dT, Gs, Ge)) / g0, GRT=dT)

        eps = (M - trap_moment_lims(g0, Tp0, SR, dT, Gs, Ge))**2

        x0 = [g0 * gscale, Tp0 * tscale]
        x = x0


        niter = 0
        while eps > area_tol:
            niter += 1
            mc = lambda x: mconst([x[0] / gscale, x[1] / tscale], M, SR, dT, Gs, Ge, 0)
            nlc = NonlinearConstraint(mc, -xtol, xtol)

            minimization = minimize(obj2, x0, bounds=tuple([bound_grad, bound_time]), constraints=[nlc], tol=xtol, options=options)
            x = minimization.x

            if minimization.success == False:
                # We are in an infeasible region. Let's try resetting the problem.
                x[0] = x[0] / 2  # Half the gradient, hopefully enough to reset.
                x[1] = round_to_GRT((M - trap_moment_lims(x[0] / gscale, 0, SR, dT, Gs, Ge)) / (x[0] / gscale), GRT=dT) * tscale

            eps = (M - trap_moment_lims(x[0] / gscale, x[1] / tscale, SR, dT, Gs, Ge))**2
            x0 = x

            if niter == max_iter:
                raise ValueError(f'The optimization did not converge after {niter} iterations. '
                                 f'Try increasing the Tpud or area_tol.')

        Gp = x[0] / gscale
        Tp = round_to_GRT(x[1] / tscale)
        Tr = round_up_to_GRT(abs(Gp - Gs) / SR, GRT=dT)
        Tf = round_up_to_GRT(abs(Gp - Ge) / SR, GRT=dT)
    else:
        # step 1 was successful
        Gp = x1
        Tp = 0
        Tr = round_up_to_GRT(abs(Gp - Gs) / SR, GRT=dT)
        Tf = round_up_to_GRT(abs(Gp - Ge) / SR, GRT=dT)

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

def mconst(x, Md, SR, dT, Gs, Ge, eps):
    Mrw = trap_moment_lims(x[0], x[1], SR, dT, Gs, Ge)
    return (Md - Mrw)**2 - eps

