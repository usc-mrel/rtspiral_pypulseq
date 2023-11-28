import numpy as np

def trap_moment_exact_time(Gp, T, SR, dT, Gs, Ge):
    Tru = np.ceil(abs(Gp - Gs) / SR / dT) * dT
    Trd = np.ceil(abs(Gp - Ge) / SR / dT) * dT
    Tp = np.round((T - Tru - Trd)/ dT) * dT
    ga = trap_moment(Gp, Tru, Tp, Trd, Gs, Ge)
    return ga

def trap_moment(Gp, Tr, Tp, Tf, Gs, Ge):
    # Calculates trapezoid moment given start and end gradient amplitude,
    # rise and fall time, flat top time, and flat top amplitude.
    return 0.5 * Tr * (Gp + Gs) + Gp * Tp + 0.5 * (Gp + Ge) * Tf


def trap_moment_lims(Gp, Tp, SR, dT, Gs, Ge):
    Tru = np.ceil(abs(Gp - Gs) / SR / dT) * dT
    Trd = np.ceil(abs(Gp - Ge) / SR / dT) * dT
    Tp = np.round(Tp / dT) * dT
    return trap_moment(Gp, Tru, Tp, Trd, Gs, Ge)


