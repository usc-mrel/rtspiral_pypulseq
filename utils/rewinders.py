#%%
import numpy as np
import matplotlib.pyplot as plt
from pypulseq import Opts
from pypulseq import (make_adc, make_sinc_pulse, make_digital_output_pulse, make_delay, 
                      make_arbitrary_grad, make_trapezoid, make_extended_trapezoid_area, 
                      calc_duration, calc_rf_center, 
                      rotate, add_gradients, make_label)
from pypulseq.Sequence.sequence import Sequence
from utils import schedule_FA, load_params
from utils.traj_utils import save_metadata
from libspiral import vds_fixed_ro, plotgradinfo, raster_to_grad, spiralgen_design
from libspiralutils import pts_to_waveform, design_rewinder_exact_time, round_up_to_GRT
from kernels.kernel_handle_preparations import kernel_handle_preparations, kernel_handle_end_preparations
from math import ceil
import copy
import argparse
import os
import warnings

def rotate_z(grad, angle):
    # grad is a 2D array with shape (n, 2)
    # angle is in radians
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # return np.dot(grad, rot_mat)
    return grad @ rot_mat


def spiral_rewinder_m1_nayak(g_grad, GRT, system, slew_ratio=1):
    # following Spiral Balanced Steady-State Free Precession Cardiac Imaging by Nayak et al. 2005
    from scipy.optimize import fsolve

    # first, rotate the gradients such that g_grad[-1, 1] = 0
    angle = np.arctan2(g_grad[-1, 1], g_grad[-1, 0])
    g_grad = rotate_z(g_grad, angle)

    M = np.cumsum(g_grad, axis=0) * (GRT) # convert to ms
    t = (np.arange(0, g_grad.shape[0]))
    tt = (t*np.ones((2, 1))).T
    M1 = np.cumsum(g_grad * tt,  axis=0) * (GRT**2) # convert to ms^2

    def equation_gradx(vars, M0, M1, Tx1, S):
        Tx2, Tx3 = vars
        f1 = ((1/2)*(Tx1**2)) - (Tx2**2) + (Tx3**2) + (M0/S)
        f2 = ((1/6)*(Tx1**3)) - ((Tx2**2)*(Tx1+Tx2)) + (Tx3**2)*(Tx1+(2*Tx2)+Tx3) + (M1/S)
        return [f1, f2]

    def equation_grady(vars, M0, M1, S):
        Ty1, Ty2 = vars
        f1 = (Ty1**2) - (Ty2**2) + (M0/S)
        f2 = (Ty1**3) - (2*Ty1*(Ty2**2)) - (Ty2**3) + (M1/S)
        return [f1, f2]
    
    def compute_m0(g_grad):
        return np.cumsum(g_grad, axis=0) * (GRT) # convert to ms
    
    def compute_m1(g_grad):
        t_  = (np.arange(0, g_grad.shape[0]))
        return np.cumsum(g_grad * t_,  axis=0) * (GRT**2)
    mamp_dir = lambda g_int, time, direction: g_int - (direction * S * time)

    S = 1e3 * (system.max_slew / system.gamma) * slew_ratio # T/m/s -> mT ms /m
    tx1 = np.abs(g_grad[-1,0] / S)
    Mx0 = M[-1, 0] # mT ms / m
    Mx1 = M1[-1, 0] # mT ms^2 / m

    tx2_best, tx3_best = None, None
    costx = 1e6
    g_rewind_x_best = None
    for _ in range(300):
        [tx2, tx3] = fsolve(equation_gradx, x0=[np.random.rand(1)*1e-3, np.random.rand(1)*1e-3], args=(Mx0,Mx1,tx1,S), xtol=1e-6)
        if tx2 > 0 and tx3 > 0:
            # construct times and amplitudes for grad x.
            times_x = np.array([0, tx1+tx2, tx1 + (2*tx2) + tx3, tx1 + (2*tx2) + (2*tx3)])
            times_x = round_up_to_GRT(times_x, GRT)
            amplitudes_x = np.zeros(4)
            amplitudes_x[0] = g_grad[-1,0]
            amplitudes_x[1] = mamp_dir(g_grad[-1,0], tx1+tx2, np.sign(g_grad[-1,0]))
            amplitudes_x[2] = mamp_dir(amplitudes_x[1], tx2+tx3, -1*np.sign(g_grad[-1,0]))
            amplitudes_x[3] = 0 # mamp(amplitudes_x[2], tx3)

            g_rewind_x = pts_to_waveform(times_x, amplitudes_x, GRT)
            #costx_test = compute_m0(np.concatenate((g_grad[:,0], g_rewind_x)))[-1] #+ compute_m1(np.concatenate((g_grad[:,0], g_rewind_x)))[-1]
            break
            """
            if costx_test < costx:
                print(costx_test)
                g_rewind_x_best = g_rewind_x
                costx = costx_test
                tx2_best = tx2
                tx3_best = tx3
            """
    
            
    #print(f"tx1: {tx1*1e3}: tx2 best: {tx2_best*1e3}, tx3 best: {tx3_best*1e3}")
    #g_rewind_x = g_rewind_x_best

    My0 = M[-1, 1]
    My1 = M1[-1, 1]

    g_rewind_y_best = None
    ty1_best, ty2_best = None, None
    costy = 1e6
    for _ in range(300):
        [ty1, ty2] = fsolve(equation_grady, x0=[np.random.rand(1)*1e-3, np.random.rand(1)*1e-3], args=(My0,My1,S), xtol=1e-6)
        if ty1 > 0 and ty2 > 0:
            # construct times and amplitudes for grad y.
            times_y = np.array([0, ty1, (2*ty1) + ty2, (2*ty1) + (2*ty2)])
            amplitudes_y = np.zeros(4)
            amplitudes_y[0] = g_grad[-1,1]
            amplitudes_y[1] = mamp_dir(amplitudes_y[0], ty1, np.sign(g_grad[-2,1] - g_grad[-1,1]))
            amplitudes_y[2] = mamp_dir(amplitudes_y[1], ty1+ty2, -1 *np.sign(g_grad[-2,1] - g_grad[-1,1]))
            amplitudes_y[3] = 0
            g_rewind_y = pts_to_waveform(times_y, amplitudes_y, GRT)

            costy = compute_m1(np.concatenate((g_grad[:,1], g_rewind_y)))[-1]

            break

            costy_test = compute_m0(np.concatenate((g_grad[:,1], g_rewind_y)))[-1] #+ compute_m1(np.concatenate((g_grad[:,1], g_rewind_y)))[-1]
            #costy_test = compute_m1(np.concatenate((g_grad[:,1], g_rewind_y)))[-1]

            if (costy_test  < costy):
                print(costy_test)
                g_rewind_y_best = g_rewind_y
                costy = costy_test
                ty1_best = ty1
                ty2_best = ty2
    
    # let's try to tweak the times of the last points a little bit.
    tweak_amount = 0.1 # ms
    for _ in range(300):
        tweak_y = (np.random.rand(1)-0.5)*tweak_amount
        times_y [3] = times_y[3] + tweak_y
        g_rewind_y_test = pts_to_waveform(times_y, amplitudes_y, GRT)
        costy_test = compute_m1(np.concatenate((g_grad[:,1], g_rewind_y)))[-1]
        if costy_test < costy:
            g_rewind_y = g_rewind_y_test

    #g_rewind_y = g_rewind_y_best
    #ty1_best, ty2_best = ty1, ty2
    #print(f"ty1 best: {ty1_best*1e3}, ty2 best: {ty2_best*1e3}")

    if len(g_rewind_x) > len(g_rewind_y):
        g_rewind_y = np.concatenate((g_rewind_y, np.zeros(len(g_rewind_x) - len(g_rewind_y))))
    else:
        g_rewind_x = np.concatenate((g_rewind_x, np.zeros(len(g_rewind_y) - len(g_rewind_x))))

    # TEST HACK
    g_grad_plot = np.concatenate((g_grad, np.stack([g_rewind_x[0:], g_rewind_y[0:]]).T))
    plotgradinfo(g_grad_plot, GRT)


    # rotate back (not necessary, but nice for comparison to other methods)
    g_rewind = np.column_stack((g_rewind_x, g_rewind_y))
    g_rewind = rotate_z(g_rewind, -angle)
    g_grad = rotate_z(g_grad, -angle)

    g_rewind_x = g_rewind[:, 0]
    g_rewind_y = g_rewind[:, 1]

    return g_grad, g_rewind_x, g_rewind_y