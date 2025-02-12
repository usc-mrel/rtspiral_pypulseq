from typing import Literal
from scipy.io import savemat
from scipy.signal import medfilt
from sigpy.mri.dcf import pipe_menon_dcf
import numpy as np
import numpy.typing as npt
import os

def save_metadata(filename: str, k_traj_adc: npt.ArrayLike, params: dict, show_plots=True, dcf_method: Literal["hoge", "pipe_menon"]="hoge", out_dir:str ="out_trajectory"):
    '''Save the trajectory and metadata to a .mat file.

    Parameters
    ----------
    filename : str
        Filename of the .mat file to save. Typically the sequence name.
    k_traj_adc : np.ndarray
        k-space trajectory in the form of a 3xN array. Unit is 1/m.
    params : dict
        Dictionary containing the following
        - n_TRs : int
            Number of repetitions.
        - n_int : int
            Number of interleaves.
        - ga_rotation : float
            Golden angle or linear rotation in degrees.
        - fov : tuple
            Field of view in meters.
        - spatial_resolution : float
            Spatial resolution in meters.
        - arm_ordering : str
            Arm ordering of the trajectory.
        - adc_dwell : float
            ADC dwell time in seconds.
        - ndiscard : int
            Number of samples to discard.
    show_plots : bool, optional
        Whether to show the plots of the DCF.
    dcf_method : Literal["hoge", "pipe_menon"], optional
        Method to use for density compensation function.
        Hoge is analytical, Pipe and Menon is iterative.
    out_dir : str, optional
        Output directory to save the .mat file.
    '''

    adc_dwell = params['adc_dwell']
    ndiscard = params['ndiscard']

    Nsample = int(k_traj_adc.shape[1]/params['n_TRs'])
    kx = k_traj_adc[0,:]
    ky = k_traj_adc[1,:]
    kz = k_traj_adc[2,:]
    kx = np.reshape(kx, (-1, Nsample)).T
    ky = np.reshape(ky, (-1, Nsample)).T
    kz = np.reshape(kz, (-1, Nsample)).T
    kxx = kx[:,0]
    kyy = ky[:,0]
    kx = kx[ndiscard:,:]
    ky = ky[ndiscard:,:]
    kz = kz[ndiscard:,:]

    if dcf_method == "hoge":
        gx = np.diff(np.concatenate(([0], kxx)), axis=0)/adc_dwell/42.58e6
        gy = np.diff(np.concatenate(([0], kyy)), axis=0)/adc_dwell/42.58e6

        # Analytical DCF formula
        # 1. Hoge RD, Kwan RKS, Bruce Pike G. Density compensation functions for spiral MRI. 
        # Magnetic Resonance in Medicine. 1997;38(1):117-128. doi:10.1002/mrm.1910380117

        cosgk = np.cos(np.arctan2(kxx, kyy) - np.arctan2(gx, gy))
        w = np.sqrt(kxx*kxx+kyy*kyy)*np.sqrt(gx*gx+gy*gy)*np.abs(cosgk)
        w = w[ndiscard:]
        w[-int(Nsample//2):] = w[-int(Nsample//2)] # need this to correct weird jump at the end and improve SNR
        w = w/np.max(w)
    elif dcf_method == "pipe_menon":
        Nsample2 = Nsample-ndiscard

        # calculate density compensation weights using Pipe and Menon's method
        w = pipe_menon_dcf(k_traj_adc[0:2, 0:(Nsample2*params['n_int'])].T, max_iter=100)
        w = w[Nsample+1:2*Nsample+1]
        w = w / (np.max(w))
        w[w > 0.1] = 0.1
        w = w / np.max(w)
        w[int(w.shape[0]*2/3):w.shape[0]] = 1
        w = medfilt(w, 11)
        w = w[ndiscard:]

    if show_plots:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(w)
        plt.xlabel('ADC sample')
        plt.ylabel('|w|')
        plt.title('DCF')
        plt.show()

    fov = params['fov']
    res = params['spatial_resolution']
    meta = {
        'fov': fov[0],
        'spatial_resolution': float(res),
        'repetitions': params['n_TRs'],
        'interleaves': params['n_int'],
        'ga_rotation': params['ga_rotation'],
        'matrix_size': [int(round(fov[0]*10/res)), int(round(fov[0]*10/res))],
        'pre_discard': ndiscard,
        'arm_ordering': params['arm_ordering'],
        'dt': adc_dwell
    }

    traj_path = os.path.join(out_dir, f'{filename}.mat')
    os.makedirs(out_dir, exist_ok=True)
    savemat(traj_path, {'kx': kx, 'ky': ky,'kz': kz, 'w' : w, 'param': meta})

def save_3Dtraj(filename, k_traj_adc, n_TRs, n_eco, n_int, ga_rotation, idx, fov, res: float, adc_dwell: float = 1e-6, ndiscard: int = 10, show_plots=True):
    Nsample = int(k_traj_adc.shape[1]/n_TRs/n_eco)
    kx = k_traj_adc[0,:]
    ky = k_traj_adc[1,:]
    kz = k_traj_adc[2,:]
    kx = np.reshape(kx, (-1, Nsample*n_eco)).T[:(Nsample),:]
    ky = np.reshape(ky, (-1, Nsample*n_eco)).T[:(Nsample),:]
    kz = np.reshape(kz, (-1, Nsample*n_eco)).T[:(Nsample),:]
    kxx = kx[:,0]
    kyy = ky[:,0]
    kx = kx[ndiscard:,:]
    ky = ky[ndiscard:,:]
    kz = kz[ndiscard:,:]

    # kx = kx[ndiscard:,0]
    # ky = ky[ndiscard:,0]
    gx = np.diff(np.concatenate(([0], kxx)), axis=0)/adc_dwell/42.58e6
    gy = np.diff(np.concatenate(([0], kyy)), axis=0)/adc_dwell/42.58e6

    # Analytical DCF formula
    # 1. Hoge RD, Kwan RKS, Bruce Pike G. Density compensation functions for spiral MRI. 
    # Magnetic Resonance in Medicine. 1997;38(1):117-128. doi:10.1002/mrm.1910380117

    cosgk = np.cos(np.arctan2(kxx, kyy) - np.arctan2(gx, gy))
    w = np.sqrt(kxx*kxx+kyy*kyy)*np.sqrt(gx*gx+gy*gy)*np.abs(cosgk)
    w = w[ndiscard:]
    w[-int(Nsample//2):] = w[-int(Nsample//2)] # need this to correct weird jump at the end and improve SNR
    w = w/np.max(w)

    if show_plots:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(w)
        plt.xlabel('ADC sample')
        plt.ylabel('|w|')
        plt.title('DCF')
        plt.show()

    meta = {
        'fov': fov[0],
        'spatial_resolution': float(res),
        'repetitions': n_TRs,
        'interleaves': n_int,
        'kspace_step_1': idx['kspace_step_1'],
        'kspace_step_2': idx['kspace_step_2'],
        'contrast':      idx['contrast'],
        'ga_rotation': ga_rotation,
        'matrix_size': [fov[0]*10/res, fov[0]*10/res, 1+np.max(idx['kspace_step_2'])],
        'pre_discard': ndiscard,
        'dt': adc_dwell
    }

    traj_path = os.path.join('out_trajectory', f'{filename}.mat')
    os.makedirs("out_trajectory", exist_ok=True)
    savemat(traj_path, {'kx': kx, 'ky': ky,'kz': kz, 'w' : w, 'param': meta})


def generate_encoding_indices(n_int, n_kz, n_rep=1, n_eco=1, kz_ordering='linear', kspace_ordering='arm'):
    '''Generate encoding indices for a 3D multiecho acquisition.

    '''

    # Generate encoding indices
    n_tr = n_int * n_kz * n_eco * n_rep

    if kspace_ordering == 'arm':
        kspace_step_1 = [ii//n_eco % n_int for ii in range(n_tr)]
        if kz_ordering == 'linear':
            kspace_step_2 = [(ii//n_int//n_eco) % n_kz for ii in range(n_tr)]
        # TODO: Fix ping-pong ordering.
        elif kz_ordering == 'ping-pong':
            kspace_step_2 = []
            c_kz = 0
            c_kz_dir = 1
            for ii in range(n_tr):
                if c_kz % n_kz == 0:
                    c_kz_dir = 1
                elif c_kz % n_kz == n_kz - 1:
                    c_kz_dir = -1
                
                kspace_step_2.append(c_kz)
                if ii % (n_int) == 0 and ii != 0:
                    c_kz += c_kz_dir

    elif kspace_ordering == 'stack':
        kspace_step_1 = [ii // n_kz for ii in range(n_tr)]
        if kz_ordering == 'linear':
            kspace_step_2 = [ii % n_kz for ii in range(n_tr)]
        elif kz_ordering == 'ping-pong':
            kspace_step_2 = []
            for ii in range(n_tr):
                if ii % (2*n_kz) < n_kz:
                    kspace_step_2.append(ii % n_kz)
                else:
                    kspace_step_2.append(n_kz - 1 - ii % n_kz)
    
    contrast = [ii % n_eco for ii in range(n_tr)]

    idx = {
        'kspace_step_1': kspace_step_1,
        'kspace_step_2': kspace_step_2,
        'contrast': contrast
    }
    return idx