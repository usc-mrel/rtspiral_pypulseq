from scipy.io import savemat
from scipy.signal import medfilt
from sigpy.mri.dcf import pipe_menon_dcf
import numpy as np
import os

def save_traj_dcf(filename, k_traj_adc, n_TRs, fov, res, ndiscard, show_plots=True):
    Nsample = int(k_traj_adc.shape[1]/n_TRs)
    Nsample2 = Nsample-ndiscard
    kx = k_traj_adc[0,:]
    ky = k_traj_adc[1,:]
    kx = np.reshape(kx, (Nsample, -1))
    ky = np.reshape(ky, (Nsample, -1))
    kx = kx[ndiscard:,:]
    ky = ky[ndiscard:,:]
    k = np.concatenate((np.reshape(kx, ((Nsample2)*n_TRs, 1)), np.reshape(ky, ((Nsample2)*n_TRs, 1))), axis=1)
    # k_max = np.max(np.abs(kx + 1j * ky))
    # k = (kx / k_max) + (1j * ky / k_max)

    # calculate density compensation weights using Pipe and Menon's method
    # w = pipe_menon_dcf(k_traj_adc[0:2, :].T, max_iter=30)
    w = pipe_menon_dcf(k, max_iter=30)

    w = w[Nsample2+1:2*Nsample2+1]
    w = w / (np.max(w))
    w[w > 0.4] = 0.4
    w = w / np.max(w)
    w[int(w.shape[0]*2/3):w.shape[0]] = 1
    w = medfilt(w, 11)

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
        'spatial_resolution': res,
        'repetitions': n_TRs,
        'matrix_size': [fov[0]*10/res, fov[0]*10/res],
        'pre_discard': ndiscard
    }

    traj_path = os.path.join('out_trajectory', f'{filename}.mat')
    savemat(traj_path, {'kx': kx, 'ky': ky, 'w' : w, 'param': meta})

