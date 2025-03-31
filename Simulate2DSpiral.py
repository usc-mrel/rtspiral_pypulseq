# %%
import os
import MRzeroCore as mr0
import torch
import numpy as np
import matplotlib.pyplot as plt
import pypulseq as pp
from scipy.io import loadmat
from PySide6.QtWidgets import QApplication, QFileDialog
import sys

def get_filepath(dir: str=''):
    '''QT based file selection UI.'''
    app = QApplication(sys.argv)
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_path, _ = QFileDialog.getOpenFileName(None, "Select an Pulseq seq file", dir, "Seq files (*.seq);;All Files (*)", options=options)
    app.shutdown()
    return file_path

# %%
# seq_path = os.path.join('out_seq/spiral_FLASH_linear16.3636_nTR110_Tread3.00_TR5.05ms_FA30.seq')
seq_path = get_filepath('out_seq')

seqp = pp.Sequence()
seqp.read(seq_path)
print('Sequence signature: ' + seqp.signature_value)
traj_name = seqp.signature_value
# Load the metadata
traj = loadmat("out_trajectory/" + traj_name)

sz = np.round(traj['param']['matrix_size'][0,0][0]).astype(int)    # spin system size / resolution
# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above

phantom_fn = 'brainweb/subject04_3T.npz'
obj_p = mr0.VoxelGridPhantom.brainweb(phantom_fn)
obj_p = obj_p.slices([60])
obj_p = obj_p.interpolate(sz[0], sz[1], 1)
# Manipulate loaded data
obj_p.T2dash[:] = 30e-3
obj_p.D *= 0
obj_p.B0 *= 0

obj_p.plot()
# Convert Phantom into simulation data
obj_p = obj_p.build()


# %%
seq0 = mr0.Sequence.import_file(seq_path)

seq0.plot_kspace_trajectory()
kspace_loc = seq0.get_kspace()
# Simulate the sequence
graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
signal = mr0.execute_graph(graph, seq0.cuda(), obj_p.cuda(), print_progress=True).cpu()


# %% Trajectory goes to torch

n_unique_angles = traj['param']['interleaves'][0,0][0,0]

kx = traj['kx'][:,:]
ky = traj['ky'][:,:]
pre_discard = traj['param']['pre_discard'][0,0][0,0]


Nread = kx.shape[0]
Nphase = kx.shape[1]
# PLOT sequence with signal in the ADC subplot
# sp_adc, t_adc = mr0.util.pulseq_plot(seqp, clear=True, signal=signal.numpy())

kspace_adc = torch.reshape((signal), (Nphase, Nread + pre_discard)).clone().t()
kspace_loc = torch.reshape(kspace_loc, (Nphase, Nread + pre_discard, 4)).clone()

# %%

ktraj = np.stack((kx, -ky), axis=2)

# find max ktraj value
kmax = np.max(np.abs(kx + 1j * ky))

# swap 0 and 1 axes to make repetitions the first axis (repetitions, interleaves, 2)
ktraj = np.swapaxes(ktraj, 0, 1)

msize = np.int16(10 * traj['param']['fov'][0,0][0,0] / traj['param']['spatial_resolution'][0,0][0,0])

ktraj = 0.5 * (ktraj / kmax) * msize

w = traj['w']
w = np.reshape(w, (1,w.shape[1]))

# %%
import sigpy as sp
from sigpy import fourier

nchannel = 1
# Reconstruct
gpu_device = -1
device = sp.Device(gpu_device)

coord_gpu = sp.to_device(ktraj, device=device)
w_gpu = sp.to_device(w, device=device)

frames = []
images = []
for arm_counter, arm in enumerate(kspace_adc.T[:,:]):
    adata = sp.to_device(arm[pre_discard:, None].T, device=device).numpy()
    frames.append(fourier.nufft_adjoint(
            adata*w_gpu,
            coord_gpu[arm_counter%n_unique_angles,:,:],
            (nchannel, msize, msize)))
    if ((arm_counter+1) % n_unique_angles == 0) and (arm_counter > 0):
        image = np.sum([sp.to_device(frame) for frame in frames], axis=0).squeeze()
        images.append(image)
        frames = []

# image = np.sum([sp.to_device(frame) for frame in frames], axis=0).squeeze()

plt.figure()
plt.imshow(np.abs(images[-1]).T, cmap='gray')

# %%
f, axs = plt.subplots(1,3)
axs[0].imshow(np.abs(images[-1]).T, cmap='gray')
axs[0].set_axis_off()
axs[1].imshow(np.abs(images[-2]).T, cmap='gray')
axs[1].set_axis_off()
axs[2].imshow(np.abs(images[0]).T, cmap='gray')
axs[2].set_axis_off()


# %%
