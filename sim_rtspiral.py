# Adapted from MRzero-core playground
import MRzeroCore as mr0
import matplotlib.pyplot as plt
import util
import torch
import numpy as np

noiselevel = 1e-4
verbose = 1
Nread = 64    # frequency encoding steps/samples
Nphase = 64    # phase encoding steps/samples

fname = 'spiral_bssfp_debug.seq'
seq0 = mr0.Sequence.from_seq_file('out_seq/' + fname).cuda()

# if verbose > 0:
    # seq0.plot_kspace_trajectory()


sz = [64, 64]
# (i) load a phantom object from file
obj_p = mr0.VoxelGridPhantom.load_mat('sim_data/numerical_brain_cropped.mat')
obj_p = obj_p.interpolate(sz[0], sz[1], 1)
# Manipulate loaded data
obj_p.T2dash[:] = 30e-3
obj_p.D *= 0
obj_p.B0 *= 1    # alter the B0 inhomogeneity
# Store PD and B0 for comparison
PD = obj_p.PD
B0 = obj_p.B0
obj_p.plot()
# Convert Phantom into simulation data
obj_p = obj_p.build().cuda()


# Simulate the sequence
graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
signal = mr0.execute_graph(graph, seq0, obj_p).cpu()


# additional noise as simulation is perfect
signal += noiselevel * np.random.randn(signal.shape[0], 2).view(np.complex128)

# S6: MR IMAGE RECON of signal ::: #####################################
if verbose > 0:
    fig = plt.figure()  # fig.clf()
    plt.subplot(411)
    plt.title('ADC signal')
    plt.plot(torch.real(signal), label='real')
    plt.plot(torch.imag(signal), label='imag')
    # this adds ticks at the correct position szread
    major_ticks = np.arange(0, Nphase * Nread, Nread)
    ax = plt.gca()
    ax.set_xticks(major_ticks)
    ax.grid()
    plt.show()
