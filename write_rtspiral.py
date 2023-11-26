from sigpy.mri import spiral
import numpy as np
import matplotlib.pyplot as plt

Nint = 30
spw = spiral(0.5, 256, 1, 1, Nint, 1, 25e-3, 120e-3, 2*np.pi*42.58e6)
Nwf = spw.shape[0]
Nro = int(Nwf/Nint)
spw1 = spw[0:Nro,:]
fix, axs = plt.subplots(2, 1)
axs[0].plot(spw1[:, 0], spw1[:, 1], 'o')
axs[0].axis('equal')
axs[1].plot(np.diff(np.concatenate(([[0, 0]], spw1), axis=0),1,0))
plt.show()