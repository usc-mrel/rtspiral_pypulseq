from _vds.lib import calc_vds
from _vds import ffi

import numpy as np
from math import exp, log, ceil
import matplotlib.pyplot as plt

# double slewmax;		/*	Maximum slew rate, G/cm/s		*/
# double gradmax;		/* 	maximum gradient amplitude, G/cm	*/
# double Tgsample;	/*	Gradient Sample period (s)		*/
# double Tdsample;	/*	Data Sample period (s)			*/
# int Ninterleaves;	/*	Number of interleaves			*/
# double *fov;		/*	FOV coefficients		*/
# int numfov;		/*	Number of FOV coefficients		*/
# double krmax;		/*	Maximum k-space extent (/cm)		*/
# int ngmax;		/*	Maximum number of gradient samples	*/
# double **xgrad;		/* 	[output] X-component of gradient (G/cm) */
# double **ygrad;		/*	[output] Y-component of gradient (G/cm)	*/
# int *numgrad;		/* 	[output] Number of gradient samples */

res = 2 #mm
Tread = 5e-3

slewmax = 17000.0
gradmax = 3.8

oversamp = 8;		# Factor by which to oversample gradient.
Tg = 1e-6/oversamp;	# gradient rate.
Td = 1e-6;			# data sampling rate.

Nint = 19

numfov = 1
fov = ffi.new("double*", numfov)
krmax = 5/res
fovs = 25.6*exp(log(1/krmax)*np.arange(0, numfov, 1))
fov[0] = fovs
ngmax = ceil(Tread/Td)

xgrad_o = ffi.new("double**")
ygrad_o = ffi.new("double**")
numgrad = ffi.new("int*")

calc_vds(slewmax, gradmax, Tg, Td, Nint, fov, numfov, krmax, ngmax, xgrad_o, ygrad_o, numgrad)

Ngrad = numgrad[0]
Gx = np.frombuffer(ffi.buffer(xgrad_o[0], 8*Ngrad), dtype=np.double)
Gy = np.frombuffer(ffi.buffer(ygrad_o[0], 8*Ngrad), dtype=np.double)

Gx = Gx[0:-1:oversamp]
Gy = Gy[0:-1:oversamp]

plt.plot(Gx)
plt.plot(Gy)
plt.show()
pass