from _vds.lib import calc_vds
from _vds import ffi

import numpy as np
from math import exp, log, ceil


def vds_design(sys: dict, Nint: int, fov: list, res: float, Tread: float) -> np.array:


    slewmax = sys['max_slew']*100
    gradmax = sys['max_grad']/10
    Td = sys['Tdwell']
    os = sys['os']

    Tg = Td/os;	# gradient rate.

    krmax = 5/res
    numfov = fov.__len__()
    ngmax = ceil(Tread/Td)

    fov_i = ffi.new("double*", numfov)

    fovs = 25.6*exp(log(1/krmax)*np.arange(0, numfov, 1))
    fov_i[0] = fovs

    # Output variables
    xgrad_o = ffi.new("double**")
    ygrad_o = ffi.new("double**")
    numgrad = ffi.new("int*")

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
    calc_vds(slewmax, gradmax, Tg, Td, Nint, fov_i, numfov, krmax, ngmax, xgrad_o, ygrad_o, numgrad)

    Ngrad = numgrad[0]
    Gx = np.frombuffer(ffi.buffer(xgrad_o[0], 8*Ngrad), dtype=np.double)
    Gy = np.frombuffer(ffi.buffer(ygrad_o[0], 8*Ngrad), dtype=np.double)

    Gx = Gx[0:-1:os]
    Gy = Gy[0:-1:os]

    g = np.column_stack([Gx,Gy])*10 # [G/cm] -> [mT/m]

    g = np.concatenate((np.zeros((1, 2)), g))
    return g


def vds_fixed_ro(sys: dict, fov: list, res: float, Tread: float) -> tuple[np.array, int]:
    vds_design(sys, Nint, fov, res, Tread)




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sys = {
        'max_slew':  170,  # [mT/m/s] 
        'max_grad':   38,  # [mT/m] 
        'Tdwell'  : 1e-6, # [s]
        'os'      :    8
        }

    Nint = 19
    fov  = [25.6] # [cm]
    res = 2 # [mm]
    Tread = 3e-3

    g = vds_design(sys, Nint, fov, res, Tread)

    plt.plot(g)
    plt.show()
    pass