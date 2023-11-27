
from _vds.lib import calc_vds
from _vds import ffi
import numpy as np
import numpy.typing as npt
from math import exp, log, ceil

def calcgradinfo(g: npt.ArrayLike, T: float = 4e-6, k0: float = 0, R: float = 0.35, L: float = 0.0014, eta: float = 17.86e-3):
    '''	Function calculates gradient information

	INPUT:
		g	gradient (G/cm) = gx + i*gy
		T	sample period (s)
		k0	initial condition for k-space.
		R	coil resistance (ohms, default =.35)
		L	coil inductance (H, default = .0014)
		eta	coil efficiency (G/cm/A, default = 1/56)

	OUTPUT:
		k	k-space trajectory (cm^(-1))
		g	gradient (G/cm)
		s	slew rate trajectory (G/cm/s)
		m1	first moment trajectory (s/cm)
		m2	second moment trajectory (s^2/cm)
		t	vector of time points (s)
		v	voltage across coil.


	B.Hargreaves, Aug 2002.
    Adopted to Python by: Bilal Tasdelen 2023'''

    gamma = 4258
    sz = g.shape
    lg = max(sz)
    k = k0 + np.cumsum(g)*gamma*T
    t = (np.arange(1, lg+1)-0.5)*T
    tt = t*np.ones((lg, 1))
    s = np.concatenate((g, np.reshape(g[-1,:], (1, 2)))) - np.concatenate(([[0, 0]], g))
    s = s[1:,:]/T
    m1 = np.cumsum(g.T.dot(tt), axis=1)*gamma*T
    m2 = np.cumsum(g.T.dot(tt.dot(tt)+T**2/12))*gamma*T
    v = (1/eta)*(L*s+R*g)
    return k, g, s, m1, m2, t, v

def vds_design(sys: dict, Nint: int, fov: list, res: float, Tread: float) -> npt.ArrayLike:


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
    k, g, s, m1, m2, time, v = calcgradinfo(g, Td)

    return k, g, s, time

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
    res = 4 # [mm]
    Tread = 3e-3

    k, g, s, t = vds_design(sys, Nint, fov, res, Tread)

    plt.plot(t*1e3, g)
    plt.show()
    pass