
from _vds.lib import calc_vds
from _vds import ffi
import numpy as np
import numpy.typing as npt
from math import exp, log, ceil, sqrt

def calcgradinfo(g: npt.ArrayLike, T: float = 4e-6, k0: float = 0, R: float = 0.35, L: float = 0.0014, eta: float = 1.7857e-4) -> \
tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """	Function calculates gradient information

	Parameters
    ----------
    g : ArrayLike
        gradient (mT/m) = [gx, gy]
    T : float
        sample period (s)
    k0 : float
        initial condition for k-space.
    R : float
        coil resistance (ohms, default =.35)
    L : float
        coil inductance (H, default = .0014)
    eta : float
        coil efficiency (G/cm/A, default = 1/56)

	Returns
    -------
    k : ArrayLike
        k-space trajectory (m^(-1))
    g : ArrayLike
        gradient (mT/m)
    s : ArrayLike
        slew rate trajectory (T/m/s)
    m1 : ArrayLike
        first moment trajectory (s/m)
    m2 : ArrayLike
        second moment trajectory (s^2/m)
    t : ArrayLike
        vector of time points (s)
    v : ArrayLike
        voltage across coil.


	B.Hargreaves, Aug 2002.
    Adopted to Python by: Bilal Tasdelen 2023
    """

    gamma = 42.58e6 # [Hz/T]
    gT = g*1e-3 # [mT/m] -> [T/m]
    k = k0 + np.cumsum(gT, axis=0)*gamma*T
    t = (np.arange(1, g.shape[0]+1)-0.5)*T
    tt = (t*np.ones((2, 1))).T
    s = np.concatenate((g, np.reshape(g[-1,:], (1, 2)))) - np.concatenate(([[0, 0]], g))
    s = s[1:,:]/T/1e3 # [mT/m/s] -> [T/m/s]
    m1 = np.cumsum(gT*tt, axis=0)*gamma*T
    m2 = np.cumsum(gT*(tt*tt+T**2/12), axis=0)*gamma*T
    v = (1/eta)*(L*s+R*gT)
    return k, g, s, m1, m2, t, v

def plotgradinfo():

    return

def vds_design(sys: dict, Nint: int, fov: list, res: float, Tread: float) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """ Given the system parameters, number of interleaves, FoV, resolution and Readout time, designs the variable density spiral.

    Parameters
    ----------
    sys : dict
        Dictionary of given system parameters:
        max_grad = Available gradient strength [mT/m]
        max_slew = Available slew rate [T/m/s]
        Tdwell = Sample (gradient) dwell time [s]
        os = How much waveform will be oversampled during the design
    Nint : int
        Number of interleaves
    fov : list
        List of FoVs to be supported [cm].
    res : float
        Resolution target [mm].
    Tread : float
        Maximum readout length [s].
    Returns
    -------
    k : ArrayLike
        Designed trajectory [mm^-1].
    g : ArrayLike
        Designed grad waveform [mT/m].
    s : ArrayLike
        Slew rate of the trajectory [T/m/s].
    time : ArrayLike
        Time axis [s].
    """

    slewmax = sys['max_slew']*100
    gradmax = sys['max_grad']/10
    Td = sys['Tdwell']
    oversamp = sys['os']

    Tg = Td/oversamp;	# gradient rate.

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
    calc_vds(slewmax, gradmax, Tg, Td, Nint, fov_i, numfov, krmax, ngmax*oversamp, xgrad_o, ygrad_o, numgrad)

    Ngrad = numgrad[0]
    Gx = np.frombuffer(ffi.buffer(xgrad_o[0], 8*Ngrad), dtype=np.double)
    Gy = np.frombuffer(ffi.buffer(ygrad_o[0], 8*Ngrad), dtype=np.double)

    Gx = Gx[0:-1:oversamp]
    Gy = Gy[0:-1:oversamp]

    g = np.column_stack([Gx,Gy])*10 # [G/cm] -> [mT/m]

    # g = np.concatenate((np.zeros((1, 2)), g))
    k, g, s, m1, m2, time, v = calcgradinfo(g, Td)

    return k, g, s, time

def vds_fixed_ro(sys: dict, fov: list, res: float, Tread: float) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, int]:
    """vds_fixed_ro Sweeps vdsmex until N interleaves found that satisfies desired res.

    Parameters
    ----------
    sys : dict
        Contains the system specs for the design, 'max_slew', 'max_grad', 'Tdwell', 'os'
    FOV : list
        Desired Field of View(s) [cm]
    res : float
        Desired resolution [mm]
    Tread : float 
        Desired readout length [s] (RO Duration = Npoints*Ts)

    Returns
    -------
    k : ArrayLike
        Designed k-space trajectory [m^-1]
    g : ArrayLike
        Designed gradient waveform [mT/m]
    time : ArrayLike
        Time axis [s]
    nint : int
        Number of interleaves that supports unalised FoV with the desired resolution.
    """

    tol = 0.99
    krmax_target = 1/(2*res*1e-3) # m^-1
    krmax = 0
    nint = 0
    while krmax < krmax_target*tol:
        nint = nint+1
        k,g,_,time = vds_design(sys, nint, fov, res, Tread)
        krmax = sqrt(k[-1,0]*k[-1,0] + k[-1,1]*k[-1,1])
    
    return k, g, time, nint



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sys = {
        'max_slew':  170,  # [T/m/s] 
        'max_grad':   38,  # [mT/m] 
        'Tdwell'  : 1e-6, # [s]
        'os'      :    8
        }

    # Test the vds_design function directly
    Nint = 19
    fov  = [25.6] # [cm]
    res = 2 # [mm]
    Tread = 3e-3

    k, g, s, t = vds_design(sys, Nint, fov, res, Tread)

    plt.figure()
    plt.plot(t*1e3, g)
    plt.xlabel('Time [ms]')
    plt.ylabel('Grad Amp [mT/m]')
    plt.title('VDS Design output')

    # Test the fixed ro function

    k2, g2, t2, nint2 = vds_fixed_ro(sys, fov, res, Tread)

    plt.figure()
    plt.plot(t2*1e3, g2)
    plt.xlabel('Time [ms]')
    plt.ylabel('Grad Amp [mT/m]')
    plt.title('VDS fixed ro output')

    plt.show()


    pass