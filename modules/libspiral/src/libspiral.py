"""
Python wrapper for the spiralgen and vds functions in the _spiralgen library.
"""
from typing import Literal, Any
from _spiralgen.lib import bnispiralgen, calc_vds
from _spiralgen import ffi
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from math import log, ceil, sqrt

def calcgradinfo(g: npt.NDArray[np.float64], T: float = 4e-6, k0: float = 0, R: float = 0.35, L: float = 0.0014, eta: float = 1.7857e-4) -> \
tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
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
    k : NDArray
        k-space trajectory (m^(-1))
    g : NDArray
        gradient (mT/m)
    s : NDArray
        slew rate trajectory (T/m/s)
    m1 : NDArray
        first moment trajectory (s/m)
    m2 : NDArray
        second moment trajectory (s^2/m)
    t : NDArray
        vector of time points (s)
    v : NDArray
        voltage across coil.


	B.Hargreaves, Aug 2002.
    Adopted to Python by: Bilal Tasdelen 2023
    """

    gamma = 42.58e6 # [Hz/T]
    gT = g*1e-3 # [mT/m] -> [T/m]
    k = k0 + np.cumsum(gT, axis=0)*gamma*T
    t = (np.arange(1, g.shape[0]+1)-0.5)*T
    tt = (t*np.ones((2, 1))).T
    s = np.diff(np.concatenate(([[0, 0]], g, g[None, -1,:])), axis=0)[1:,:]/T/1e3 # [mT/m/s] -> [T/m/s]
    m1 = np.cumsum(gT*tt, axis=0)*gamma*T
    m2 = np.cumsum(gT*(tt*tt+T**2/12), axis=0)*gamma*T
    v = (1/eta)*(L*s+R*gT)
    return k, g, s, m1, m2, t, v

def plotgradinfo(g, T: float = 4e-6):
    import matplotlib.pyplot as plt

    k, g, s, m1, m2, t, v = calcgradinfo(g, T)
    tms = t*1e3

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].plot(k[:,0], k[:,1])
    axs[0, 0].set_title('k-Space Trajectory')

    axs[0, 0].spines['top'].set_color('none')
    axs[0, 0].spines['bottom'].set_position('zero')
    axs[0, 0].spines['left'].set_position('zero')
    axs[0, 0].spines['right'].set_color('none')
    axs[0, 0].set(aspect='equal')

    axs[0, 0].set_xlabel('$k_x [m^{-1}]$', loc='left')
    axs[0, 0].set_ylabel('$k_y [m^{-1}]$', loc='top')

    axs[0, 1].plot(tms, k[:,0], label='k_x')
    axs[0, 1].plot(tms, k[:,1], label='k_y')
    axs[0, 1].plot(tms, np.sqrt(k[:,0]*k[:,0] + k[:,1]*k[:,1]), '--', label='|k|')
    axs[0, 1].set_title('k-Space vs. Time')
    axs[0, 1].set_xlabel('Time [ms]')
    axs[0, 1].set_ylabel('k-Space position [$m^{-1}$]')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[0, 2].plot(tms, g[:,0], label='$G_x$')
    axs[0, 2].plot(tms, g[:,1], label='$G_y$')
    axs[0, 2].plot(tms, np.sqrt(g[:,0]*g[:,0] + g[:,1]*g[:,1]), '--', label='|G|')

    axs[0, 2].set_title('Gradient vs. Time')
    axs[0, 2].set_xlabel('Time [ms]')
    axs[0, 2].set_ylabel('Amplitude [mT/m]')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    axs[1, 0].plot(tms, s[:,0], label='$S_x$')
    axs[1, 0].plot(tms, s[:,1], label='$S_y$')
    axs[1, 0].plot(tms, np.sqrt(s[:,0]*s[:,0] + s[:,1]*s[:,1]), '--', label='|S|')

    axs[1, 0].set_title('Slew Rate vs. Time')
    axs[1, 0].set_xlabel('Time [ms]')
    axs[1, 0].set_ylabel('Slew Rate [T/m/s]')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(tms, m1[:,0], label='$M1_x$')
    axs[1, 1].plot(tms, m1[:,1], label='$M1_y$')
    axs[1, 1].plot(tms, np.sqrt(m1[:,0]*m1[:,0] + m1[:,1]*m1[:,1]), '--', label='|M1|')

    axs[1, 1].set_title('1st Moment vs. Time')
    axs[1, 1].set_xlabel('Time [ms]')
    axs[1, 1].set_ylabel('1st Moment [s/m]')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[1, 2].plot(tms, m2[:,0], label='$M2_x$')
    axs[1, 2].plot(tms, m2[:,1], label='$M2_y$')
    axs[1, 2].plot(tms, np.sqrt(m2[:,0]*m2[:,0] + m2[:,1]*m2[:,1]), '--', label='|M2|')

    axs[1, 2].set_title('2nd Moment vs. Time')
    axs[1, 2].set_xlabel('Time [ms]')
    axs[1, 2].set_ylabel('2nd Moment [$s^2$/m]')
    axs[1, 2].legend()
    axs[1, 2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[1, 2].grid(True)

    return fig

def spiralgen_design(sys: dict, Nint: int, fov: float, res: float, Tread: float, 
                     us_type: Literal['linear', 'quadratic', 'hanning'] = 'linear', us_factor: float = 1, us_transition: tuple[float, float] = (1, 1)) \
      -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """ Given the system parameters, number of interleaves, FoV, resolution and Readout time, designs the variable density spiral.
    Calls spiral generation function bnispiralgen by James Pipe.
    
    Parameters
    ----------
    sys : dict
        Dictionary of given system parameters:
        max_grad  = Available gradient strength [mT/m]
        max_slew  = Available slew rate [mT/m/ms]
        spiral_type = 0 for Archimedean, 3 for Fermat
    Nint : int
        Number of interleaves
    fov : float
        FoV to be supported [m].
    res : float
        Resolution target [m].
    Tread : float
        Maximum readout length [s].
    us_type : str
        Type of undersampling transition. Options are 'linear', 'quadratic', or 'hanning'.
    us_factor : float
        Undersampling factor. This is the final spacing in the spiral.
    us_transition : tuple
        Undersampling transition points. This is a tuple of two floats, (us_0, us_1), where us_0 is the first transition point and us_1 is the second transition point.
        The spacing will be constant between us_0 and us_1, and then will transition to the final spacing.
        
    Returns
    -------
    k : NDArray
        Designed trajectory [mm^-1].
    g : NDArray
        Designed grad waveform [mT/m].
    grew : NDArray
        Designed rewinders [mT/m]
    s : NDArray
        Slew rate of the trajectory [T/m/s].
    time : NDArray
        Time axis [s].
    """
    #define GRAST    0.005 /* Desired gradient raster time (ms) */
    #define subrast    5      /* number of numerical cycles per gradient raster time */ 

    slewmax = sys['max_slew']
    gradmax = sys['max_grad']
    Td = 10e-6 # 10us
    oversamp = 1

    ngmax = ceil(Tread/Td)*oversamp*2

    spparams = ffi.new("float[20]")

    # Output variables
    xgrad_o = ffi.new("float[]", ngmax)
    ygrad_o = ffi.new("float[]", ngmax)
    zgrad_o = ffi.new("float[]", ngmax)

    spgrad_na = ffi.new("int*")
    spgrad_nb = ffi.new("int*")
    spgrad_nc = ffi.new("int*")
    spgrad_nd = ffi.new("int*")

    #define spARRSIZE  20

    #define spGAMMA     0
    #define spGMAX      1
    #define spSLEWMAX   2
    spparams[0] = 42.577 # kHz/mT
    spparams[1] = gradmax
    spparams[2] = slewmax

    #define spGTYPE     3
    spparams[3] = 2  # 0 = calculate through readout
                     # 1 = include grad ramp-down
                     # 2 = include rewinder to end at k=0
                     # 3 = include first moment comp */

    #define spFOVXY     4
    #define spFOVZ      5
    #define spRESXY     6
    #define spRESZ      7
    #define spARMS      8
    spparams[4] = fov
    spparams[5] = 0
    spparams[6] = res
    spparams[7] = 0
    spparams[8] = Nint # number of spiral interleaves*/

    #define spSTYPE     9
    #define spUSTYPE   10
    #define spUS0      11
    #define spUS1      12
    #define spUSR      13
    spparams[9] = sys['spiral_type']        # 0 = Archimedean
                                            # 1 = Cylinder DST 
                                            # 2 = Spherical DST
                                            # 3 = Fermat:Floret

    # the next 4 variables are for variable density spirals 
    # they create a transition in the radial spacing as the k-space radius goes from 0 to 1, i.e.
    #    0 < kr < us_0 : spacing = Nyquist distance 
    # us_0 < kr < us_1 : spacing increases to us_r (affected by ustype)
    # us_1 < kr < 1    : spacing = us_r
    
    # rate of change in undersampling
    #  0 = linear
    #  1 = quadratic
    #  2 = hanning 
    if us_type == 'linear':
        spparams[10] = 0
    elif us_type == 'quadratic':
        spparams[10] = 1
    elif us_type == 'hanning':
        spparams[10] = 2
    else:
        raise ValueError('Invalid us_type. Choose from "linear", "quadratic", or "hanning".')       

    spparams[11] = us_transition[0] # us_0
    spparams[12] = us_transition[1] # us_1
    spparams[13] = us_factor # us_r

    #define spDWELL    14
    #define spREADPTS  15

    # spparams[14]
    # spparams[15]
    #define spSLOP_PER 16
    # For sloppy spirals, this lets us define periodicity in units of iteration loop time */
    # set this to zero if you do not want sloppy spirals */
    spparams[16] = 0

    # void bnispiralgen(float* spparams, int maxarray, float *gxarray, float *gyarray, float *gzarray, 
    #                  int *spgrad_na, int *spgrad_nb, int *spgrad_nc, int *spgrad_nd)

    #       This function takes parameters passed in spparams array and
    #   returns a single spiral arm calculated numerically

    #   The corresponding gradient waveforms are in gxarray and gyarray
    #   spgrad_na reflects the number of gradient points to reach the end of k-space
    #   spgrad_nb = spgrad_na + the number of gradient points to ramp G to zero
    #   spgrad_nc = spgrad_nb + the number of gradient points to rewind k to zero
    #   spgrad_nd = spgrad_nc + the number of gradient points for first moment compensation

    #   Assignments below indicate units of input parameters
    #   All units input using kHz, msec, mT, and m!

    #   grad = gm exp(i theta) i.e. gm, theta are magnitude and angle of gradient
    #   kloc = kr exp(i phi)   i.e. kr, phi are magnitude and angle of k-space
    #   alpha = theta - phi    the angle of the gradient relative to that of k-space
    #                          (alpha = Pi/2, you go in a circle
    #                           alpha = 0, you go out radially)

    #   The variable rad_spacing determines the radial spacing
    #   in units of the Nyquist distance.
    #   rad_spacing = 1 gives critical sampling
    #   rad_spacing > 1 gives undersampling
    #   rad_spacing can vary throughout spiral generation to create variable density spirals

    bnispiralgen(spparams, ngmax, xgrad_o, ygrad_o, zgrad_o, spgrad_na, spgrad_nb, spgrad_nc, spgrad_nd)

    Ngrad = spgrad_na[0]
    Ngrew = spgrad_nc[0]
    Gx = np.frombuffer(ffi.buffer(xgrad_o, 4*Ngrew), dtype=np.float32)
    Gy = np.frombuffer(ffi.buffer(ygrad_o, 4*Ngrew), dtype=np.float32)
    Gz = np.frombuffer(ffi.buffer(zgrad_o, 4*Ngrew), dtype=np.float32)

    g    = np.column_stack([Gx[:Ngrad], Gy[:Ngrad]]) # [mT/m]
    grew = np.column_stack([Gx[Ngrad:Ngrew], Gy[Ngrad:Ngrew]]) # [mT/m]

    k, g, s, m1, m2, time, v = calcgradinfo(g, Td)

    return k, g, grew, s, time


def vds_design(sys: dict, Nint: int, fov: list, res: float, Tread: float) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """ Given the system parameters, number of interleaves, FoV, resolution and Readout time, designs the variable density spiral.
    Calls spiral generation function calc_vds by Brian Brian Hargreaves.

    Parameters
    ----------
    sys : dict
        Dictionary of given system parameters:
        max_grad  = Available gradient strength [mT/m]
        max_slew  = Available slew rate [T/m/s]
        adc_dwell = Sample (gradient) dwell time [s]
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
    k : NDArray
        Designed trajectory [mm^-1].
    g : NDArray
        Designed grad waveform [mT/m].
    s : NDArray
        Slew rate of the trajectory [T/m/s].
    time : NDArray
        Time axis [s].
    """

    slewmax = sys['max_slew']*100
    gradmax = sys['max_grad']/10
    Td = sys['adc_dwell']
    oversamp = sys['os']

    Tg = Td/oversamp	# gradient rate.

    krmax = 5/res
    numfov = fov.__len__()
    ngmax = ceil(Tread/Td)

    fov_i = ffi.new("double[]", numfov)
    fovs = np.exp(log(1/krmax)*np.arange(0, numfov, 1))

    for ii,fv in enumerate(fovs):
        fov_i[ii] = fov[ii]*fv

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

def vds_fixed_ro(sys: dict, fov: list, res: float, Tread: float) -> tuple[npt.NDArray | None, npt.NDArray | None, npt.NDArray | None, int]:
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
    k : NDArray
        Designed k-space trajectory [m^-1]
    g : NDArray
        Designed gradient waveform [mT/m]
    time : NDArray
        Time axis [s]
    nint : int
        Number of interleaves that supports unalised FoV with the desired resolution.
    """

    tol = 0.98
    krmax_target = 1/(2*res*1e-3) # m^-1
    krmax = 0
    nint = 0
    k, g, time = None, None, None
    while krmax < krmax_target*tol:
        nint = nint+1
        k,g,_,time = vds_design(sys, nint, fov, res, Tread)
        # print(f'nint={nint}, krmax={krmax}, tread={time[-1]} \n')
        krmax = sqrt(k[-1,0]*k[-1,0] + k[-1,1]*k[-1,1])
    
    return k, g, time, nint

def raster_to_grad(g: npt.NDArray, adc_dwell: float, grad_dwell: float) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    t = (np.arange(1, g.shape[0]+1)-0.5)*adc_dwell
    t_grad = np.arange(0, np.round(g.shape[0]/(grad_dwell/adc_dwell)))*grad_dwell
    t = np.concatenate(([0], t))
    g = np.concatenate(([[0,0]], g), axis=0)

    g_grad = np.empty((t_grad.shape[0], 2))
    g_grad[:,0] = np.interp(t_grad, t, g[:,0])
    g_grad[:,1] = np.interp(t_grad, t, g[:,1])

    return t_grad, g_grad


if __name__ == "__main__":

    print('Testing the spiralgen_design function')
    sys = {
        'max_slew':  120,  # [T/m/s] 
        'max_grad':   24,  # [mT/m] 
        'spiral_type': 0,  # 0, 3
        }

    Nint = 15
    fov  = 24*1e-2 # [m]
    res = 2.4e-3 # [m]
    Tread = 3e-3

    k, g, grew, s, t = spiralgen_design(sys, Nint, fov, res, Tread)

    plotgradinfo(np.vstack([g, grew]), 10e-6)
    plt.show()

    print('Testing the vds_design function')
    sys = {
        'max_slew'  :  170,  # [T/m/s] 
        'max_grad'  :   38,  # [mT/m] 
        'adc_dwell' : 1e-6, # [s]
        'os'        :    8
        }

    # Test the vds_design function directly
    Nint = 19
    fov  = [25.6] # [cm]
    res = 2 # [mm]
    Tread = 3e-3

    k, g, s, t = vds_design(sys, Nint, fov, res, Tread)

    fig = plotgradinfo(g, sys['adc_dwell'])
    fig.suptitle('VDS Design output', fontsize=16)

    plt.show()

    # Test the fixed ro function

    k2, g2, t2, nint2 = vds_fixed_ro(sys, fov, res, Tread)
    fig = plotgradinfo(g2, sys['adc_dwell'])
    fig.suptitle('VDS fixed RO output', fontsize=16)

    plt.show()

    pass