
from _spiralgen.lib import bnispiralgen
from _spiralgen import ffi
import numpy as np
import numpy.typing as npt
from math import exp, log, ceil

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

def spiralgen_design(sys: dict, Nint: int, fov: float, res: float, Tread: float) -> npt.ArrayLike:
    """ Given the system parameters, number of interleaves, FoV, resolution and Readout time, designs the variable density spiral.

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
    Returns
    -------
    k : ArrayLike
        Designed trajectory [mm^-1].
    g : ArrayLike
        Designed grad waveform [mT/m].
    grew : ArrayLike
        Designed rewinders [mT/m]
    s : ArrayLike
        Slew rate of the trajectory [T/m/s].
    time : ArrayLike
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
    spparams[10] = 0                    # rate of change in undersampling
                                        #  0 = linear
                                        #  1 = quadratic
                                        #  2 = hanning 

    spparams[11] = 1
    spparams[12] = 1
    spparams[13] = 1

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

    # Gx = Gx[0:-1:oversamp]
    # Gy = Gy[0:-1:oversamp]
    # Gz = Gz[0:-1:oversamp]

    g    = np.column_stack([Gx[:Ngrad], Gy[:Ngrad]]) # [mT/m]
    grew = np.column_stack([Gx[Ngrad:Ngrew], Gy[Ngrad:Ngrew]]) # [mT/m]

    k, g, s, m1, m2, time, v = calcgradinfo(g, Td)

    return k, g, grew, s, time


if __name__ == "__main__":
    import matplotlib.pyplot as plt

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

    plotgradinfo(np.row_stack([g, grew]), 10e-6)
    plt.show()
    pass