
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

def vds_design(sys: dict, Nint: int, fov: list, res: float, Tread: float) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """ Given the system parameters, number of interleaves, FoV, resolution and Readout time, designs the variable density spiral.

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
    Td = sys['adc_dwell']
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

    tol = 0.98
    krmax_target = 1/(2*res*1e-3) # m^-1
    krmax = 0
    nint = 0
    while krmax < krmax_target*tol:
        nint = nint+1
        k,g,_,time = vds_design(sys, nint, fov, res, Tread)
        # print(f'nint={nint}, krmax={krmax}, tread={time[-1]} \n')
        krmax = sqrt(k[-1,0]*k[-1,0] + k[-1,1]*k[-1,1])
    
    return k, g, time, nint

def raster_to_grad(g: npt.ArrayLike, adc_dwell: float, grad_dwell: float) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    t = (np.arange(1, g.shape[0]+1)-0.5)*adc_dwell
    t_grad = np.arange(0, np.round(g.shape[0]/(grad_dwell/adc_dwell)))*grad_dwell
    t = np.concatenate(([0], t))
    g = np.concatenate(([[0,0]], g), axis=0)

    g_grad = np.empty((t_grad.shape[0], 2))
    g_grad[:,0] = np.interp(t_grad, t, g[:,0])
    g_grad[:,1] = np.interp(t_grad, t, g[:,1])

    return t_grad, g_grad



if __name__ == "__main__":
    import matplotlib.pyplot as plt

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