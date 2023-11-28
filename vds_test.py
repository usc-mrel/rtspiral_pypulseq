from libvds.vds import vds_design
import matplotlib.pyplot as plt


sys = {
    'max_slew'  :  170,  # [T/m/s] 
    'max_grad'  :   38,  # [mT/m] 
    'adc_dwell' : 1e-6, # [s]
    'os'        :    8
    }

Nint = 19
fov  = [25.6] # [cm]
res = 2 # [mm]
Tread = 3e-3

k, g, s, t = vds_design(sys, Nint, fov, res, Tread)

plt.plot(t*1e3, g)
plt.show()
pass