import numpy as np
import matplotlib.pyplot as plt

from Cinema.Prompt import PromptFileReader
from Cinema.Interface import plotStyle

plotStyle()
const_c = 299792458 #m/s
const_planck = 4.135667662e-15 #eV*s
const_massn = 1.674e-27 #kg
const_ev2j = 1.602176634e-19 #J/eV
const_ang2m = 1.0e-10 #m/AA
pathlength = 10 #m


def get_wavelength(velocity):

    velocity = np.array(velocity)
    return const_planck/velocity/const_massn*const_ev2j/const_ang2m

def get_maxwell_tof(wavelength):

    wavelength = np.array(wavelength)
    return 2 * ((949 / T) ** 2) / (wavelength ** 5) * np.exp(-(949/T)/wavelength**2)

def norm_f(x, y, numbin):

    x = np.array(x)
    y = np.array(y)
    integral = (y * x.max()/numbin).sum()
    factor = 1/integral

    return factor

def mc_data(filename):

    f = np.loadtxt(filename)
    nvals, x, y, weight = f[:,3], f[:,0], f[:,1], f[:,3]*f[:,1]
    return nvals, x, y, weight

def pt_data(filename):

    f = PromptFileReader(filename)
    x=f.getData('edge')
    y=f.getData('content')
    return x[:-1], y/np.diff(x)

def tol_correct(y):

    tol = 1.0e-1
    y = np.array(y)
    y = tol * (y<tol) + y * (y>=tol)
    return y

# read prompt data
# -- from local output
xp, yp = pt_data('./prompt/ScorerTOF_Out_seed4096.mcpl.gz')
xp_norm, yp_norm=pt_data('./prompt/ScorerTOF_In_seed4096.mcpl.gz')

# read mcstas data
# -- from mcstas code
nvals, xm, ym, weightm = mc_data('./mcstas_diskchopper/TOF.dat')
nvals_n, xm_n, ym_n, weightm_n = mc_data('./mcstas_diskchopper/TOF_in.dat')

# Sear formular
T = 293 # unit K
xs = np.linspace(0,0.02,10000)[1:]
ys = get_maxwell_tof(get_wavelength(pathlength/xs))

plt.yscale('log')
plt.plot(xp,  tol_correct(yp*norm_f(xp_norm, yp_norm, 10000)), linestyle='-',  color='b', label="Prompt")
plt.plot(xm/1e6, tol_correct(ym*norm_f(xm_n/1e6, ym_n, 10000)), linestyle=':', linewidth=5, color='red',label="McStas")
# plt.plot(xs, ys*norm_f(xs,ys,10000), linestyle='-.', color='g', linewidth=2, label="Sears") #To benchmark with formular
plt.xlabel('Time-of-flight(s)')
plt.ylabel('Normalized Intersity')
plt.legend()
plt.tight_layout()

plt.show()