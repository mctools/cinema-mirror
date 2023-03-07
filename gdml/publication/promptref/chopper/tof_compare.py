import numpy as np
import matplotlib.pyplot as plt
from mccodelib import mcplotloader

from Cinema.Prompt import PromptFileReader
from Cinema.Interface import plotStyle

import yn_ana_mcpl_code

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

    Loader = mcplotloader.McCodeDataLoader(filename)
    Loader.load()

    node = Loader.plot_graph
    # n = node.getnumdata()
    data = node.getdata_idx(0)
    nvals = np.array(data.Nvals)
    x = np.array(data.xvals)
    y = np.array(data.yvals) 
    return nvals, x, y, y*nvals

def mc_data2(filename):

    f = np.loadtxt(filename)
    return f

def pt_data(filename):

    f = PromptFileReader(filename)
    x=f.getData('edge')
    y=f.getData('content')
    return x[:-1], y/np.diff(x)

def pt_data2(filepattern, seedStart, seedEnd):

    psd = yn_ana_mcpl_code.McplAnalysor1D(f'./*{filepattern}_seed*.mcpl.gz')
    x_hist, content_hist = psd.getHistMany(seedStart, seedEnd) 
    return x_hist[:-1], content_hist/np.diff(x_hist)

def tol_correct(y):

    tol = 1.0e-1
    y = np.array(y)
    y = tol * (y<tol) + y * (y>=tol)
    return y

# read prompt data
# -- from local output
xp, yp = pt_data('ScorerTOF_Out_seed4096.mcpl.gz')
xp_norm, yp_norm=pt_data('ScorerTOF_In_seed4096.mcpl.gz')
# -- from parallel output
# xp, yp = pt_data2('TOF_Out', 1, 30)
# xp_norm, yp_norm = pt_data2('TOF_In', 1, 30)

# read mcstas data
# -- from mcstas code
nvals, xm, ym, weightm = mc_data('./mcstas_diskchopper/TOF.dat')
nvals_n, xm_n, ym_n, weightm_n = mc_data('./mcstas_diskchopper/TOF_in.dat')

# -- from numpy
# file = mc_data2('./Test_DiskChoppers_20230228_222851/TOF.dat')
# print(file)

# Sear formular
T = 293 # unit K
xs = np.linspace(0,0.02,10000)[1:]
ys = get_maxwell_tof(get_wavelength(pathlength/xs))

plt.yscale('log')
plt.plot(xp,  tol_correct(yp*norm_f(xp_norm, yp_norm, 10000)), linestyle='-',  color='b', label="Prompt")
plt.plot(xm/1e6, tol_correct(ym*norm_f(xm_n/1e6, ym_n, 10000)), linestyle=':', linewidth=5, color='red',label="McStas")

# plt.plot(xs, ys*norm_f(xs,ys,10000), linestyle='-.', color='g', linewidth=2, label="Sears")
plt.xlabel('Time-of-flight(s)')
plt.ylabel('Normalized Intersity')
plt.legend()
plt.tight_layout()

print(f'Counts from mcstas: {weightm.sum()}')
print(f'Counts from prompt: {yp.sum()}')
plt.show()