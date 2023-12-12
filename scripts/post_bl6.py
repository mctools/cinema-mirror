import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

c = 299792458 #m/s
amu2eVc2 = 931494095.17
n_mass = 1.00866491588 #amu
n_mass_eVc2 = n_mass * amu2eVc2
n_mass_eVms = n_mass_eVc2 / (c ** 2)
h = 4.135667662e-15 #planck const [eV*s]
d = 3.355e-10 #m
rlt_n = 1 #reflection order
# spectoscopy paras
r2 = 600 * 0.001 #m
z2 = 300 * 0.001 #m
e2 = (h ** 2) * (rlt_n ** 2 ) / (8 * n_mass_eVms * (d ** 2)) * (1 + ((r2 * 0.5 / z2) ** 2))
t2 = 4 * n_mass_eVms * d * z2 / (h * rlt_n) 

l1 = 17000 * 0.001 #m

f = h5py.File('totTof','r')
center = np.array(f.get('center'))
weight = np.array(f.get('weight'))
f.close()

t1 = center - t2
e1 = n_mass_eVms * 0.5 * ((l1 ** 2) / (t1 ** 2))

hw = e1 - e2
# print(t2)
hw = hw[-850:]
weight = weight[-850:]

fig = plt.figure()

peaks, _ = find_peaks(weight, height=0.01, distance=50)
result = peak_widths(weight, peaks)
left_x = np.interp(result[2], np.arange(850), hw)
right_x = np.interp(result[3], np.arange(850), hw)
reso = (left_x-right_x)/hw[peaks]
print(reso)
plt.plot(hw, weight)
plt.plot(hw[peaks], weight[peaks], "x")
plt.savefig('hwSpectrum')
