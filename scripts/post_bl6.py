import h5py
import numpy as np
import matplotlib.pyplot as plt

c = 299792458 #m/s
amu2eVc2 = 931494095.17
n_mass = 1.00866491588 #amu
n_mass_eVc2 = n_mass * amu2eVc2
n_mass_eVms = n_mass_eVc2 / c ** 2
h = 4.135667662e-15 #planck const [eV*s]
d = 3.355e-10 #m
rlt_n = 1 #reflection order
# spectoscopy paras
r2 = 300 * 0.001 #m
z2 = 260 * 0.001 #m
e2 = h ** 2 * rlt_n ** 2 / (8 * n_mass_eVms * d ** 2) * (1 + (r2 * 0.5 / z2) ** 2)
t2 = 4 * n_mass_eVms * d / (h * rlt_n) * z2

l1 = 17000 * 0.001 #m

f = h5py.File('totTof','r')
center = np.array(f.get('center'))
weight = np.array(f.get('weight'))

t1 = center - t2

hw = n_mass_eVms * 0.5 * (l1 ** 2 / t1 ** 2) - e2
# print(hw)

fig = plt.figure()
plt.plot(hw, weight)
plt.savefig('hwSpectrum')
f.close()