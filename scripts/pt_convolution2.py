#!/usr/bin/env python3
from scipy.signal import unit_impulse
from scipy.signal import convolve
import matplotlib.pyplot as plt
import numpy as np

aa = 1e-10 #m/aa
cm = 1e-2 #m/cm
const_aa2cm = aa / cm #cm/aa
const_neutron_mass = 1.674e-27 #kg
const_intensity_inp = 1e6
pulse_loc = 0.02
sampleLength_z = 0.2 #cm
sample_density_aa3 = 5 #per aa3
sample_density_cm3 = sample_density_aa3 / (const_aa2cm**3) #n/aa3 / (cm/aa)^3 = n/cm^3

def get_Rw(i_inp, w):
    rw = 0
    for j in range(len(i_inp)-1):
        e_in = (e_axi[j+1] + e_axi[j]) * 0.5
        e_out = e_in - w if e_in>w else e_in
        prob_noCollide = np.exp(- getMacroXs(e_in, sample_density_cm3, e_out) * sampleLength_z)
        # print(prob_not_interaction) 
        rw += i_inp[j] * np.sqrt(e_out/e_in) * (1-prob_noCollide)
    return rw

def getMacroXs(e_in, density, e_out, Sw=1):
    microXs = np.sqrt(e_out/e_in) * Sw * 1e-24 #cm2
    return density * microXs

def moveAxis(axis, deviation):
    for i in range(len(axis)):
        axis[i] = axis[i] + deviation
    return axis

e_pulse = int((pulse_loc-0.)*1000)
sw = unit_impulse(200, e_pulse)
# print(sw)

e_axi = np.arange(0,0.2,0.001)
e_axi2 = np.zeros(len(e_axi))

# incoming intensity
i_inp = np.zeros(200)
rw = np.zeros(200)
e_low = int((0.11 - 0.) * 1000)
e_high = int((0.12 - 0.) * 1000)
i_ave = const_intensity_inp / (e_high-e_low)
for i in range(e_low, e_high+1):
    i_inp[i] = i_ave



w_axi = np.zeros(200)
rw = np.zeros(100)
for w in np.arange(0.,0.1,0.001):
    index = int(w/0.001)
    rw[index]=get_Rw(i_inp, w)
w_axi2 = np.zeros(len(w_axi))
# iw_scattered = unit_impulse(45, e_pulse) * const_intensity_inp
# rw = np.zeros(len(iw))
# rw = np.flip(iw)
for j in range(len(e_axi)):
    w_axi2[j] = -e_axi[-(j+1)]+pulse_loc
# print(rw)
cv = convolve(sw, rw, 'full')

# plt.plot(e_axi, sw)
plt.plot(e_axi, i_inp, label='Incident intensity I(E)')
# plt.plot(e_axi2, rw)
plt.plot(moveAxis(e_axi[1:101], 0.02), rw, label='Calculated by formulation R(w)')
# plt.plot(e_axi, iw_scattered)
# plt.plot(w_axi2, rw, label='Calculated Resolution function R(w)')
cv_axi = np.arange(0, len(sw)+len(rw)-1) * 0.001 - 0.02
plt.plot(cv_axi, cv, label='Convolution S*R(w)')
# plt.plot(cv)
plt.xticks(np.arange(-0.04,0.4,0.02))
plt.grid()
plt.legend()
plt.show()


# a = np.array([0,1,0])
# b = np.array([3,2,1,-1])
# c = convolve(a,b,'full')
# print(c)