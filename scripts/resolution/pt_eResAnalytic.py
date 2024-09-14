#!/usr/bin/env python3

# violini:
# energy resolution from (23) of
# https://www.sciencedirect.com/science/article/pii/S016890021301423X
# 
# Suffix: 
# ms -> monochromatic to sample
# pm -> pulse shaping to monochromatic
# sd -> sample to detector

import numpy as np

const_neutron_mass_amu = 1.00866491588
const_c  = 299792458 #(m/s)/c
const_dalton2eVc2 =  931494095.17
const_neutron_mass_evc2 = const_neutron_mass_amu * const_dalton2eVc2 / (const_c**2); #[ eV/(m/s)^2 ]

def ekin2velocity(ekin):
    '''
    e = 0.5 * m * v**2
    v = sqrt(2*e/m)
    '''
    return np.sqrt(2*ekin/const_neutron_mass_evc2) 


def ste2timeWindow(ste):
    '''
    Uniform time window calcul from standard deviation.
    '''
    return np.sqrt(12 * ste**2)

# mv^3 = [eV * m/s]
# (mv^3)^2 = [eV^2 * (m/s)^2]
# abg = [m^2 * s^2] = [m^2 * s^2]
# (L_pm*L_sd)^2 = [m^4]
# sigma_hw = [eV^2 * (m/s)^2 * m^2 * s^2 / m^4] = [eV^2]


e_in = 0.10324919397990652 #eV
w = 0.03
e_out = e_in - w

lms_ave = 1300 * 1e-3 #mm 1.3m
lsd_ave = 3000 * 1e-3 #mm 3m
lpm_ave = 25000 *1e-3 #mm 25m

ste_p = 38 *1e-6#us
ste_m = 6 *1e-6#us
std_lms = 0 *1e-2#cm

v_ave = ekin2velocity(e_in)
vprime_ave = ekin2velocity(e_out)

print(v_ave, vprime_ave, v_ave/vprime_ave)
alpha_square = (lms_ave + lsd_ave * (v_ave/vprime_ave)**3)**2 * ste_p**2
beta_square = ((lms_ave + lsd_ave * (v_ave/vprime_ave)**3)**2 + lpm_ave**2) * ste_m **2
gamma_square = (lpm_ave/v_ave)**2 * std_lms**2

var_hw = (const_neutron_mass_evc2 * v_ave**3)**2 * (alpha_square + beta_square + gamma_square) / (lpm_ave*lsd_ave)**2
ste_hw = np.sqrt(var_hw)
print(ste_hw)
# print("Time windows: ")
# print(f"First chopper: {ste2timeWindow(ste_p) * 1e6} us")
# print(f"Second chopper: {ste2timeWindow(ste_m) * 1e6} us")
