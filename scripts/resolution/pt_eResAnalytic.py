#!/usr/bin/env python3

# energy resolution from (23) of
# https://www.sciencedirect.com/science/article/pii/S016890021301423X
# 
# Suffix: 
# ms -> monochromatic to sample
# pm -> pulse shaping to monochromatic
# sd -> sample to detector

import numpy as np

const_neutron_mass_amu = 1.00866491588
const_c  = 299792458
const_dalton2eVc2 =  931494095.17
const_neutron_mass_evc2 = const_neutron_mass_amu * const_dalton2eVc2 / (const_c*const_c); #[ eV/(Aa/s)^2 ]

def ekin2velocity(ekin):
    '''
    e = 0.5 * m * v**2
    v = sqrt(2*e/m)
    '''
    return np.sqrt(2*ekin/const_neutron_mass_evc2) * const_c

e_in = 0.0253 #eV
w = 0
e_out = e_in - w

lms_ave = 1300 #mm 1.3m
lsd_ave = 3000 #mm 3m
lpm_ave = 25000 #mm 25m

ste_p = 0.05
ste_m = 0.05
std_lms = 0

v_ave = ekin2velocity(e_in)
vprime_ave = ekin2velocity(e_out)

alpha_square = (lms_ave + lsd_ave * (v_ave/vprime_ave)**3)**2 * ste_p**2
beta_square = ((lms_ave + lsd_ave * (v_ave/vprime_ave)**3)**2 + lpm_ave**2) * 0
gamma_square = (lpm_ave/v_ave)**2 * std_lms**2

var_hw = (const_neutron_mass_evc2 * v_ave**3)**2 * (alpha_square + beta_square + gamma_square)
print(var_hw)