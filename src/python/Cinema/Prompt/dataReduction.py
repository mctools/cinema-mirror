#!/usr/bin/env python3

LIGHT_C = 299792458 #m/s
AMU_2_EVC2 = 931494095.17
MASS_N = 1.00866491588 #amu
MASS_N_EVC2 = MASS_N * AMU_2_EVC2
MASS_N_EVMS = MASS_N_EVC2 / (LIGHT_C ** 2)
PLANK_H = 4.135667662e-15 #planck const [eV*s]


def tof2hw_INS(z2, r2, tof, weight):
    import numpy as np

    d = 3.355e-10 #m
    rlt_n = 1 #reflection order

    # spectoscopy paras
    e2 = (PLANK_H ** 2) * (rlt_n ** 2 ) / (8 * MASS_N_EVMS * (d ** 2)) * (1 + ((r2 * 0.5 / z2) ** 2))
    t2 = 4 * MASS_N_EVMS * d * z2 / (PLANK_H * rlt_n) 

    l1 = 17000 * 0.001 #m

    center = np.array(tof)
    weight = np.array(weight)

    t1 = center - t2
    e1 = MASS_N_EVMS * 0.5 * ((l1 ** 2) / (t1 ** 2))

    hw = e1 - e2

    hw = hw[-850:]
    weight = weight[-850:]
    return hw, weight

def getPeakReso(hw, weight):
    import numpy as np
    from scipy.signal import find_peaks, peak_widths
    peaks, _ = find_peaks(weight, height=0.01, distance=50)
    result = peak_widths(weight, peaks)
    left_x = np.interp(result[2], np.arange(850), hw)
    right_x = np.interp(result[3], np.arange(850), hw)
    reso = (left_x-right_x)/hw[peaks]
    return reso
