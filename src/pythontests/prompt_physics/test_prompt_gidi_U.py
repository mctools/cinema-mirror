#!/usr/bin/env python3
import numpy as np
import prompt_suite as ps
import matplotlib.pyplot as plt

expected = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 1568.0, 1856.0, 8503.0, 86150.0, 203.0, 128.0, 5.0, 0.0, 0.0]
cfg ='freegas::U92/18gcm3/U_is_U238'
incidentEnergy = 1e6
energyBinNum = 20
energyRange = [1, 1e8]
partnum = 100000
manualcheck = False # plot and visual check

def printExpected(counts):
    ct = list(counts)
    for c in ct:
        if isinstance(c,np.ndarray):
            print(list(c), sep=',')
        else:
            print(c)
    
if manualcheck:
    import openmc_suite as os
    counts = ps.promptRun(cfg, incidentEnergy, -5, energyBinNum, energyRange[0], energyRange[1], partnum=partnum, plot=True)
    os.openmcRun(cfg, incidentEnergy, energyBinNum, energyRange[0], energyRange[1], partnum=partnum)
    printExpected(counts)
    plt.show()
else:
    counts = ps.promptRun(cfg, incidentEnergy, -5, energyBinNum, energyRange[0], energyRange[1], partnum=partnum)
    printExpected(counts)
    np.testing.assert_allclose(counts[2], expected, rtol=1e-15)