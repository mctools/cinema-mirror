#!/usr/bin/env python3
import numpy as np
import prompt_suite as ps
import matplotlib.pyplot as plt

# Check H1 at 1e-4
expected = [2.0, 2.0, 4.0, 26.0, 64.0, 261.0, 608.0, 1335.0, 3032.0, 6113.0, 12374.0, 22024.0, 28511.0, 20072.0, 4114.0, 77.0, 0.0, 0.0, 0.0, 0.0]
cfg ='freegas::H/1gcm3/H_is_H1'
incidentEnergy = 1e-4
energyBinNum = 20
energyRange = [1e-6, 1e1]
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
