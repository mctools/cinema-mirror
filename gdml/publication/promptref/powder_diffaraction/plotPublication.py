import numpy as np
from Cinema.Prompt import PromptFileReader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import sys, os

paper_x, paper_y = None, None

scanData = np.loadtxt('./data/pus-Data.dat')
simuData = np.loadtxt('./data/prompt-SimuData.dat')


fig, ax = plt.subplots(1,1)
ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))

l2, = plt.plot(scanData[:,0], scanData[:,1]*5, 'ko', markersize=6.0)
l1, = plt.plot(simuData[:,0], simuData[:,1], 'r', linewidth=2.0)

plt.ylim(0,1.1)
plt.xlim(0,130.1)
plt.xlabel("scattering angle, deg", fontsize=20)
plt.ylabel("Count rate, arb. unit", fontsize=20)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.legend([l1, l2], ['Prompt NCrystal simulated', 'IFE PUS measured'] ,fontsize=15)
plt.show()
