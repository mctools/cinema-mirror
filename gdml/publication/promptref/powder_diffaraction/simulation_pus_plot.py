import numpy as np
from Cinema.Prompt import PromptFileReader
from Cinema.Interface import plotStyle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import sys, os

pusData = np.loadtxt('PUS_data.dat')
simuData = np.loadtxt('prompt_data.dat')

plotStyle()
fig, ax = plt.subplots(1,1)
ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))

l2, = plt.plot(pusData[:,0], pusData[:,1]*5, 'ko')
l1, = plt.plot(simuData[:,0], simuData[:,1], 'r')

plt.ylim(0,1.1)
plt.xlim(0,130.1)
plt.xlabel("scattering angle, deg")
plt.ylabel("Count rate, arb. unit")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.legend([l1, l2], ['Prompt NCrystal simulated', 'IFE PUS measured'])
plt.grid()
plt.tight_layout()
plt.show()
