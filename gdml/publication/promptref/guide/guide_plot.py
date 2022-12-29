#!/usr/bin/env python3

from Cinema.Prompt import PromptFileReader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import argparse
from Cinema.Interface import plotStyle
plotStyle()

f = PromptFileReader('ScorerPSD_Monitor2_seed4096.mcpl.gz')
data=f.getData('content')
count=f.getData('hit')
x=f.getData('xedge'); y=f.getData('yedge'); X, Y = np.meshgrid(x, y)
fig=plt.figure()
ax = fig.add_subplot(111)
pcm = ax.pcolormesh(X, Y, data.T, cmap=plt.cm.jet,shading='auto')
fig.colorbar(pcm, ax=ax)
plt.xlabel('x, mm')
plt.ylabel('y, mm')
plt.tight_layout()

fig=plt.figure()
ax = fig.add_subplot(111)
mcstas_data = np.loadtxt('Monitor2_xy_1672190448.x_y')[:20,:]
pcm = ax.pcolormesh(X, Y, mcstas_data, cmap=plt.cm.jet,shading='auto')
fig.colorbar(pcm, ax=ax)
plt.xlabel('x, mm')
plt.ylabel('y, mm')
plt.tight_layout()

################################################

plt.figure()
plt.subplot(111)

pixel_width = 0.0035 #in m, 70mm against 20 pixels

incidentNum = PromptFileReader('ScorerPSD_Monitor1_seed4096.mcpl.gz').getData('content').sum()
x=(x[:-1]+np.diff(x)*0.5)*1e-3
plt.plot(x, data.sum(axis=0)/incidentNum/pixel_width, label='Promt')

mcstas_monitor1_flux = (np.loadtxt('Monitor1_xt_1672190448.x_y')[:20,:]).sum()

plt.plot(x, mcstas_data.sum(axis=0)/mcstas_monitor1_flux/pixel_width, 'o', label='McStas')
plt.legend()
plt.xlabel(r'y, m')
plt.ylabel(r'Intensity $\int f(x,y) \mathrm{d}x$, n$^{-1}$ m$^{-1}$')

plt.tight_layout()

# plt.subplot(212)
# plt.plot(y[:-1]+np.diff(x)*0.5, data.sum(axis=1))
# plt.xlabel('integral y')
# plt.show()

plt.show()
