from Cinema.Prompt import PromptFileReader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import argparse

from Cinema.Interface import plotStyle

plotStyle()

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--linear', action='store_true', dest='logscale', help='colour bar in log scale')
f = PromptFileReader('ScorerPSD_Scat0Bias0p1_1e8_1A_o_seed4096.mcpl.gz')
args=parser.parse_args()
data=f.getData('hit')
count=f.getData('hit')
x=f.getData('xedge'); y=f.getData('yedge'); X, Y = np.meshgrid(x, y)
fig=plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('x, mm')
plt.ylabel('y, mm')
if args.logscale:
  pcm = ax.pcolormesh(X, Y, data.T, cmap=plt.cm.jet, norm=colors.LogNorm(vmin=1, vmax=data.max()), shading='auto')
else:
  pcm = ax.pcolormesh(X, Y, data.T, cmap=plt.cm.jet,shading='auto')
fig.colorbar(pcm, ax=ax)
count=count.sum()
integral= data.sum()
plt.figure()
plt.subplot(211)
plt.plot(x[:-1]+np.diff(x)*0.5, data.sum(axis=1))
plt.xlabel('integral x')
plt.title('ScorerPSD_Scat0Bias0p1_1e8_1A_o_seed4096')
plt.subplot(212)
plt.plot(y[:-1]+np.diff(y)*0.5, data.sum(axis=0))
plt.xlabel('integral y')
plt.show()
