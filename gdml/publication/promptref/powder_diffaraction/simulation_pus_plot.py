import numpy as np
from Cinema.Prompt import PromptFileReader
from Cinema.Interface import plotStyle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import sys, os
plotStyle()

pusData = np.loadtxt('PUS_data.dat')
f = PromptFileReader(f'ScorerDeltaMomentum_ST_template_seed4096.mcpl.gz')
x=f.getData('edge')
x=x[:-1]+np.diff(x)*0.5
y=f.getData('content')


fig, ax = plt.subplots(1,1)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))

l2, = plt.plot(pusData[:,0], pusData[:,1]/pusData[:,1].sum()*100, 'ko')
l1, = plt.plot(x-0.22, y/y.sum()/1.38*100, 'r')

# plt.ylim(0,1.1)
# plt.xlim(0,130.1)
plt.grid()

plt.xlabel("scattering angle, deg")
plt.ylabel("Count rate, arb. unit")
plt.legend([l1, l2], ['Simulated', r'PUS measured@1.5549$\AA$ '])

plt.tight_layout()
plt.show()
