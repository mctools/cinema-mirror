import numpy as np
from Cinema.Prompt import PromptFileReader
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--linear', action='store_true', dest='logscale', help='colour bar in log scale')
args=parser.parse_args()
f = PromptFileReader('ScorerNeutronSq_PofQ2_HW_seed22.mcpl.gz')
x=f.getData('edge')
y=f.getData('content')
if args.logscale:
  plt.semilogy(x[:-1],y/np.diff(x), label=f'total weight={y.sum()}')
else:
  plt.plot(x[:-1],y/np.diff(x), label=f'total weight={y.sum()}')
plt.grid()
plt.legend()
plt.show()
