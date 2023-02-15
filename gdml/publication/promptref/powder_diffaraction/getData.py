import numpy as np
from Cinema.Prompt import PromptFileReader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import sys, os

totalX, totalY = None, None
for name in glob.glob("./result/ScorerNeutronStheta_*.mcpl.gz"):
    print( name )
    f = PromptFileReader(name)
    x=f.getData('edge')
    y=f.getData('content')
    totalY = y if totalY is None else totalY+y
    totalX = x

ymax = max((totalY[10:]/np.diff(totalX[10:]))[400:8000])
xs = totalX[10:-1]
ys = (totalY/np.diff(totalX)/ymax)[10:]

fname = open("./data/prompt-SimuData.dat", "w+")
for i in range(len(xs)):
    print( "%.3f\t%.6f" %(xs[i], ys[i]), file=fname )
fname.close()
