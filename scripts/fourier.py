#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from Cinema.Interface import plotStyle
plotStyle()

lim=8
x=np.linspace(-lim,lim,101)
x_new=np.linspace(-lim*2,lim*2,201)
y=np.exp(-x*x/2)

plt.semilogy(x, y)

yy = fftconvolve(y,y)
yy_ex = np.convolve(y,y)

plt.figure()
plt.semilogy(x_new, yy, 's',label='direct fftconv')
plt.semilogy(x_new, yy_ex, label='ref')
plt.legend(loc=0)

a=2
distort = np.exp(-a*x)
recv = np.exp(a*x_new)

plt.figure()
y_dis = y*distort
yy_dis = fftconvolve(y_dis, y_dis)
# plt.figure()
plt.semilogy(x_new, yy_dis*recv, 's', label = 'distort')
plt.semilogy(x_new, yy_ex,label='ref')
plt.legend(loc=0)
plt.show()

