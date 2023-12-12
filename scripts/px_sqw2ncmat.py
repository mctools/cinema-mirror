#!/usr/bin/env python3

import numpy as np
const_eV2kk = 1/2.072124652399821e-3
const_hbar = 6.582119569509068e-16

import h5py
filename = 'qehist.h5'

with h5py.File(filename, "r") as f:
    omega = f['omega'][()]
    q = f['q'][()]
    sqw = f['s'][()]


kt = 0.0253
#################################################################################
import matplotlib.pyplot as plt
fig=plt.figure()
ax = fig.add_subplot(111)
H = sqw.T

X, Y = np.meshgrid(q, omega*const_hbar)
import matplotlib.colors as colors
pcm = ax.pcolormesh(X, Y, H, cmap=plt.cm.jet,  norm=colors.LogNorm(vmin=H.max()*1e-4, vmax=H.max()),)
fig.colorbar(pcm, ax=ax)
plt.xlabel('Q, Aa^-1')
plt.ylabel('energy, eV')
#################################################################################

enSize=100
QSize=250
maxQ=15

def flip(rawin, axis=0, factor=1.):
    s = [slice(None)]*rawin.ndim
    s[axis] = slice(-2,0,-1)
    return  np.concatenate((rawin, rawin[tuple(s)]*factor),axis=axis)

en = flip(omega*const_hbar,factor=-1)

s = [slice(None)]*sqw.ndim
s[1] = slice(-2,0,-1)
d2=np.concatenate((sqw,sqw[tuple(s)]*np.exp(-en[(enSize-1):-1]/kt)),axis=1)

#################################################################################
import matplotlib.pyplot as plt
fig=plt.figure()
ax = fig.add_subplot(111)
H = d2.T

X, Y = np.meshgrid(q, en)
import matplotlib.colors as colors
pcm = ax.pcolormesh(X, Y, H, cmap=plt.cm.jet,  norm=colors.LogNorm(vmin=H.max()*1e-4, vmax=H.max()),)
fig.colorbar(pcm, ax=ax)
plt.xlabel('Q, Aa^-1')
plt.ylabel('energy, eV')
#################################################################################

        
def sqw2sab(sqw, Q, fre, kt):
    alpha = Q*Q/(kt*const_eV2kk)
    beta = fre*const_hbar/kt
    sab = sqw*0.5*kt*kt*const_eV2kk/(2*np.pi*const_hbar)
    return sab, alpha, beta

sab, alpha, beta = sqw2sab(d2, q, en/const_hbar, kt)
knl={}
knl['alphagrid'] = alpha
knl['betagrid'] = beta
knl['sab'] = sab
#################################################################################
import matplotlib.pyplot as plt
fig=plt.figure()
ax = fig.add_subplot(111)
H = sab.T

X, Y = np.meshgrid(alpha, beta)
import matplotlib.colors as colors
pcm = ax.pcolormesh(X, Y, H, cmap=plt.cm.jet,  norm=colors.LogNorm(vmin=H.max()*1e-4, vmax=H.max()),)
fig.colorbar(pcm, ax=ax)
plt.xlabel('alpha')
plt.ylabel('beta')
#################################################################################

import NCrystal.ncmat as ncncmat
import pathlib
_res="""
NCMAT v5

@DENSITY
  1.0 atoms_per_aa3

@DYNINFO
  element  Si
  fraction 1
  type     scatknl
  temperature 300
  """
for n in ('alphagrid','betagrid','sab'):
    _res += ncncmat.formatVectorForNCMAT(n,knl[n])#NB: Hoping for no entries like 0r100 
pathlib.Path('gendata.txt').write_text(_res)