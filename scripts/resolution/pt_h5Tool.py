#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def read_hdf5(file):
    import h5py
    f = h5py.File(file, 'r')
    neutrondata = f['neutron']
    w = neutrondata['energy'][()]
    q = neutrondata['q'][()]
    s = neutrondata['s'][()]
    return w, q, s

def plot_swq(w, q, swq, log=False):
    from matplotlib.pyplot import pcolormesh
    import matplotlib.colors as colors
    fig = plt.figure()
    ww, qq = np.meshgrid(w, q)

    if log:
        im = pcolormesh(qq, ww, swq, cmap=plt.cm.jet, norm=colors.LogNorm(vmin=swq.max()*1e-3, vmax=swq.max()), shading='auto')
    else:
        im = pcolormesh(qq, ww, swq, cmap=plt.cm.jet, shading='auto')
    plt.xlabel('q(aa-1)')
    plt.ylabel('w(eV)')
    # plt.legend()
    plt.colorbar(im)

def plot_sw(w, q, swq, fmt='--r', normalize=False, yscale = 1):
    if normalize:
        sw = swq.sum(0) / swq.sum()
    else:
        sw = swq.sum(0)
    sw = sw * yscale 
    plt.plot(w, sw, fmt)

if __name__ == '__main__':
    pass