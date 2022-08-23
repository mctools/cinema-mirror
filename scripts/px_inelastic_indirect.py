#!/usr/bin/env python3

import h5py
from Cinema.Interface.sctfunc import QeSqw
import matplotlib.pyplot as plt
import numpy as np
from Cinema.Interface.units import hbar
import argparse
from Cinema.Interface import plotStyle
plotStyle()

#parameters
#scattering angle(scatAngle) in degree
#energy out in eV
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--scattering-angle', action='store', type=float, required=True,
                    dest='scatAngle', help='scattering angle(scatAngle) in degree')
# 180-135
# down scattering, as only calculated by px_inelastic_direct.py
# omega should be negtive, however, it is positive in the calculatoin
parser.add_argument('-e', '--energy-out', action='store', type=float, required=True,
                    dest='enout', help='energy out in eV')
args=parser.parse_args()

#0.0363271
#1cm-1=0.00012eV
#TFXA 135deg 32cm-1 (0.00384eVe), in the book, Vibrational Spectroscopy with Neutrons With Applications in Chemistry, Biology, Materials Science and Catalysis (Philip C. H. Mitchell, Stewart F Parker etc.) page 101
#TFXA 135deg 24cm-1 (0.00288eV), https://www.isis.stfc.ac.uk/Pages/TOSCA-History.aspx
#according to ncrystal, graphite 002 d-spacing is 3.3555, as lamba=2*d*sin(135), lambda is 4.7454Aa(i.e. 3.63271meV). It is like the book is correct.

scatAngle = args.scatAngle
enout = args.enout

sqw = QeSqw('qehist.h5')
enin = np.linspace(enout, enout+0.5, 1000)
sqw.plot(color_order=1e-4)

s, q, w=sqw.calXSAtFixedAngle( enin, enout, scatAngle)

plt.plot(q, w)

plt.figure()
plt.plot((enin-enout)/0.00012, s)
plt.grid()
plt.xlabel(r'energy, cm^{-1}')
plt.ylabel(r'cross section, arb. unit')
plt.tight_layout()
plt.show()
