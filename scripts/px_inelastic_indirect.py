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
parser.add_argument('-a', '--scattering-angle', action='store', type=float, dest='scatAngle', help='scattering angle(scatAngle) in degree')
# 180-135
# down scattering, as only calculated by px_inelastic_direct.py
# omega should be negtive, however, it is positive in the calculatoin
parser.add_argument('-e', '--energy-out', action='store', type=float,
                    dest='enout', help='energy out in eV')

parser.add_argument('-i', '--input', action='store', type=str, default='qehist.h5',
                    dest='input', help='input 2D Sqw from  the inelastic coherent script')

parser.add_argument('--instrument', action='store', type=str, default='',
                    dest='instrument', help='Pre-defined parameters for a given instrument, including "TFXA": 135deg,  32cm-1 ')
args=parser.parse_args()

#0.0363271
#1cm-1=0.00012eV
#TFXA 135deg 32cm-1 (0.00384eVe), in the book, Vibrational Spectroscopy with Neutrons With Applications in Chemistry, Biology, Materials Science and Catalysis (Philip C. H. Mitchell, Stewart F Parker etc.) page 101
#TFXA 135deg 24cm-1 (0.00288eV), https://www.isis.stfc.ac.uk/Pages/TOSCA-History.aspx
#according to ncrystal, graphite 002 d-spacing is 3.3555, as lamba=2*d*sin(135), lambda is 4.7454Aa(i.e. 3.63271meV). It is like the book is correct.

scatAngle = args.scatAngle
enout = args.enout
inputfile = args.input

if args.instrument:
    if args.instrument=='TFXA':
        scatAngle = 135
        enout = 0.00384
    else:
        print(f'The instrument name {args.instrument} is unknown. Use the -h option to see all the supported instruments')
        import sys
        sys.exit()

if (scatAngle is None) or (enout is None):
    print(f'Both scattering angle and the energy out of the analysor need to be defined')
    import sys
    sys.exit()


sqw = QeSqw(inputfile)
enin = np.linspace(enout, enout+0.5, 1000)
sqw.plot(color_order=1e-4)
plt.tight_layout()

sqw.plot(color_order=1e-4)
s, q, w=sqw.calXSAtFixedAngle( enin, enout, scatAngle)
plt.plot(q, w, '--r', linewidth=2)
plt.tight_layout()

plt.figure()
plt.plot((enin-enout)/0.00012, s/np.trapz(s, (enin-enout)/0.00012)/9*1000, label='PiXiu coherent single phonon')

plt.grid()
plt.xlabel(r'energy, cm^{-1}')
plt.ylabel(r'cross section, arb. unit')
plt.tight_layout()
plt.show()
