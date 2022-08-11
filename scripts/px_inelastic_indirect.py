#!/usr/bin/env python3

import h5py
from Cinema.Interface.sctfunc import QeSqw
import matplotlib.pyplot as plt
import numpy as np
from Cinema.Interface.units import hbar

#1cm-1=0.00012eV
#TFXA 135deg 32cm-1 (0.00384eVe), in the book, Vibrational Spectroscopy with Neutrons With Applications in Chemistry, Biology, Materials Science and Catalysis (Philip C. H. Mitchell, Stewart F Parker etc.) page 101
#TFXA 135deg 24cm-1 (0.00288eV), https://www.isis.stfc.ac.uk/Pages/TOSCA-History.aspx
#according to ncrystal, graphite 002 d-spacing is 3.3555, as lamba=2*d*sin(135), lambda is 4.7454Aa(i.e. 3.63271meV). It is like the book is correct.


sqw = QeSqw('qehist.h5', 20)
scatAngle = 180-135
#down scattering, as only calculated by px_inelastic_direct.py
# omega should be negtive, however, it is positive in the calculatoin
enout = 0.0363271
enin = np.linspace(enout, 0.5, 1000)
sqw.plot(color_order=1e-4)

s, q, w=sqw.calXSAtFixedAngle( enin, enout, scatAngle)

plt.plot(q, w)

plt.figure()
plt.plot((enin-enout)/0.00012, s)
plt.grid()

plt.show()
