#!/usr/bin/env python

import numpy as np
from PiXiu.PhononCalc.Hdf5Mesh import Hdf5Mesh
from PiXiu.Common.Units import *

#['lattice', 'mass', 'scl', 'pos']
lattice=np.array([[2.27983857518, 0.0, 0.0],
[-1.13991928759, 1.97439812263, 0.0],
[0.0 ,0.0 ,3.527773]])
mass=np.array([9,9])
pos=np.array([[0.333333333333, 0.666666666667, 0.75],[0.666666666667, 0.333333333333, 0.25]])
bc = np.array([0.779, 0.779	]) #scat length
kt=1002*8.6173324e-5

calc = Hdf5Mesh(lattice, mass, pos, bc, kt, './bigdata/Be_554_888/meshT.hdf5')

# import matplotlib.pyplot as plt
# cen, hist = calc.dos()
# plt.plot(cen, hist)
# plt.show()

print('First qpoint', calc.qpoint[0])
# print('matrix', calc.calmsd2())
# print('isotropic msd', calc.isoMsd())



enSize=250
maxNum=1
jump=1

qSize=200*maxNum
maxQ=20

S=np.zeros([qSize, enSize])
Q=np.zeros(qSize)
en=np.zeros(enSize)

import multiprocessing as mp
from itertools import product


for h in range(0,maxNum+1,jump):  # half a space
        for k in range(-maxNum,maxNum+1,jump):
            hkllist=[]
            for l in range(-maxNum,maxNum+1,jump):
                if h==0:
                    if k<0:
                        continue #half a plane
                    elif k==0 and l<0: #half an axis, keeping 0,0,0
                        continue
                print('processing', (h,k,l))
                hkl=np.array([h,k,l])
                if np.linalg.norm(np.dot(hkl,calc.lattice_reci)) > maxQ:
                    print('skiping hkl')
                    continue
                Spart, Q, en = calc.calcSqw(hkl, maxQ, enSize, qSize)
                if not(h==0 and k==0 and l==0):
                    Spart *= 2
                S += Spart #space group 1

            #     hkllist.append((hkl,maxQ, enSize, qSize))
            # with mp.Pool(processes=8) as pool:
            #     results = pool.starmap(calc.calcSqw, hkllist )

# np.savetxt('S.dat',S)
# np.savetxt('Q.dat',Q)
# np.savetxt('en.dat',en)
calc.show(S, Q, en)
