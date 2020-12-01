#!/usr/bin/env python3

import numpy as np
from PiXiu.PhononCalc import MeshCell

kt =0.0253 #temperature in kelvin
calc = MeshCell('./data/Al/mesh.hdf5', './data/Al/cell.json', kt)

enSize=100
QSize=300
maxQ=10.

S, Q, en = calc.calcPowder(maxQ, enSize, QSize)
calc.show(S, Q, en)

#
# import multiprocessing as mp
# from itertools import product
#
# for h in range(0,maxNum+1,jump):  # half a space
#         for k in range(-maxNum,maxNum+1,jump):
#             hkllist=[]
#             for l in range(-maxNum,maxNum+1,jump):
#                 if h==0:
#                     if k<0:
#                         continue #half a plane
#                     elif k==0 and l<0: #half an axis, keeping 0,0,0
#                         continue
#                 print('processing', (h,k,l))
#                 hkl=np.array([h,k,l])
#                 if np.linalg.norm(np.dot(hkl,calc.lattice_reci)) > maxQ:
#                     print('skiping hkl')
#                     continue
#                 Spart, Q, en = calc.calcSqw(hkl, maxQ, enSize, qSize)
#                 if not(h==0 and k==0 and l==0):
#                     Spart *= 2
#                 S += Spart #space group 1
#
#             #     hkllist.append((hkl,maxQ, enSize, qSize))
#             # with mp.Pool(processes=8) as pool:
#             #     results = pool.starmap(calc.calcSqw, hkllist )
#
# # np.savetxt('S.dat',S)
# # np.savetxt('Q.dat',Q)
# # np.savetxt('en.dat',en)
# calc.show(S, Q, en)
