#!/usr/bin/env python3

import numpy as np
import os

filelist=['SimpleThermalGun.gdml', 'IsotropicGun.gdml', 'UniModeratorGun.gdml', 'MPIGun.gdml', 'MaxwellianGun.gdml']
valuelist=[[5894.344942918124, 1895., 1895.], [5894.344942918124, 5406., 5406.], [ 5894.344942918124, 78823., 78823.], [5894.344942918124, 86395., 86395.], [5894.344942918124, 83811., 83811.]]
f1='ScorerNeutronSq_SofQ_seed103_edge.npy'
f2='ScorerNeutronSq_SofQ_seed103_content.npy'
f3='ScorerNeutronSq_SofQ_seed103_hit.npy'

for i in range(0,5):
    os.system(f'rm {f1}')
    os.system(f'rm {f2}')
    os.system(f'rm {f3}')
    os.system('prompt -g %s -s 103 -n 1e5'%filelist[i])

    x=np.load(f1)
    w=np.load(f2)
    hit=np.load(f3)

    res = np.array([x.sum(), w.sum(), hit.sum()])
    np.set_printoptions(precision=16)
    print(res)
    np.testing.assert_allclose(res, valuelist[i], rtol=1e-15)

print('passed prompt_gun test')
