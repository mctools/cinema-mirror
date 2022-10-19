#!/usr/bin/env python3

import numpy as np
import os
from Cinema.Prompt import PromptFileReader

filelist=['SimpleThermalGun.gdml', 'IsotropicGun.gdml', 'UniModeratorGun.gdml', 'MPIGun.gdml', 'MaxwellianGun.gdml']
valuelist=[[5894.344942918124, 1895., 1895.], [5894.344942918124, 5406., 5406.], [ 5894.344942918124, 78823., 78823.], [5894.344942918124, 86395., 86395.], [5894.344942918124, 83811., 83811.]]
f1='ScorerNeutronSq_SofQ_seed103.mcpl.gz'

for i in range(0,5):
    os.system(f'rm {f1}')
    os.system('prompt -g %s -s 103 -n 1e5'%filelist[i])

    f = PromptFileReader(f1)
    hist_weight = f.getData('content').sum()
    hist_hit = f.getData('hit').sum()
    hist_edge = f.getData('edge').sum()

    res = np.array([hist_edge, hist_weight, hist_hit])
    np.set_printoptions(precision=16)
    print(res)
    np.testing.assert_allclose(res, valuelist[i], rtol=1e-15)

print('passed prompt_gun test')
