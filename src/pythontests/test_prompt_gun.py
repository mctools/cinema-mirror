#!/usr/bin/env python3

import numpy as np
import os
from Cinema.Prompt import PromptFileReader

guntest_dict ={}
guntest_dict['SimpleThermalGun'] = {'gdml': 'SimpleThermalGun.gdml', 'mcpl': 'ScorerDeltaMomentum_SofQ_seed103.mcpl.gz', 'value': [5894.344942918124, 10.994339431719403, 16.]}
guntest_dict['IsotropicGun'] = {'gdml': 'IsotropicGun.gdml', 'mcpl': 'ScorerDeltaMomentum_SofQ_seed103.mcpl.gz', 'value': [5894.344942918124, 56.378238724171055, 55.]}
guntest_dict['UniModeratorGun'] = {'gdml': 'UniModeratorGun.gdml', 'mcpl': 'ScorerDeltaMomentum_SofQ_seed103.mcpl.gz', 'value': [5894.344942918124, 7640.151101725564, 798.]}
guntest_dict['MPIGun'] = {'gdml': 'MPIGun.gdml', 'mcpl': 'ScorerDeltaMomentum_SofQ_seed103.mcpl.gz', 'value': [5894.344942918124, 3887.8295735861293, 851.]}
guntest_dict['MaxwellianGun'] = {'gdml': 'MaxwellianGun.gdml', 'mcpl': 'ScorerDeltaMomentum_SofQ_seed103.mcpl.gz', 'value': [5894.344942918124, 3767.290904484039, 832.]}

for gunname in guntest_dict:
    gun = guntest_dict[gunname]
    os.system('rm %s' % gun['mcpl'])
    os.system('prompt -g %s -s 103 -n 1e3' % gun['gdml'])
    
    f = PromptFileReader(gun['mcpl'])
    hist_weight = f.getData('content').sum()
    hist_hit = f.getData('hit').sum()
    hist_edge = f.getData('edge').sum()
    
    res = np.array([hist_edge, hist_weight, hist_hit])
    np.set_printoptions(precision=16)
    print(res)
    np.testing.assert_allclose(res, gun['value'], rtol=1e-15)

print('passed prompt_gun test')
