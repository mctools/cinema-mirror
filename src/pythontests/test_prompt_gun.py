#!/usr/bin/env python3

import numpy as np
import os
from Cinema.Prompt import PromptFileReader

guntest_dict ={}
guntest_dict['SimpleThermalGun'] = {'gdml': 'SimpleThermalGun.gdml', 'mcpl': 'ScorerNeutronSq_SofQ_seed103.mcpl.gz', 'value': [5894.344942918124, 14., 14.]}
guntest_dict['IsotropicGun'] = {'gdml': 'IsotropicGun.gdml', 'mcpl': 'ScorerNeutronSq_SofQ_seed103.mcpl.gz', 'value': [5894.344942918124, 55., 55.]}
guntest_dict['UniModeratorGun'] = {'gdml': 'UniModeratorGun.gdml', 'mcpl': 'ScorerNeutronSq_SofQ_seed103.mcpl.gz', 'value': [5894.344942918124, 790., 790.]}
guntest_dict['MPIGun'] = {'gdml': 'MPIGun.gdml', 'mcpl': 'ScorerNeutronSq_SofQ_seed103.mcpl.gz', 'value': [5894.344942918124, 855., 855.]}
guntest_dict['MaxwellianGun'] = {'gdml': 'MaxwellianGun.gdml', 'mcpl': 'ScorerNeutronSq_SofQ_seed103.mcpl.gz', 'value': [5894.344942918124, 850., 850.]}

for gunname in guntest_dict:
    gun = guntest_dict[gunname]
    os.system('rm %s'%gun['mcpl'])
    os.system('prompt -g %s -s 103 -n 1e3'%gun['gdml'])
    
    f = PromptFileReader(gun['mcpl'])
    hist_weight = f.getData('content').sum()
    hist_hit = f.getData('hit').sum()
    hist_edge = f.getData('edge').sum()
    
    res = np.array([hist_edge, hist_weight, hist_hit])
    np.set_printoptions(precision=16)
    print(res)
    np.testing.assert_allclose(res, gun['value'], rtol=1e-15)

print('passed prompt_gun test')