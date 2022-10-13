#!/usr/bin/env python3

import numpy as np
import os

f1='ScorerNeutronSq_SofQ_seed4096_edge.npy'
f2='ScorerNeutronSq_SofQ_seed4096_content.npy'
f3='ScorerNeutronSq_SofQ_seed4096_hit.npy'

os.system(f'rm {f1}')
os.system(f'rm {f2}')
os.system(f'rm {f3}')

os.system('prompt -g watersphere_bias.gdml -n 1e5')

x=np.load(f1)
w=np.load(f2)
hit=np.load(f3)

res = np.array([x.sum(), w.sum(), hit.sum()])
np.set_printoptions(precision=16)
print(res)
np.testing.assert_allclose(res, [5894.344942918124, 2001.947312844668, 3333.], rtol=1e-15)
print('passed prompt test')
