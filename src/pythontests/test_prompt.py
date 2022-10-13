#!/usr/bin/env python3

import numpy as np
import os

os.system('prompt -g watersphere_bias.gdml -n 1e5')

x=np.load('ScorerNeutronSq_SofQ_seed4096_edge.npy')
w=np.load('ScorerNeutronSq_SofQ_seed4096_content.npy')
hit=np.load('ScorerNeutronSq_SofQ_seed4096_hit.npy')

res = np.array([x.sum(), w.sum(), hit.sum()])
np.set_printoptions(precision=16)
print(res)
np.testing.assert_allclose(res, [5894.344942918124, 2001.947312844668, 3333.], rtol=1e-15)
print('passed prompt test')
