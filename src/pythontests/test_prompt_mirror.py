#!/usr/bin/env python3

import numpy as np
import os
from Cinema.Prompt import PromptFileReader

f1='ScorerPSD_Monitor2_seed836213.mcpl.gz'
os.system(f'rm {f1}')
os.system('prompt -g guide.gdml -n 1e4 -s 836213')


f = PromptFileReader(f1)

hist_weight = f.getData('content').sum()
hist_hit = f.getData('hit').sum()
hist_edge = f.getData('xedge').sum()

np.set_printoptions(precision=16)

res = np.array([hist_edge, hist_weight, hist_hit])
print(res)
np.testing.assert_allclose(res, [0.,1893.2899347456905, 2062.], rtol=1e-15)
