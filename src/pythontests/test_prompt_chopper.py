#!/usr/bin/env python3

import numpy as np
import os
from Cinema.Prompt import PromptFileReader

f1='ScorerTOF_Out_seed836214.mcpl.gz'
os.system(f'rm {f1}')
os.system('prompt -g chopper.gdml -n 1e5 -s 836214')


f = PromptFileReader(f1)

hist_weight = f.getData('content').sum()
hist_edge = f.getData('edge').sum()

np.set_printoptions(precision=16)
res = np.array([hist_edge, hist_weight])
print(res)
np.testing.assert_allclose(res, [0.9900000000000001, 25828.], rtol=1e-15)
