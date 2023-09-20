#!/usr/bin/env python3

from Cinema.Prompt.scorer import Scorer
import numpy as np

class Scorer4test(Scorer):
    def __init__(self) -> None:
        super().__init__()
        self.cfg_scorer = 'Scorer'
        self.cfg_name = 'Testing'
        self.cfg_num = 20.0
    
scr = Scorer4test()
print(scr.cfg)
np.testing.assert_string_equal(scr.cfg, 'scorer=Scorer;name=Testing;num=20.0;')

scr.cfg_num = f'{scr.cfg_num}test'
np.testing.assert_string_equal(scr.cfg, 'scorer=Scorer;name=Testing;num=20.0test;')
