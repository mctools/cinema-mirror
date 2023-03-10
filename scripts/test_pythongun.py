#!/usr/bin/env python3

from Cinema.Prompt import Launcher, Visualiser
from Cinema.Prompt.geo import Box, LogicalVolume, Transformation3D
from Cinema.Prompt.gun import PythonGun

from Cinema.Prompt.utils import call_python_method


gun = PythonGun()

import time
tic = time.perf_counter() 
for i in range(100000):
    gun.pyGenerate()
print(f'time {time.perf_counter()-tic}s')

