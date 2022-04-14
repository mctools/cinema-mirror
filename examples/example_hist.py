#!/usr/bin/env python3

################################################################################
##                                                                            ##
##  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        ##
##                                                                            ##
##  Copyright 2021-2022 Prompt developers                                     ##
##                                                                            ##
##  Licensed under the Apache License, Version 2.0 (the "License");           ##
##  you may not use this file except in compliance with the License.          ##
##  You may obtain a copy of the License at                                   ##
##                                                                            ##
##      http://www.apache.org/licenses/LICENSE-2.0                            ##
##                                                                            ##
##  Unless required by applicable law or agreed to in writing, software       ##
##  distributed under the License is distributed on an "AS IS" BASIS,         ##
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  ##
##  See the License for the specific language governing permissions and       ##
##  limitations under the License.                                            ##
##                                                                            ##
################################################################################

import matplotlib.pyplot as plt
import numpy as np
from Prompt.Math.Hist import Hist1D
from Prompt.Math import *
import Prompt.Math.Units as Units

#converters
ekin = 0.0253*Units.eV
angle = 45.*Units.deg
eout = 0.03*Units.eV
wl = 1.8*Units.Aa

print(2*np.pi/eKin2k(ekin))
print(angleCosine2Q(np.cos(angle), ekin, eout))
print(wl2ekin(wl))
print(ekin2v(ekin))

num = 100000
hist = Hist1D(0.,1.2,50)
hist.fillmany(x=np.random.random(num), weight=np.random.random(num))

plt.plot(hist.getEdge()[:-1], hist.getWeight())
plt.show()
