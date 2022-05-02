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

from Cinema.Interface import *
import numpy as np

eKin2k = importFunc('pt_eKin2k', type_dbl, [type_dbl] )
angleCosine2Q = importFunc('pt_angleCosine2Q', type_dbl, [type_dbl, type_dbl, type_dbl] )
wl2ekin = importFunc('pt_wl2ekin', type_dbl, [type_dbl] )
ekin2wl = importFunc('pt_ekin2wl', type_dbl, [type_dbl] )
#output unit mm/sec`
ekin2v = importFunc('pt_ekin2speed', type_dbl, [type_dbl] )
v2ekin = importFunc('pt_speed2ekin', type_dbl, [type_dbl] )

# def elasticQ(cosAngle, fl)
angleCosine2QMany = np.vectorize(angleCosine2Q)
v2ekinMany = np.vectorize(v2ekin)

# def qElastic(l)
