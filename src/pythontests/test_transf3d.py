#!/usr/bin/env python3

################################################################################
##                                                                            ##
##  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        ##
##                                                                            ##
##  Copyright 2021-2024 Prompt developers                                     ##
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



from Cinema.Prompt.geo import Transformation3D
import numpy as np

for i in np.arange(10):
    angles=np.random.random(3)*np.pi
    mat1 = Transformation3D(rot_z=angles[0], rot_new_x=angles[1], rot_new_z=angles[2])
    mat2 = Transformation3D()
    mat2.setRot(rot_z=angles[0], rot_new_x=angles[1], rot_new_z=angles[2])
    data = np.random.random([1000,3])
    data2 = np.copy(data)
    np.testing.assert_allclose(data, data2, rtol=1e-15, atol=1e-15)

    mat1.transformInplace(data)
    mat2.transformInplace(data2)
    np.testing.assert_allclose(data, data2, rtol=1e-15, atol=1e-15)
