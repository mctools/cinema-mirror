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

from ..Interface import *

_pt_PythonGun_new = importFunc('pt_PythonGun_new', type_voidp, [type_pyobject])
_pt_PythonGun_delete = importFunc('pt_PythonGun_delete', None, [type_voidp])
_pt_PythonGun_pushToStack = importFunc('pt_PythonGun_pushToStack', None, [type_voidp, type_npdbl1d])


class PythonGun():
    def __init__(self):
        self.cobj = _pt_PythonGun_new(self)
        
    def __del__(self):
        _pt_PythonGun_delete(self.cobj)

    def generate(self):
        pdata = np.zeros(9)
        pdata[0] = self.sampleEnergy()
        pdata[1] = self.sampleWeight()
        pdata[2] = self.sampleTime()
        pdata[3:6] = self.samplePosition()
        pdata[6:9]  = self.sampleDirection()        
        _pt_PythonGun_pushToStack(self.cobj, pdata)      
    
    def sampleEnergy(self):
        return 0.0253

    def sampleWeight(self):
        return 1. 
    
    def sampleTime(self):
        return 0.
    
    def samplePosition(self):
        return np.array([0.,0.,0.])
    
    def sampleDirection(self):
        return np.array([0.,0.,1.])



    

