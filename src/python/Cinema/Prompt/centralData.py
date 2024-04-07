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


from ..Interface import *


_pt_CentralData_getInstance = importFunc('pt_CentralData_getInstance', type_voidp, [] )

_pt_CentralData_getGidiThreshold = importFunc('pt_CentralData_getGidiThreshold', type_dbl, [type_voidp] )
_pt_CentralData_setGidiThreshold = importFunc('pt_CentralData_setGidiThreshold', None, [type_voidp, type_dbl] )

_pt_CentralData_getGidiPops = importFunc('pt_CentralData_getGidiPops', type_cstr, [type_voidp] )
_pt_CentralData_setGidiPops = importFunc('pt_CentralData_setGidiPops', None, [type_voidp, type_cstr] )

_pt_CentralData_getGidiMap = importFunc('pt_CentralData_getGidiMap', type_cstr, [type_voidp] )
_pt_CentralData_setGidiMap = importFunc('pt_CentralData_setGidiMap', None, [type_voidp, type_cstr] )

_pt_CentralData_getEnableGidi = importFunc('pt_CentralData_getEnableGidi', type_bool, [type_voidp] )
_pt_CentralData_setEnableGidi = importFunc('pt_CentralData_setEnableGidi', None, [type_voidp, type_bool] )

_pt_CentralData_getEnableGidiPowerIteration = importFunc('pt_CentralData_getEnableGidiPowerIteration', type_bool, [type_voidp] )
_pt_CentralData_setEnableGidiPowerIteration = importFunc('pt_CentralData_setEnableGidiPowerIteration', None, [type_voidp, type_bool] )


@singleton
class CentralData():
    def __init__(self):
        self.cobj = _pt_CentralData_getInstance()

    def getGidiThreshold(self):
        return _pt_CentralData_getGidiThreshold(self.cobj)
    
    def setGidiThreshold(self, t):
        _pt_CentralData_setGidiThreshold(self.cobj, t)

    # fixme: backslash in the path can not be decoded!
    def getGidiPops(self):
        return _pt_CentralData_getGidiPops(self.cobj).decode('utf-8')
    
    def setGidiPops(self, s):
        _pt_CentralData_setGidiPops(self.cobj, s.encode('utf-8'))

    def getGidiMap(self):
        return _pt_CentralData_getGidiMap(self.cobj).decode('utf-8').replace('', '')
    
    def setGidiMap(self, s):
        _pt_CentralData_setGidiMap(self.cobj, s.encode('utf-8'))

    def getEnableGidi(self):
        return _pt_CentralData_getEnableGidi(self.cobj)
    
    def setEnableGidi(self, b):
        _pt_CentralData_setEnableGidi(self.cobj, b)

    def getEnableGidiPowerIteration(self):
        return _pt_CentralData_getEnableGidiPowerIteration(self.cobj)
    
    def setEnableGidiPowerIteration(self, b):
        _pt_CentralData_setEnableGidiPowerIteration(self.cobj, b)
    
    
