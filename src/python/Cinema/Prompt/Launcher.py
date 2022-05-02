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

from ..Interface import *

_pt_Launcher_getInstance = importFunc('pt_Launcher_getInstance', type_voidp, [] )
_pt_Launcher_setSeed = importFunc('pt_Launcher_setSeed', None, [type_voidp, type_sizet] )
_pt_Launcher_loadGeometry = importFunc('pt_Launcher_loadGeometry', None, [type_voidp, type_cstr] )
_pt_Launcher_getTrajSize = importFunc('pt_Launcher_getTrajSize', type_sizet, [type_voidp])
_pt_Launcher_getTrajectory = importFunc('pt_Launcher_getTrajectory', None, [type_voidp, type_npdbl2d])
_pt_Launcher_go = importFunc('pt_Launcher_go', None, [type_voidp, type_sizet, type_dbl, type_bool])

class Launcher():
    def __init__(self):
        self.cobj = _pt_Launcher_getInstance()

    def setSeed(self, seed):
        _pt_Launcher_setSeed(self.cobj, seed)

    def loadGeometry(self, fileName):
        _pt_Launcher_loadGeometry(self.cobj, fileName.encode('utf-8'));

    def getTrajSize(self):
        return _pt_Launcher_getTrajSize(self.cobj)

    def getTrajectory(self):
        trjsize = self.getTrajSize()
        trj = np.zeros([trjsize, 3])
        _pt_Launcher_getTrajectory(self.cobj, trj)
        return trj

    def go(self, numPrimary, printPrecent=0.1, recordTrj=False):
        _pt_Launcher_go(self.cobj, numPrimary, printPrecent, recordTrj)
