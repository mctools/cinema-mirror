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

from enum import Enum, unique
import numpy as np
from .DataLoader import DataLoader

# class DataLoader():
#     def __init__(self):
#         self.tof = 1 #vector
#         self.pid = 1 #vector
#         self.tofpidMat = 1 #matrix
#         self.tofMonitor = 1  #vector or matrix
#         self.protonPulse = 1 #vector
#         self.protonCharge = 1 #vector
#         self.distMod2Monitor = 1 #vector
#         self.distMod2Sample =1 #double

@unique
class Normalise(Enum):
    skip = 0
    byMonitor = 1
    byProtonCharge = 2

class RunData(DataLoader):
    def __init__(self, fname, moduleName, normMethod = Normalise.byMonitor):
        super().__init__(fname, moduleName)
        self.normalise(normMethod)

    def normalise(self, normMethod):
        if normMethod == Normalise.skip:
            pass

        elif normMethod == Normalise.byMonitor:
            totMonitor  = self.tofMonitor.sum()
            if totMonitor == 0:
                raise RunTimeError('Monitor count is zero')
            self.tofpidMat = self.tofpidMat.astype(float)
            self.tofpidMat /= totMonitor

        elif normMethod == Normalise.byProtonCharge:
            totCharge = self.protonCharge.sum()
            if totCharge == 0:
                raise RunTimeError('Proton charge is zero')

        else:
            raise RunTimeError('Unknown normalise method')



class SampleData(RunData):
    def __init__(self, fname, moduleName, bkgRun=None, holderRun=None):
        super().__init__(fname, moduleName)
        if bkgRun:
            self -= bkgRun
        if holderRun:
            self -= holderRun
