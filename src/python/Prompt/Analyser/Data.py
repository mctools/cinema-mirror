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
    def __init__(self, normMethod = Normalise.byMonitor):
        super().__init__()
        self.normalise(normMethod)

    def normalise(self, normMethod):
        if normMethod == Normalise.skip:
            pass

        elif normMethod == Normalise.byMonitor:
            totMonitor  = self.tofMonitor.sum()
            if totMonitor == 0:
                raise RunTimeError('Monitor count is zero')
            self.tofpidMat /= totMonitor

        elif normMethod == Normalise.byProtonCharge:
            totCharge = self.protonCharge.sum()
            if totCharge == 0:
                raise RunTimeError('Proton charge is zero')

        else:
            raise RunTimeError('Unknown normalise method')

    def compatible(self, other):
        np.testing.assert_almost_equal(self.tof, other.tof)
        np.testing.assert_almost_equal(self.pid, other.pid)
        np.testing.assert_almost_equal(self.protonPulse, other.protonPulse)
        np.testing.assert_almost_equal(self.distMod2Monitor, other.distMod2Monitor)
        np.testing.assert_almost_equal(self.distMod2Sample, other.distMod2Sample)

    # += operator
    def __iadd__(self, other):
        self.compatible(other)
        self.tofpidMat += other.tofpidMat
        self.tofMonitor += other.tofMonitor
        self.protonCharge += other.protonCharge

    # -= operator
    def __isub__(self, other):
        self.compatible(other)
        self.tofpidMat -= other.tofpidMat
        self.tofMonitor -= other.tofMonitor
        self.protonCharge -= other.protonCharge

    # /= operator
    def __idiv__(self, other):
        self.compatible(other)
        self.tofpidMat = np.divide(self.tofpidMat, other.tofpidMat, where=(other.tofpidMat!=0))
        self.tofMonitor = np.divide(self.tofMonitor, other.tofMonitor, where=(other.tofMonitor!=0))
        self.protonCharge = np.divide(self.protonCharge, other.protonCharge, where=(other.protonCharge!=0))

    def scale(self, factor):
        self.tofpidMat *= factor
        self.tofMonitor *= factor
        self.protonCharge *= factor


class SampleData(RunData):
    def __init__(self, bkgRun=None, holderRun=None):
        super().__init__()
        if bkgRun:
            self -= bkgRun
        if holderRun:
            self -= holderRun
