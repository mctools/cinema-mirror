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

from Interface import *

#!/usr/bin/env python3
import numpy as np
import h5py

def readKeys(content, file):
    try:
        for key in file.keys():
            content.append(file[key].name)
            subfile=file.get(file[key].name)
            readKeys(content,subfile)
    except AttributeError as e:
        print(e)


class DataLoader():
    def __init__(self, fname, moduleName):
        hf=h5py.File(fname,'r')
        self.tof = hf[f'/csns/instrument/{moduleName}/time_of_flight'][()] #vector
        print(self.tof)
        self.pid = hf[f'/csns/instrument/{moduleName}/pixel_id'][()] #vector
        self.tofpidMat = hf[f'/csns/instrument/{moduleName}/histogram_data'][()] #matrix
        self.tofMonitor = hf[f'/csns/histogram_data/monitor01/histogram_data'][()]  #vector or matrix
        self.protonPulse = 1 #vector
        self.protonCharge = 1 #vector
        self.distMod2Monitor = 1 #vector
        self.distMod2Sample =1 #double

        # file_content=[]
        # hf=h5py.File(fname,'r')
        # readKeys(file_content, hf)
        # for var in file_content:
        #     print(var)
        hf.close()

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
        return self

    # -= operator
    def __isub__(self, other):
        self.compatible(other)
        self.tofpidMat -= other.tofpidMat
        self.tofMonitor -= other.tofMonitor
        self.protonCharge -= other.protonCharge
        return self

    def divide(self, other):
        self.compatible(other)
        self.tofpidMat = np.divide(self.tofpidMat, other.tofpidMat, where=(other.tofpidMat!=0))
        self.tofMonitor = np.divide(self.tofMonitor, other.tofMonitor, where=(other.tofMonitor!=0))
        self.protonCharge = np.divide(self.protonCharge, other.protonCharge, where=(other.protonCharge!=0))

    def scale(self, factor):
        self.tofpidMat *= factor
        self.tofMonitor *= factor
        self.protonCharge *= factor
