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
import glob, os

def readKeys(content, file):
    try:
        for key in file.keys():
            content.append(file[key].name)
            subfile=file.get(file[key].name)
            readKeys(content,subfile)
    except AttributeError as e:
        print(e)


class IDFLoader():
    def __init__(self, dir):
        self.dict = {}
        for file in glob.glob(dir+'/*.txt'):
            self.dict[os.path.basename(file)] = np.loadtxt(file)


class DataLoader():
    def __init__(self, fname):
        self.tof = 1 #vector
        self.pid = 1 #vector
        self.tofpidMat = 1 #matrix
        self.tofMonitor = 1  #vector or matrix
        self.protonPulse = 1 #vector
        self.protonCharge = 1 #vector
        self.distMod2Monitor = 1 #vector
        self.distMod2Sample =1 #double

        file_content=[]
        hf=h5py.File(fname,'r')
        readKeys(file_content, hf)
        for var in file_content:
            print(var)

        module10203_hist  =  hf['/csns/instrument/module10203/histogram_data'][()]
        module10203_pixel =  hf['/csns/instrument/module10203/pixel_id'][()]
        module10203_tof   =  hf['/csns/instrument/module10203/time_of_flight'][()]
        print(module10203_hist.shape, module10203_pixel.shape, module10203_tof.shape)
        hf.close()
