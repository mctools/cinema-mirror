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

import numpy as np
from scipy.spatial import KDTree
import glob, os
from ..Math.Hist import Hist1D, Hist2D

class DD():
    def __init__(self, xmin=0., xmax=60., xbin=300):
        self.dd = {}
        self.xmin = xmin
        self.xmax = xmax
        self.xbin = xbin

    def get(self, key1, key2):
        if self.dd.get(key1) is None:
            return None
        return self.dd.get(key1).get(key2)

    def set(self, key1, key2, data):
        if self.dd.get(key1) is None:
            self.dd[key1] = {}
            hist = Hist1D(self.xmin, self.xmax, self.xbin)
            hist.fill(data)
            self.dd[key1][key2] = hist
        elif self.dd.get(key1).get(key2) is None:
            hist = Hist1D(self.xmin, self.xmax, self.xbin)
            hist.fill(data)
            self.dd[key1][key2] = hist
        else:
            self.dd[key1][key2].fill(data)

class PixelLocator(KDTree):
    def __init__(self, pixelID, location, tolerence=None):
        super().__init__(location)
        self.pixelID = pixelID
        self.tolerence = tolerence

    def locate(self, locations, numNearestPt=1):
        dist, idx = self.query(locations, k=numNearestPt)
        #fixme: print warnings or error if dist is
        return self.pixelID[idx], dist

    def generateDD(self, fileNamewild, tofbinwidth=8):
        dd = DD()
        for fileName in glob.glob(fileNamewild):
            tof_us = np.loadtxt(fileName, usecols=(0))
            tof = (tof_us/tofbinwidth).astype(int)
            pos = np.loadtxt(fileName, usecols=(1,2,3))*1e-3
            pid, dist = self.locate(pos)
            pididx = pid - self.pixelID[0]
            #  4:Qe, Qt, ekin_tof, ekin0,  ekin, wgt
            data = np.loadtxt(fileName, usecols=(5))
            for p, t, d in zip(pid, tof, data):
                dd.set(p, t, d)
        return dd

    def processHitMat(self, fileNamewild, Tofdensity, tofbinwidth=8):
        hist1dqe = Hist1D(0,70,500)
        hist1dqt = Hist1D(0,70,500)

        for fileName in glob.glob(fileNamewild):
            tof_us = np.loadtxt(fileName, usecols=(0))
            tof = (tof_us/tofbinwidth).astype(int)
            pos = np.loadtxt(fileName, usecols=(1,2,3))*1e-3
            pid, dist = self.locate(pos)
            pididx = pid - self.pixelID[0]

            print(pididx, tof)
            tof[np.where(tof>4999)]=4999 #fixme
            w = Tofdensity[pididx, tof]
            w[np.where(w<0.)]=0. #fixme
            print(w)
            print(pididx.min(), pididx.max(), tof.min(), tof.max())
            #  Qe, Qt, ekin_tof, ekin0,  ekin, wgt
            Qe = np.loadtxt(fileName, usecols=(4))
            hist1dqe.fillmany(Qe, w)

            Qt = np.loadtxt(fileName, usecols=(5))
            hist1dqt.fillmany(Qt, w)

            # Qt = np.loadtxt(fileName, usecols=(5))
            # hist2d_qt.fillmany(pid, tof, Qt)

        hist1dqe.plot()
        hist1dqt.plot(True)
        return hist1dqe, hist1dqt

    # def getModuleDensity(self, fileNamewild, tofbinwidth=8):
    #     hist2d_qt = Hist2D(self.pixelID[0]+1-0.5, self.pixelID[-1]+0.5, self.pixelID.size, 0, 40000, 40000//tofbinwidth)
    #     hist2d_qe = Hist2D(self.pixelID[0]+1-0.5, self.pixelID[-1]+0.5, self.pixelID.size, 0, 40000, 40000//tofbinwidth)
    #
    #     for fileName in glob.glob(fileNamewild):
    #         print(f'getModuleDensity wgt {fileName}')
    #         tof = np.loadtxt(fileName, usecols=(0))
    #         pos = np.loadtxt(fileName, usecols=(1,2,3))*1e-3
    #         pid, dist = self.locate(pos)
    #         pid = pid* 1.0
    #         #  Qe, Qt, ekin_tof, ekin0,  ekin, wgt
    #         Qe = np.loadtxt(fileName, usecols=(4))
    #         hist2d_qe.fillmany(pid, tof, Qe)
    #
    #         Qt = np.loadtxt(fileName, usecols=(5))
    #         hist2d_qt.fillmany(pid, tof, Qt)
    #
    #     qedensity = hist2d_qe.getDensity()
    #     qtdensity = hist2d_qt.getDensity()
    #     return qedensity, qtdensity

    def readWgtFile(self, fileNamewild, tofbinwidth=8):
        hist1d = Hist1D(0,70,500)
        hist1d2 = Hist1D(0,70,500)

        for fileName in glob.glob(fileNamewild):
            print(f'wgt {fileName}')
            tof = np.loadtxt(fileName, usecols=(0))
            pos = np.loadtxt(fileName, usecols=(1,2,3))*1e-3
            pid, dist = self.locate(pos)
            Qe = np.loadtxt(fileName, usecols=(4))
            hist1d.fillmany(Qe)

            Qt = np.loadtxt(fileName, usecols=(5))
            hist1d2.fillmany(Qt)

        qesq = hist1d.getWeight()
        qtsq = hist1d2.getWeight()
        hist1d.plot()
        hist1d2.plot(True)
        return hist1d.getCentre(), np.divide(qtsq, qesq, where=(qesq!=0))

class IDFLoader():
    def __init__(self, dir):
        self.idf = {}
        for file in glob.glob(dir+'/*.txt'):
            pid = np.loadtxt(file, dtype=int, usecols=(0))
            loc = np.loadtxt(file, dtype=float, usecols=(1,2,3))
            basename =  os.path.basename(file)
            self.idf[basename[:-4]] = PixelLocator(pid, loc)
            print(f'IDFLoader loaded file {basename}, contains {pid.size} pixels')

    def query(self, key):
        return self.idf[key]
