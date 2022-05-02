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

_pt_placedVolNum = importFunc('pt_placedVolNum', type_sizet, [])
_pt_printMesh = importFunc("pt_printMesh", type_voidp, [])
_pt_meshInfo = importFunc("pt_meshInfo", None,  [type_sizet, type_sizet, type_sizetp, type_sizetp, type_sizetp])
_pt_getMeshName = importFunc("pt_getMeshName", type_cstr,  [type_sizet])
_pt_getMesh = importFunc("pt_getMesh", None,  [type_sizet, type_sizet, type_npdbl2d, type_npszt1d, type_npszt1d])

class Mesh():
    def __init__(self):
        self.nMax=self.placedVolNum()
        self.n = 0

    def placedVolNum(self):
        return _pt_placedVolNum()

    def printMesh(self):
        _pt_printMesh()

    def getMeshName(self):
        return _pt_getMeshName(self.n).decode('utf-8')

    def meshInfo(self, nSegments=10):
        npoints = type_sizet()
        nPlolygen = type_sizet()
        faceSize = type_sizet()
        npoints.value = 0
        nPlolygen.value = 0
        faceSize.value = 0
        _pt_meshInfo(self.n, nSegments, ctypes.byref(npoints), ctypes.byref(nPlolygen), ctypes.byref(faceSize))
        return self.getMeshName(), npoints.value, nPlolygen.value, faceSize.value

    def getMesh(self, nSegments=10):
        name, npoints, nPlolygen, faceSize = self.meshInfo(nSegments)
        if npoints==0:
            return name, np.array([]), np.array([])
        vert = np.zeros([npoints, 3], dtype=float)
        NumPolygonPoints = np.zeros(nPlolygen, dtype=type_sizet)
        facesVec = np.zeros(faceSize+nPlolygen, dtype=type_sizet)
        _pt_getMesh(self.n, nSegments, vert, NumPolygonPoints, facesVec)

        return name, vert, facesVec


    def __iter__(self):
        self.n = -1
        return self

    def __next__(self):
        if self.n < self.nMax-1:
            self.n += 1
            return self
        else:
            raise StopIteration
