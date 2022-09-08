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

# void* pt_Transformation3D_new(void *trfm3Dobj);
# void pt_Transformation3D_delete(void *trfm3Dobj);
# void pt_Transformation3D_multiple(void *trfm3Dobj1, void *trfm3Dobj2);
# void pt_Transformation3D_transform(void *trfm3Dobj1, size_t numPt, double *in, double *out);

_pt_Transformation3D_new = importFunc('pt_Transformation3D_new', type_voidp, [type_voidp])
_pt_Transformation3D_newfromID = importFunc('pt_Transformation3D_newfromID', type_voidp, [type_uint])
_pt_Transformation3D_delete = importFunc('pt_Transformation3D_delete', None, [type_voidp])
_pt_Transformation3D_multiple = importFunc('pt_Transformation3D_multiple', None, [type_voidp, type_voidp])
_pt_Transformation3D_transform = importFunc('pt_Transformation3D_transform', None, [type_voidp, type_sizet, type_npdbl2d, type_npdbl2d])
_pt_Transformation3D_print = importFunc('pt_Transformation3D_print', ctypes.c_char_p, [type_voidp])



class MeshHelper(object):
    def __init__(self, id):
        self.cobj = _pt_Transformation3D_newfromID(id)
        print(f'Created Transfromation {self.print()}') #fixme

    def __del__(self):
        _pt_Transformation3D_delete(self.cobj)

    def multiple(self, cobjmatrix):
        _pt_Transformation3D_multiple(self.cobj, cobjmatrix)

    def tansform(self, input):
        out = np.zeros_like(input, dtype=float)
        _pt_Transformation3D_transform(self.cobj, input.shape[0], input, out)
        return out

    def print(self):
        return _pt_Transformation3D_print(self.cobj).decode('utf-8')

_pt_placedVolNum = importFunc('pt_placedVolNum', type_sizet, [])
_pt_printMesh = importFunc("pt_printMesh", type_voidp, [])
_pt_meshInfo = importFunc("pt_meshInfo", None,  [type_sizet, type_sizet, type_sizetp, type_sizetp, type_sizetp])
_pt_getMeshName = importFunc("pt_getMeshName", type_cstr,  [type_sizet])
_pt_getMesh = importFunc("pt_getMesh", None,  [type_sizet, type_sizet, type_npdbl2d, type_npszt1d, type_npszt1d])

_pt_numDaughters = importFunc("pt_numDaughters", type_sizet, [type_sizet])
_pt_getDaughterID = importFunc("pt_getDaughterID", None,  [type_sizet, type_sizet, type_npuint1d, type_npuint1d])

class Mesh():
    def __init__(self):
        self.nMax=self.placedVolNum()
        self.n = 0

        daughter2Mother = {}
        id2tMat = {}
        #iterate through all physical volumes
        for id in range(self.nMax):
            id2tMat[id] = MeshHelper(id)
            phyid, logid = self.getDaughterID(id)
            print(f'***dau of vol {id}, physical daughter {phyid}, corrispending logical ID {logid}')
            for dautID in phyid:
                if daughter2Mother.get(dautID):
                    daughter2Mother[dautID].append(id)
                else:
                    daughter2Mother[dautID] = [id]
        print(daughter2Mother)

        #a play around :)
        matrix = id2tMat[3]
        loc = np.array([[0,0,0],[0,0.,0]])
        output = matrix.tansform(loc)
        print(output)

        matrix.multiple(matrix.cobj)
        output = matrix.tansform(loc)
        print(output)

        # stop the shit
        # while True:
        #     pass

    def placedVolNum(self):
        return _pt_placedVolNum()

    def getDaughterID(self, volid):
        num = _pt_numDaughters(volid)
        dauphy = np.zeros(num, dtype='uint32')
        daulog = np.zeros(num, dtype='uint32')
        _pt_getDaughterID(volid, num, dauphy, daulog)
        return dauphy, daulog

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
