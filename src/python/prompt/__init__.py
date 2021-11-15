#!/usr/bin/env python3

import ctypes
import numpy as np
import glob
import os
import time

_sizet, _sizetp = (ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t))
_int,_intp,_uint,_uintp,_dbl,_dblp,_cstr,_voidp = (ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                                   ctypes.c_uint,ctypes.POINTER(ctypes.c_uint), ctypes.c_double,
                                                   ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_void_p)
_npdbl1d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
_npdbl2d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS')
_npdbl3d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='C_CONTIGUOUS')
_npszt1d=np.ctypeslib.ndpointer(dtype=ctypes.c_size_t,ndim=1,flags='C_CONTIGUOUS')

def _getPromptLib():
    _ptpath = os.getenv('PTPATH')
    if _ptpath is None:
        raise IOError('PTPATH enviroment is not set')
    libfile = glob.glob(_ptpath +'/promptbin/src/cxx/libprompt_core.so')[0]
    return ctypes.CDLL(libfile), _ptpath

_taklib, _ptpath = _getPromptLib()

def _findData(fileName):
    pass

def _cFunc(funcName, resType, argType):
    func = getattr(_taklib, funcName)
    func.restype = resType
    func.argtypes = argType
    return func

_pt_placedVolNum = _cFunc('pt_placedVolNum', _sizet, [])
_pt_printMesh = _cFunc("pt_printMesh", _voidp, [])
_pt_meshInfo = _cFunc("pt_meshInfo", None,  [_sizet, _sizet, _sizetp, _sizetp, _sizetp])
_pt_getMeshName = _cFunc("pt_getMeshName", _cstr,  [_sizet])
_pt_getMesh = _cFunc("pt_getMesh", None,  [_sizet, _sizet, _npdbl2d, _npszt1d, _npszt1d])

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
        npoints = _sizet()
        nPlolygen = _sizet()
        faceSize = _sizet()
        npoints.value = 0
        nPlolygen.value = 0
        faceSize.value = 0
        _pt_meshInfo(self.n, nSegments, ctypes.byref(npoints), ctypes.byref(nPlolygen), ctypes.byref(faceSize))
        return self.getMeshName(), npoints.value, nPlolygen.value, faceSize.value

    def getMesh(self, nSegments=10):
        name, npoints, nPlolygen, faceSize = self.meshInfo(nSegments)
        vert = np.zeros([npoints, 3], dtype=float)
        NumPolygonPoints = np.zeros(nPlolygen, dtype=_sizet)
        facesVec = np.zeros(faceSize, dtype=_sizet)
        _pt_getMesh(self.n, nSegments, vert, NumPolygonPoints, facesVec)

        faces=[]
        curPos=0
        for numPoints in NumPolygonPoints:
            curPos += int(numPoints)
            faces.append(np.array(facesVec[int(curPos-numPoints):curPos], dtype=int))
        return name, vert, faces


    def __iter__(self):
        self.n = -1
        return self

    def __next__(self):
        if self.n < self.nMax-1:
            self.n += 1
            return self
        else:
            raise StopIteration


_pt_Launcher_getInstance = _cFunc('pt_Launcher_getInstance', _voidp, [] )
_pt_Launcher_setSeed = _cFunc('pt_Launcher_setSeed', None, [_voidp, _sizet] )
_pt_Launcher_loadGeometry = _cFunc('pt_Launcher_loadGeometry', None, [_voidp, _cstr] )

class Launcher():
    def __init__(self):
        self.cobj = _pt_Launcher_getInstance()

    def setSeed(seed):
        _pt_Launcher_setSeed(self.cobj, seed)

    def loadGeometry(self, fileName):
        _pt_Launcher_loadGeometry(self.cobj, fileName.encode('utf-8'));


import pyvista as pv
import trimesh
import random
class Visualiser():
    def __init__(self, printWorld=False):
        self.plotter = pv.Plotter()
        self.worldMesh = Mesh()
        if printWorld:
            self.worldMesh.printMesh()
        self.loadMesh()

    def loadMesh(self):
        for am in self.worldMesh:
            name = am.getMeshName()
            print(f'loading mesh {name}')
            if name!='World':
                name, points, faces = am.getMesh(100)
                rcolor = random.choice(['red', 'grey', 'yellow', 'blue', 'black'])

                face3p = []
                face4p = []
                for face in faces:
                    if face.size == 3:
                        face3p.append(face)
                    elif face.size == 4:
                        face4p.append(face)
                rcolor = np.random.random(3)
                if len(face4p) !=0 :
                    tmesh4 = pv.wrap(trimesh.Trimesh(points, faces=face4p, process=False))
                    self.plotter.add_mesh(tmesh4, color=rcolor, opacity=0.3)

                if len(face3p) !=0 :
                    tmesh3 = pv.wrap(trimesh.Trimesh(points, faces=face3p, process=False))
                    self.plotter.add_mesh(tmesh3, color=rcolor, opacity=0.3)

    def show(self):
        self.plotter.show()
