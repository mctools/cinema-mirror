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
import numpy as np
from ..Interface import *

#box
_pt_Box_new = importFunc('pt_Box_new', type_voidp, [type_dbl, type_dbl, type_dbl])
_pt_Box_delete = importFunc('pt_Box_delete', None, [type_voidp] )
_pt_Tube_new = importFunc('pt_Tube_new', type_voidp, [type_dbl, type_dbl, type_dbl, type_dbl, type_dbl] )
_pt_Trapezoid_new = importFunc('pt_Trapezoid_new', type_voidp, [type_dbl, type_dbl, type_dbl, type_dbl, type_dbl] )
_pt_Sphere_new = importFunc('pt_Sphere_new', type_voidp, [type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl] )
_pt_Polyhedron_new = importFunc('pt_Polyhedron_new', type_voidp, [type_dbl, type_dbl, type_int, type_int, type_npdbl1d, type_npdbl1d, type_npdbl1d] )
_pt_ArbTrapezoid_new = importFunc('pt_ArbTrapezoid_new', type_voidp, [type_dblp, type_dblp, type_dblp, type_dblp, type_dblp, type_dblp, type_dblp, type_dblp, type_dbl])

#Tessellated
_pt_Tessellated_new = importFunc('pt_Tessellated_new', type_voidp, [type_sizet, type_npint641d, type_npsbl2d] )


class Solid:
    def __init__(self) -> None:
        pass

class Box(Solid):
    def __init__(self, hx, hy, hz):
        super().__init__()
        self.cobj = _pt_Box_new(hx, hy, hz)
        self.hx = hx
        self.hy = hy
        self.hz = hz

    # the memory should be managed by the Volume. 
    # fixme: double check if it is release at the very end
    def __del__(self):
        # _pt_Box_delete(self.cobj)
        pass

class Tube(Solid):
    def __init__(self, rmin, rmax, z, startphi = 0, deltaphi = 360):
        super().__init__()
        self.cobj = _pt_Tube_new(rmin, rmax, z, np.deg2rad(startphi), np.deg2rad(deltaphi))

class Sphere(Solid):
    def __init__(self, rmin, rmax, startphi=0., deltaphi=2*np.pi, starttheta=0., deltatheta=np.pi):
        super().__init__()
        self.cobj = _pt_Sphere_new(rmin, rmax, startphi, deltaphi, starttheta, deltatheta)

class Trapezoid(Solid):
    def __init__(self, x1, x2, y1, y2, z) -> None:
        super().__init__()
        self.cobj = _pt_Trapezoid_new(x1, x2, y1, y2, z)

class Polyhedron(Solid):
    def __init__(self, zPlanes, rMin, rMax, sideCount=6,
                 phiStart_deg=0, phiDelta_deg=360) -> None:
        super().__init__()
        zp, rmin, rmax = np.array(zPlanes), np.array(rMin), np.array(rMax)
        if zp.size!=rmin.size or rmin.size!=rmax.size:
            raise RuntimeError('the sizes of zPlanes, rMin and rMax are not equal')    
        
        self.cobj = _pt_Polyhedron_new(np.deg2rad(phiStart_deg), np.deg2rad(phiDelta_deg), int(sideCount), int(zp.size), 
                 zp, rmin, rmax)

class Tessellated(Solid): #this one is not working
    def __init__(self, faces, points, tranMat=None) -> None:
        super().__init__()
        if tranMat is not None:
            tranMat.transformInplace(points)
        self.cobj = _pt_Tessellated_new(faces.shape[0], faces, points)

class ArbTrapezoid(Solid):
    def __init__(self, xy1 : np.ndarray, xy2 : np.ndarray, xy3 : np.ndarray, xy4 : np.ndarray,
                 xy5 : np.ndarray, xy6 : np.ndarray, xy7 : np.ndarray, xy8 : np.ndarray, halfz) -> None:
        super().__init__()
        # if not xy1.flags['C_CONTIGUOUS']:         TODO: solid or not when ndarray is not contiguous?
            # xy1 = np.ascontiguousarray(xy1, dtype=xy1.dtype)
        # xy1 = ctypes.cast(xy1.ctypes.data, type_dblp)
        xy1 = xy1.astype(np.double).ctypes.data_as(type_dblp)
        xy2 = xy2.astype(np.double).ctypes.data_as(type_dblp)
        xy3 = xy3.astype(np.double).ctypes.data_as(type_dblp)
        xy4 = xy4.astype(np.double).ctypes.data_as(type_dblp)
        xy5 = xy5.astype(np.double).ctypes.data_as(type_dblp)
        xy6 = xy6.astype(np.double).ctypes.data_as(type_dblp)
        xy7 = xy7.astype(np.double).ctypes.data_as(type_dblp)
        xy8 = xy8.astype(np.double).ctypes.data_as(type_dblp)

        self.cobj = _pt_ArbTrapezoid_new(xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, halfz)