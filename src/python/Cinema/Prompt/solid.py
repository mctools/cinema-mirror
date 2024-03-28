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
_pt_Cone_new = importFunc('pt_Cone_new', type_voidp, [type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl])
_pt_CutTube_new = importFunc('pt_CutTube_new', type_voidp, [type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dblp, type_dblp])
_pt_HypeTube_new = importFunc('pt_HypeTube_new', type_voidp, [type_dbl, type_dbl, type_dbl, type_dbl, type_dbl])

#Tessellated
_pt_Tessellated_new = importFunc('pt_Tessellated_new', type_voidp, [type_sizet, type_npint641d, type_npsbl2d] )



class Solid:
    def __init__(self) -> None:
        pass

    def sanityCheckBase(self, *args): 
        for p in args:
            if p < 0:
                raise ValueError(f"Invalid input! {p} Should be positive value or zero!")
            
    def sanityCheckRelation(self, min, max):
        if min > max:
            raise ValueError(f"Invalid inputs! rmin ({min}) should less than or equal rmax ({max}))!")

    def convert2pointer(self, p : np.ndarray):
        """
        If a pointer points to an array, its elements can be read and written using standard subscript and slice accesses
        Ref: https://docs.python.org/3/library/ctypes.html#ctypes._Pointer
        """
        p_c = p.astype(np.double).ctypes.data_as(type_dblp)
        return p_c

class Box(Solid):
    def __init__(self, hx, hy, hz):
        self.sanityCheckBase(hx, hy, hz)
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
        self.sanityCheckBase(rmin, rmax, z, deltaphi)
        self.cobj = _pt_Tube_new(rmin, rmax, z, np.deg2rad(startphi), np.deg2rad(deltaphi))

class Sphere(Solid):
    def __init__(self, rmin, rmax, startphi=0., deltaphi=2*np.pi, starttheta=0., deltatheta=np.pi):
        self.sanityCheckBase(rmin, rmax, deltaphi, deltatheta)
        self.cobj = _pt_Sphere_new(rmin, rmax, startphi, deltaphi, starttheta, deltatheta)

class Trapezoid(Solid):
    def __init__(self, x1, x2, y1, y2, z) -> None:
        self.sanityCheckBase(x1, x2, y1, y2, z)
        self.cobj = _pt_Trapezoid_new(x1, x2, y1, y2, z)

class Polyhedron(Solid):
    def __init__(self, zPlanes, rMin, rMax, sideCount=6,
                 phiStart_deg=0, phiDelta_deg=360) -> None:
        self.sanityCheckBase(zPlanes, rMin, rMax, sideCount,phiDelta_deg)
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
        self.sanityCheckBase(halfz)
        # if not xy1.flags['C_CONTIGUOUS']:         TODO: solid or not when ndarray is not contiguous?
            # xy1 = np.ascontiguousarray(xy1, dtype=xy1.dtype)
        # xy1 = ctypes.cast(xy1.ctypes.data, type_dblp)
        vectors = (xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8)
        p_vecs = []
        for vec in vectors:
            p_vecs.append(self.convert2pointer(vec))
        self.cobj = _pt_ArbTrapezoid_new(p_vecs[0], p_vecs[1], p_vecs[2], p_vecs[3], p_vecs[4], p_vecs[5], p_vecs[6], p_vecs[7], halfz)

class Cone(Solid):
    def __init__(self, rmaxBot, rmaxTop, z, rminBot = 0, rminTop = 0, startPhi = 0, deltaPhi = 360) -> None:
        self.sanityCheckBase(rmaxBot, rmaxTop, z, rminBot, rminTop, deltaPhi)
        self.sanityCheckRelation(rminBot, rmaxBot)
        self.sanityCheckRelation(rminTop, rmaxTop)
        self.cobj = _pt_Cone_new(rminBot, rmaxBot, rminTop, rmaxTop, z, np.deg2rad(startPhi), np.deg2rad(deltaPhi))
       

class CutTube(Solid):
    def __init__(self, rmax, halfHeight, botNormal, topNormal, rmin = 0, sphi = 0, dphi = 360) -> None:
        raise NotImplementedError("CutTube got problems, See issue!")
        # TODO:fix tracing point location problem
        # super().__init__()
        # self.sanityCheckBase(rmin, rmax, halfHeight, dphi)
        # botN = self.convert2pointer(botNormal)
        # topN = self.convert2pointer(topNormal)
        # self.sanityCheck(rmin, rmax)
        # self.cobj = _pt_CutTube_new(rmin, rmax, halfHeight, np.deg2rad(sphi), np.deg2rad(dphi), botN, topN)
        
class HypebolicTube(Solid):
    def __init__(self, rmax, inst, outst, halfHeight, rmin = 0) -> None:
        super().__init__()
        self.sanityCheckBase(rmin, rmax, inst, outst, halfHeight)
        self.sanityCheckRelation(rmin, rmax)
        self.cobj = _pt_HypeTube_new(rmin, rmax, inst, outst, halfHeight)
