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
from .Mesh import _pt_Transformation3D_transform

__all__ = ['Box', 'Volume', 'Transformation3D']

#box
_pt_Box_new = importFunc('pt_Box_new', type_voidp, [type_dbl, type_dbl, type_dbl])
_pt_Box_delete = importFunc('pt_Box_delete', None, [type_voidp] )


#Tessellated
_pt_Tessellated_new = importFunc('pt_Tessellated_new', type_voidp, [type_sizet, type_npint641d, type_npsbl2d] )


#Volume
_pt_Volume_new = importFunc('pt_Volume_new', type_voidp, [type_cstr, type_voidp])
_pt_Volume_delete = importFunc('pt_Volume_delete', None, [type_voidp] )
_pt_Volume_placeChild = importFunc('pt_Volume_placeChild', None, [type_voidp, type_cstr, type_voidp, type_voidp, type_int])

_pt_Volume_id = importFunc('pt_Volume_id', type_uint, [type_voidp])

_pt_Transformation3D_newfromdata = importFunc('pt_Transformation3D_newfromdata', type_voidp, [type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl])
_pt_Transformation3D_delete = importFunc('pt_Transformation3D_delete', None, [type_voidp] )


#resource manager 
_pt_ResourceManager_addNewVolume = importFunc('pt_ResourceManager_addNewVolume', None, [type_uint])
_pt_ResourceManager_addScorer = importFunc('pt_ResourceManager_addScorer', None, [type_uint, type_cstr])
_pt_ResourceManager_addSurface = importFunc('pt_ResourceManager_addSurface', None, [type_uint, type_cstr])
_pt_ResourceManager_addPhysics = importFunc('pt_ResourceManager_addPhysics', None, [type_uint, type_cstr])


class Box:
    def __init__(self, hx, hy, hz):
        self.cobj = _pt_Box_new(hx, hy, hz)

    # the memory should be managed by the Volume. 
    # fixme: double check if it is release at the very end
    def __del__(self):
        # _pt_Box_delete(self.cobj)
        pass

class Tessellated:
    def __init__(self, faces, points, tranMat=None) -> None:
        if tranMat is not None:
            tranMat.transformInplace(points)
        self.cobj = _pt_Tessellated_new(faces.shape[0], faces, points)

class Transformation3D:
    def __init__(self, x, y, z, phi=0, theta=0, psi=0, sx=1., sy=1., sz=1.):
        # RScale followed by rotation followed by translation.
        self.cobj = _pt_Transformation3D_newfromdata(x, y, z, phi, theta, psi, sx, sy, sz)

    def __del__(self):
        _pt_Transformation3D_delete(self.cobj)

    def transformInplace(self, input):
        _pt_Transformation3D_transform(self.cobj, input.shape[0], input, input)
        return input
        
class Volume:
    def __init__(self, volname, solid, matCfg=None, scorerCfg=None, surfaceCfg=None):
        self.child = []
        self.cobj = _pt_Volume_new(volname.encode('utf-8'), solid.cobj)
        volid = self.getLogicalID(self.cobj)

        _pt_ResourceManager_addNewVolume(volid)
        
        if matCfg is None:
            _pt_ResourceManager_addPhysics(volid, "freegas::H1/1e-26kgm3".encode('utf-8')) # set as the universe
        else:
            _pt_ResourceManager_addPhysics(volid, matCfg.encode('utf-8')) 

        if scorerCfg is not None:
            if isinstance(scorerCfg, list):
                for cfg in scorerCfg:
                    _pt_ResourceManager_addScorer(volid, cfg.encode('utf-8')) 
            else:
                _pt_ResourceManager_addScorer(volid, scorerCfg.encode('utf-8')) 

        if surfaceCfg is not None:
            _pt_ResourceManager_addSurface(volid, surfaceCfg.encode('utf-8')) 

    def __del__(self):
        # the memory should be managed by the Volume. 
        # otherwise the code will give the warning message:
        #    ""deregistering an object from GeoManager while geometry is closed""
        # _pt_Volume_delete(self.cobj)
        pass

    def placeChild(self, name, logVolume, transf=Transformation3D(0,0,0), scorerGroup=0):
        self.child.append(logVolume)
        _pt_Volume_placeChild(self.cobj, name.encode('utf-8'), logVolume.cobj, transf.cobj, scorerGroup)

    def getLogicalID(self, cobj=None):
        if cobj is None: # reutrn the ID of this volume
            return _pt_Volume_id(self.cobj)
        else:
            return _pt_Volume_id(cobj)


