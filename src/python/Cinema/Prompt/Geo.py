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


_pt_UnplacedBox_new = importFunc('pt_UnplacedBox_new', type_voidp, [type_dbl, type_dbl, type_dbl])
_pt_UnplacedBox_delete = importFunc('pt_UnplacedBox_delete', None, [type_voidp] )

_pt_LogicalVolume_new = importFunc('pt_LogicalVolume_new', type_voidp, [type_cstr, type_voidp])
_pt_LogicalVolume_delete = importFunc('pt_LogicalVolume_delete', None, [type_voidp] )
_pt_LogicalVolume_placeDaughter = importFunc('pt_LogicalVolume_placeDaughter', None, [type_voidp, type_cstr, type_voidp, type_voidp])

_pt_LogicalVolume_id = importFunc('pt_LogicalVolume_id', type_uint, [type_voidp])

_pt_Transformation3D_newfromdata = importFunc('pt_Transformation3D_newfromdata', type_voidp, [type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl])
_pt_Transformation3D_delete = importFunc('pt_Transformation3D_delete', None, [type_voidp] )


#resource manager 
_pt_ResourceManager_addNewVolume = importFunc('pt_ResourceManager_addNewVolume', None, [type_uint])
_pt_ResourceManager_addScorer = importFunc('pt_ResourceManager_addScorer', None, [type_uint, type_cstr])
_pt_ResourceManager_addSurface = importFunc('pt_ResourceManager_addSurface', None, [type_uint, type_cstr])
_pt_ResourceManager_addPhysics = importFunc('pt_ResourceManager_addPhysics', None, [type_uint, type_cstr])

class UnplacedBox:
    def __init__(self, hx, hy, hz):
        self.cobj = _pt_UnplacedBox_new(hx, hy, hz)

    # the memory should be managed by vecgeom. 
    # fixme: double check if it is release at the very end
    def __del__(self):
        # _pt_UnplacedBox_delete(self.cobj)
        pass

class LogicalVolume:
    def __init__(self, volname, unplacedvolume, matCfg=None, scorerCfg=None, surfaceCfg=None):
        self.child = []
        self.cobj = _pt_LogicalVolume_new(volname.encode('utf-8'), unplacedvolume.cobj)

        volid = self.getID()

        _pt_ResourceManager_addNewVolume(volid)
        
        if matCfg is None:
            _pt_ResourceManager_addPhysics(volid, "freegas::H1/1e-26kgm3".encode('utf-8')) # set as the universe
        else:
            _pt_ResourceManager_addPhysics(volid, matCfg.encode('utf-8')) 

        if scorerCfg is not None:
            _pt_ResourceManager_addScorer(volid, scorerCfg.encode('utf-8')) 

        if surfaceCfg is not None:
            _pt_ResourceManager_addSurface(volid, surfaceCfg.encode('utf-8')) 

    def __del__(self):
        # the memory should be managed by vecgeom. 
        # fixme: double check if it is release at the very end
        # _pt_LogicalVolume_delete(self.cobj)
        pass

    def placeChild(self, name, unplacedVolume, transf):
        self.child.append(unplacedVolume)
        _pt_LogicalVolume_placeDaughter(self.cobj, name.encode('utf-8'), unplacedVolume.cobj, transf.cobj)

    def getID(self):
        return _pt_LogicalVolume_id(self.cobj)


class Transformation3D:
    def __init__(self, x, y, z, phi, theta, psi):
        self.cobj = _pt_Transformation3D_newfromdata(x, y, z, phi, theta, psi)

    def __del__(self):
        _pt_Transformation3D_delete(self.cobj)