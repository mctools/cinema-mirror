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

from scipy.spatial.transform import Rotation as scipyRot
from ..Interface import *
from .Mesh import _pt_Transformation3D_transform

__all__ = ['Volume', 'Transformation3D']


#Volume
_pt_Volume_new = importFunc('pt_Volume_new', type_voidp, [type_cstr, type_voidp])
_pt_Volume_delete = importFunc('pt_Volume_delete', None, [type_voidp] )
_pt_Volume_placeChild = importFunc('pt_Volume_placeChild', None, [type_voidp, type_cstr, type_voidp, type_voidp, type_int])

_pt_Volume_id = importFunc('pt_Volume_id', type_uint, [type_voidp])

_pt_Transformation3D_newfromdata = importFunc('pt_Transformation3D_newfromdata', type_voidp, [type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl])
_pt_Transformation3D_delete = importFunc('pt_Transformation3D_delete', None, [type_voidp] )
_pt_Transformlation3D_setRotation  = importFunc('pt_Transformlation3D_setRotation', None, [type_voidp, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl] )

#resource manager 
_pt_ResourceManager_addNewVolume = importFunc('pt_ResourceManager_addNewVolume', None, [type_uint])
_pt_ResourceManager_addScorer = importFunc('pt_ResourceManager_addScorer', None, [type_uint, type_cstr])
_pt_ResourceManager_addSurface = importFunc('pt_ResourceManager_addSurface', None, [type_uint, type_cstr])
_pt_ResourceManager_addPhysics = importFunc('pt_ResourceManager_addPhysics', None, [type_uint, type_cstr])


class Transformation3D:
    def __init__(self, x=0., y=0., z=0., rot_z=0., rot_new_y=0., rot_new_z=0., sx=1., sy=1., sz=1.):
        # RScale followed by rotation followed by translation.
        self.cobj = _pt_Transformation3D_newfromdata(x, y, z, rot_z, rot_new_y, rot_new_z, sx, sy, sz)
        r = scipyRot.from_euler('zyz', [rot_z, rot_new_y, -rot_new_z], degrees=True)
        print(f'python matrix {rot_z, rot_new_y, rot_new_z},  {r.as_matrix()} \n')


    def __del__(self):
        _pt_Transformation3D_delete(self.cobj)

    def _setRot(self, rot : scipyRot):
        mat = rot.as_matrix()
        _pt_Transformlation3D_setRotation(self.cobj, mat[0,0], mat[0,1], mat[0,2],
                                          mat[1,0], mat[1,1], mat[1,2],
                                          mat[2,0], mat[2,1], mat[2,2])
        
    def rotAxis(self, angle, axis, degrees=True):
        self._setRot(scipyRot.from_rotvec(angle * axis/np.linalg.norm(axis), degrees=degrees))
        return self
        
    #  a wrapper of scipy.spatial.transform.Rotation    
    def setRot_from_quau(self, ):
        pass

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


