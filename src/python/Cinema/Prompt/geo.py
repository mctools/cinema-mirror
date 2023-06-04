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
from copy import deepcopy

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
    def __init__(self, x=0., y=0., z=0., rot_z=0., rot_new_x=0., rot_new_z=0., degrees = True):
        self.cobj = _pt_Transformation3D_newfromdata(x, y, z, 0, 0, 0, 1.,1.,1.)
        self.sciRot = scipyRot.from_euler('ZXZ', [rot_z, rot_new_x, rot_new_z], degrees)
        self.translation = np.array([x, y, z])
        self.sciRotMatrix = self.sciRot.as_matrix()

    def __deepcopy__(self, memo):
        copy = type(self)()
        memo[id(self)] = copy
        copy.cobj = _pt_Transformation3D_newfromdata(self.translation[0], self.translation[1], self.translation[2], 
                                                     0, 0, 0, 1.,1.,1.)
        copy.sciRot = scipyRot.from_matrix(self.sciRotMatrix)
        copy.py2cppConv()
        copy.translation = self.translation
        return copy

    def __del__(self):
        _pt_Transformation3D_delete(self.cobj)
        
    def __mul__(self, other):
        '''
        Transformation following another parent transformation.
        '''
        rot = self.getRot() * other.getRot()
        transl = self.translation + self.getRot().apply(other.getTranslation()) 
        transf = Transformation3D(transl[0], transl[1], transl[2])
        transf.sciRot = rot
        transf.setSciRot()
        return transf

    def py2cppConv(self):
        mat = self.sciRot.as_matrix()
        self.sciRotMatrix = mat
        # print(mat)
        _pt_Transformlation3D_setRotation(self.cobj, mat[0,0], mat[0,1], mat[0,2],
                                          mat[1,0], mat[1,1], mat[1,2],
                                          mat[2,0], mat[2,1], mat[2,2])
        
    def applyRotAxis(self, angle, axis, degrees=True):
        rot = scipyRot.from_rotvec(angle * axis/np.linalg.norm(axis), degrees=degrees)
        self.sciRot *= rot
        self.py2cppConv()
        return self
    
    def applyRotX(self, angle, degrees=True):
        rot = scipyRot.from_rotvec(angle * np.array([1,0,0.]), degrees=degrees)
        self.sciRot *= rot
        self.py2cppConv()
        return self
    
    def applyRotY(self, angle, degrees=True):
        rot = scipyRot.from_rotvec(angle * np.array([0,1,0.]), degrees=degrees)
        self.sciRot *= rot
        self.py2cppConv()
        return self
    
    def applyRotZ(self, angle, degrees=True):
        rot = scipyRot.from_rotvec(angle * np.array([0,0,1.]), degrees=degrees)
        self.sciRot *= rot
        self.py2cppConv()
        return self
    
    def applyRotxyz(self, rotx, roty, rotz, degrees=True):
        rot = scipyRot.from_euler('xyz', [rotx, roty, rotz], degrees=degrees)
        self.sciRot *= rot
        self.py2cppConv()
        return self

    def setRotByAlignement(self, rotated, original) :  # rotated, original are with shape (N, 3)
        self.sciRot, rssd = scipyRot.align_vectors(original, rotated)
        self.py2cppConv()
        return self

    def setSciRot(self):
        self.py2cppConv()
        return self
    
    def getRot(self):
        return self.sciRot
        
    def getTranslation(self):
        return self.translation

    #  a wrapper of scipy.spatial.transform.Rotation    
    def setRot(self, rot_z=0., rot_new_x=0., rot_new_z=0., degrees = True):
        self.py2cppConv(scipyRot.from_euler('ZXZ', [rot_z, rot_new_x, rot_new_z], degrees=degrees))
        return self

    def transformInplace(self, input):
        _pt_Transformation3D_transform(self.cobj, input.shape[0], input, input)
        return input

# class Transformation3D:
#     def __init__(self, x=0., y=0., z=0., rot_z=0., rot_new_x=0., rot_new_z=0.):
#         # RScale followed by rotation followed by translation.
#         self.cobj = _pt_Transformation3D_newfromdata(x, y, z, rot_z, rot_new_x, rot_new_z, 1.,1.,1.)
#         self.rotation = scipyRot.from_euler('ZXZ', [rot_z, rot_new_x, rot_new_z], degrees=True)
#         self.translation = np.array([x, y, z])

#     def __del__(self):
#         _pt_Transformation3D_delete(self.cobj)

#     def _setRot(self, rot : scipyRot):
#         mat = self.rotation_matrix = rot.as_matrix()
#         _pt_Transformlation3D_setRotation(self.cobj, mat[0,0], mat[0,1], mat[0,2],
#                                           mat[1,0], mat[1,1], mat[1,2],
#                                           mat[2,0], mat[2,1], mat[2,2])
        
#     def rotAxis(self, angle, axis, degrees=True):
#         self._setRot(scipyRot.from_rotvec(angle * axis/np.linalg.norm(axis), degrees=degrees)) # fixme: axis/np.linalg.norm(axis) seems to be broadcasted
#         return self
    
#     def rotxyz(self, rotx, roty, rotz, degrees=True):
#         self._setRot(scipyRot.from_euler('xyz', [rotx, roty, rotz], degrees=degrees))
#         return self
    
#     def rotX(self, angle, degrees=True):
#         self._setRot(scipyRot.from_rotvec(angle * np.array([1,0,0.]), degrees=degrees))
#         return self
    
#     def rotY(self, angle, degrees=True):
#         self._setRot(scipyRot.from_rotvec(angle * np.array([0,1.,0.]), degrees=degrees))
#         return self
    
#     def rotZ(self, angle, degrees=True):
#         self._setRot(scipyRot.from_rotvec(angle * np.array([0,0.,1.]), degrees=degrees))
#         return self
        
#     #  a wrapper of scipy.spatial.transform.Rotation    
#     def setRot(self, rot_z=0., rot_new_x=0., rot_new_z=0., degrees = True):
#         self._setRot(scipyRot.from_euler('ZXZ', [rot_z, rot_new_x, rot_new_z], degrees=degrees))
#         return self

#     def rotMatix(self, matrix):
#         self._setRot(scipyRot.from_matrix(matrix))
#         return self
    
#     def getRot(self):
#         return self.

#     def getRotMatix(self):
#         return self.rotation_matrix
    
#     def transformInplace(self, input):
#         _pt_Transformation3D_transform(self.cobj, input.shape[0], input, input)
#         return input
        
class Volume:
    def __init__(self, volname, solid, matCfg=None, surfaceCfg=None):
        self.volname = volname
        self.solid = solid
        self.child = []
        self.cobj = _pt_Volume_new(volname.encode('utf-8'), solid.cobj)
        self.volid = self.getLogicalID(self.cobj)
        self.matCfg = matCfg
        self.surfaceCfg = surfaceCfg

        _pt_ResourceManager_addNewVolume(self.volid)
        
        if matCfg is None:
            self.setMaterial('freegas::H1/1e-26kgm3') # set as the universe
        else:
            self.setMaterial(matCfg) 

        if surfaceCfg is not None:
            self.setSurface(surfaceCfg) 

    def __del__(self):
        # the memory should be managed by the Volume. 
        # otherwise the code will give the warning message:
        #    ""deregistering an object from GeoManager while geometry is closed""
        # _pt_Volume_delete(self.cobj)
        pass

    def setMaterial(self, cfg : str):
        _pt_ResourceManager_addPhysics(self.volid, cfg.encode('utf-8')) # set as the universe

    def addScorer(self, cfg : str):
         _pt_ResourceManager_addScorer(self.volid, cfg.encode('utf-8')) 

    def setSurface(self, cfg : str):
        _pt_ResourceManager_addSurface(self.volid, cfg.encode('utf-8')) 

    def placeChild(self, name, logVolume, transf=Transformation3D(0,0,0), scorerGroup=0):
        self.child.append(logVolume)
        _pt_Volume_placeChild(self.cobj, name.encode('utf-8'), logVolume.cobj, transf.cobj, scorerGroup)
        return self
    
    def placeArray(self, array, transf = Transformation3D(), marker = '', count = 0):
        marker = f'{marker}{count}'
        print(transf.sciRot.as_matrix())
        for i_mem in array.eleAncs:
            if isinstance(i_mem[0], Volume):
                self.placeChild(f'phyvol_{marker}_{i_mem[0].volname}', i_mem[0], transf * i_mem[1].refFrame)
            else:
                count = count + 1
                self.placeArray(i_mem[0], i_mem[1].refFrame, i_mem[1].marker, count = count)



    def getLogicalID(self, cobj=None):
        if cobj is None: # reutrn the ID of this volume
            return _pt_Volume_id(self.cobj)
        else:
            return _pt_Volume_id(cobj)


