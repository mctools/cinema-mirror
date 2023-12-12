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
from .scorer import Scorer

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
    
    @property
    def euler(self):
        return self.sciRot.as_euler('xyz', True)

    def __deepcopy__(self, memo):
        copy = type(self)()
        memo[id(self)] = copy
        copy.cobj = _pt_Transformation3D_newfromdata(self.translation[0], self.translation[1], self.translation[2], 
                                                     0, 0, 0, 1.,1.,1.)
        copy.sciRot = scipyRot.from_matrix(self.sciRotMatrix)
        copy.py2cppConv()
        copy.translation = self.translation
        copy.sciRotMatrix = self.sciRotMatrix
        return copy

    def __del__(self):
        _pt_Transformation3D_delete(self.cobj)
        
    # def __mul__(self, other):
    #     '''
    #     Transformation following another parent transformation.
    #     a dot B: a project in B, successive add up
    #     '''
    #     rot = self.getRotMatrix().dot(other.getRotMatrix())
    #     transl = self.translation + other.getTranslation().dot(self.getRotMatrix())
    #     transf = Transformation3D(transl[0], transl[1], transl[2])
    #     transf.sciRot = self.sciRot * other.sciRot
    #     transf.applyTrans(rot)
    #     return transf

    def __mul__(self, other):
        rot = self.getRotMatrix().dot(other.getRotMatrix())
        transl = self.translation + self.getRotMatrix().dot(other.getTranslation())
        transf = Transformation3D(transl[0], transl[1], transl[2])
        transf.sciRot = self.sciRot * other.sciRot
        transf.applyTrans(rot)
        return transf

    def inv(self):
        inversion = type(self)()
        inversion.translation = - self.translation
        inversion.cobj = _pt_Transformation3D_newfromdata(inversion.translation[0], 
                                                          inversion.translation[1], 
                                                          inversion.translation[2], 
                                                          0, 0, 0, 1.,1.,1.)
        inversion.sciRot = self.sciRot.inv()
        inversion.py2cppConv()
        return inversion

    def py2cppConv(self):
        mat = self.sciRot.as_matrix()
        self.sciRotMatrix = mat
        # print(mat)
        _pt_Transformlation3D_setRotation(self.cobj, mat[0,0], mat[0,1], mat[0,2],
                                          mat[1,0], mat[1,1], mat[1,2],
                                          mat[2,0], mat[2,1], mat[2,2])
        
    def applyRotAxis(self, angle, axis, degrees=True):
        axis = np.array(axis)
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

    def applyTrans(self, refMatrix):
        # self.sciRotMatrix = self.sciRot.as_matrix()
        self.sciRotMatrix = mat = self.sciRotMatrix.dot(refMatrix)
        self.sciRot = scipyRot.from_matrix(self.sciRotMatrix)
        _pt_Transformlation3D_setRotation(self.cobj, mat[0,0], mat[0,1], mat[0,2],
                                          mat[1,0], mat[1,1], mat[1,2],
                                          mat[2,0], mat[2,1], mat[2,2])

    def setRotByAlignement(self, rotated, original) :  # rotated, original are with shape (N, 3)
        self.sciRot, rssd = scipyRot.align_vectors(original, rotated)
        self.py2cppConv()
        return self

    def setSciRot(self, sciRot):
        self.sciRot = deepcopy(sciRot)
        self.py2cppConv()
        return self
    
    def getRotMatrix(self):
        return self.sciRotMatrix
        
    def getTranslation(self):
        return self.translation

    def getTransformationTo(self, other):
        return self.inv() * other


    #  a wrapper of scipy.spatial.transform.Rotation    
    def setRot(self, rot_z=0., rot_new_x=0., rot_new_z=0., degrees = True):
        self.sciRot = scipyRot.from_euler('ZXZ', [rot_z, rot_new_x, rot_new_z], degrees)
        self.py2cppConv()

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
    scorerDict = {}

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
            if isinstance(matCfg, str):
                self.setMaterial(matCfg) 
            else:
                self.setMaterial(matCfg.cfg) 

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

    def addScorer(self, scorer : Scorer or str):
        import re
        if isinstance(scorer, str):
            name = re.search(r'\s*name\s*=\s*\w*\s*;', scorer)
            name = re.search(r'=.*;', name.group())
            name = re.sub(r'[^\w]', '', name.group())
            self.__class__.scorerDict[name] = scorer
            _pt_ResourceManager_addScorer(self.volid, scorer.encode('utf-8')) 
        else:
            cfg = scorer.cfg
            self.__class__.scorerDict[scorer.cfg_name] = cfg
            _pt_ResourceManager_addScorer(self.volid, cfg.encode('utf-8')) 

    def setSurface(self, cfg : str):
        _pt_ResourceManager_addSurface(self.volid, cfg.encode('utf-8')) 

    def placeChild(self, name, logVolume, transf=Transformation3D(0,0,0), scorerGroup=0):
        self.child.append(logVolume)
        _pt_Volume_placeChild(self.cobj, name.encode('utf-8'), logVolume.cobj, transf.cobj, scorerGroup)
        return self
    
    def placeArray(self, array, transf = None, marker = '', count = 0):
        if transf == None:
            transf = array.refFrame
        marker = f'{marker}{count}'
        for i_mem in array.members:
            if isinstance(array.element, Volume):
                transf_t = transf * i_mem.refFrame 
                # transf_t.sciRot = deepcopy(transf.sciRot)
                # transf_t.applyTrans(i_mem.refFrame.sciRotMatrix)
                self.placeChild(f'phyvol_{marker}_{array.element.volname}', array.element, transf_t)
            else:
                count = count + 1
                self.placeArray(array.element, transf * i_mem.refFrame, i_mem.marker, count = count)



    def getLogicalID(self, cobj=None):
        if cobj is None: # reutrn the ID of this volume
            return _pt_Volume_id(self.cobj)
        else:
            return _pt_Volume_id(cobj)


