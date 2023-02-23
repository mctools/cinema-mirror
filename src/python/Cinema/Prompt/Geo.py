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

_pt_setWorld = importFunc('pt_setWorld', None, [type_voidp])

_pt_UnplacedBox_new = importFunc('pt_UnplacedBox_new', type_voidp, [type_dbl, type_dbl, type_dbl])
_pt_UnplacedBox_delete = importFunc('pt_UnplacedBox_delete', None, [type_voidp] )

_pt_LogicalVolume_new = importFunc('pt_LogicalVolume_new', type_voidp, [type_cstr, type_voidp])
_pt_LogicalVolume_delete = importFunc('pt_LogicalVolume_delete', None, [type_voidp] )
_pt_LogicalVolume_placeDaughter = importFunc('pt_LogicalVolume_placeDaughter', None, [type_voidp, type_cstr, type_voidp, type_voidp])

_pt_Transformation3D_newfromdata = importFunc('pt_Transformation3D_newfromdata', type_voidp, [type_dbl, type_dbl, type_dbl, type_dbl, type_dbl, type_dbl])
_pt_Transformation3D_delete = importFunc('pt_Transformation3D_delete', None, [type_voidp] )


def setWorld(logicalvol):
    _pt_setWorld(logicalvol.cobj)


class UnplacedBox:
    def __init__(self, hx, hy, hz):
        self.cobj = _pt_UnplacedBox_new(hx, hy, hz)

    # the memory should be managed by vecgeom. 
    # fixme: double check if it is release at the very end
    def __del__(self):
        # _pt_UnplacedBox_delete(self.cobj)
        pass

class LogicalVolume:
    def __init__(self, volname, unplacedvolume):
        self.reflist = []
        self.reflist.append(unplacedvolume)
        self.cobj = _pt_LogicalVolume_new(volname.encode('utf-8'), unplacedvolume.cobj)

    def __del__(self):
        # the memory should be managed by vecgeom. 
        # fixme: double check if it is release at the very end
        # _pt_LogicalVolume_delete(self.cobj)
        pass

    def placeDaughter(self, name, unplacedVolume, transf):
        self.reflist.append(unplacedVolume)
        _pt_LogicalVolume_placeDaughter(self.cobj, name.encode('utf-8'), unplacedVolume.cobj, transf.cobj)


class Transformation3D:
    def __init__(self, x, y, z, phi, theta, psi):
        self.cobj = _pt_Transformation3D_newfromdata(x, y, z, phi, theta, psi)

    def __del__(self):
        _pt_Transformation3D_delete(self.cobj)