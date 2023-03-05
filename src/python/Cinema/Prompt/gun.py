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

_pt_PythonGun_new = importFunc('pt_PythonGun_new', type_voidp, [type_pyobject])
_pt_PythonGun_delete = importFunc('pt_PythonGun_delete', None, [type_voidp])
_pt_PythonGun_generate = importFunc('pt_PythonGun_generate', None, [type_voidp])


class PythonGun():
    def __init__(self):
        self.cobj = _pt_PythonGun_new(self)
        
    def __del__(self):
        _pt_PythonGun_delete(self.cobj)

    # This method will be called by the c++ 
    def generate(self):
        pass
        # print('shoting!')
        # &m_ekin, &m_weight, &m_time,
        # &m_pos.x(), &m_pos.y(), &m_pos.z(),
        # &m_dir.x(), &m_dir.y(), &m_dir.z())
        return 1234., 2234.,3,4,5,6, 0,0,1
        
    def pyGenerate(self):
        _pt_PythonGun_generate(self.cobj)


    

