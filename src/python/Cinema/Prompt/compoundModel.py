
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

from ..Interface import *
from ctypes import Structure, c_int32, c_uint64



_pt_makeCompoundModel = importFunc('pt_makeCompoundModel', type_voidp, [type_cstr] )
_pt_deleteCompoundModel = importFunc('pt_deleteCompoundModel', None, [type_voidp] )
_pt_CompoundModel_getxs = importFunc('pt_CompoundModel_getxs', type_dbl, [type_voidp, type_dbl] )

class CompoundModel:
    def __init__(self, cfg) -> None:
        self.cobj = _pt_makeCompoundModel(cfg.encode('utf-8')) 

    def __del__(self):
        _pt_deleteCompoundModel(self.cobj)
    
    def xs(self, ekin):
        return _pt_CompoundModel_getxs(self.cobj, ekin)
