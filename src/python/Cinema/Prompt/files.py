
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

_pt_MCPLBinaryWrite_new = importFunc('pt_MCPLBinaryWrite_new', type_voidp, [type_cstr, type_bool, type_bool, type_bool] )
_pt_MCPLBinaryWrite_delete = importFunc('pt_MCPLBinaryWrite_delete', type_voidp, [type_voidp] )
_pt_MCPLBinaryWrite_write = importFunc('pt_MCPLBinaryWrite_write', type_voidp, [type_voidp, type_voidp] )


class MCPLBinaryWrite:
    def __init__(self, fn, enable_double=False, enable_extra3double=False, 
                enable_extraUnsigned=False) -> None:
        self.cobj = _pt_MCPLBinaryWrite_new(fn, enable_double, enable_extra3double, enable_extraUnsigned )

    def __del__(self):
        _pt_MCPLBinaryWrite_delete(self.cobj)
    
    def write(self, par):
        _pt_MCPLBinaryWrite_write(self.cobj, par)