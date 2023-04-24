
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

class Scorer():
    def __init__(self) -> None:
        self.para = {}

    def makeCfg(self):     
        cfg = ''
        for k, v in self.__dict__.items():
            if k.__contains__('cfg_'):
                cfg += k.replace('cfg_','') 
                cfg += '='
                cfg += str(v)
                cfg += ';'
        return cfg

class PSD(Scorer):
    def __init__(self) -> None:
        super().__init__()
        self.cfg_name = 'PSD'
        self.cfg_xmin = -1.
        self.cfg_xmax = 1.
        self.cfg_numbin_x = 10 
        self.cfg_ymin = -1.
        self.cfg_ymax = 1.
        self.cfg_numbin_y = 10 
        self.cfg_ptstate = 'SURFACE' 
        self.cfg_type = 'XY'


    def makeCfg(self):
        cfg = 'Scorer=PSD;'
        return cfg + super().makeCfg()
        
