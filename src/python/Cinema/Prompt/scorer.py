
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
        for k, v in self.para.items():
            cfg += k 
            cfg += '='
            cfg += str(v)
            cfg += ';'
        return cfg

class PSD(Scorer):
    def __init__(self) -> None:
        super().__init__()
        self.para['name']='PSD'
        self.para['xmin']=-25
        self.para['xmax']=25
        self.para['ymin']=-25
        self.para['ymax']=25
        self.para['numbin_y']=25
        self.para['ptstate']='SURFACE'
        self.para['type']='XY'
        self.para['Scorer']='PSD'
        # user should modify paramters outside the class for now
        # set setpsttype helper method should be added later
