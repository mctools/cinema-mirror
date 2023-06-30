
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

from .solid import Box
from .geo import Volume

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
        self.cfg_type = 'XZ'


    def makeCfg(self):
        cfg = 'Scorer=PSD;'
        return cfg + super().makeCfg()
    
def makeFlatPSD(name, x, y, z, numbin_x, numbin_y):
    """Make a PSD scorer of Volume BOX.

    Args:
        name (str): name of scorer
        x (float): half length of horizontal dimension
        y (float): half length of vertical dimension
        z (float): half length of depth dimension
        numbin_x (int): number of bins of horizontal dimension
        numbin_y (int): number of bins of vertical dimension

    Returns:
        cfg, vol: config string of scorer, Volume instance with PSD attached to
    """
    xmin = - x 
    xmax = x 
    ymin = - y
    ymax = y

    det = PSD()
    det.cfg_name = name
    det.cfg_xmin = xmin
    det.cfg_xmax = xmax
    det.cfg_numbin_x = numbin_x

    det.cfg_ymin = ymin
    det.cfg_ymax = ymax
    det.cfg_numbin_y = numbin_y
    cfg = det.makeCfg()
    vol = Volume(name, Box(x,y,z))
    vol.addScorer(cfg)
    return cfg, vol
        
