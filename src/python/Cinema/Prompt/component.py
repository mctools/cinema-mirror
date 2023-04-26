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


from .solid import Trapezoid, Tube
from .geo import Volume

class SurfaceProcess():

    def __init__(self) -> None:
        pass

    def makeCfg(self):     
        cfg = ''
        for k, v in self.__dict__.items():
            if k.__contains__('cfg_'):
                cfg += k.replace('cfg_','') 
                cfg += '='
                cfg += str(v)
                cfg += ';'
        return cfg

class DiskChopper(SurfaceProcess):

    def __init__(self) -> None:
        super().__init__()
        self.cfg_physics = 'DiskChopper'
        self.cfg_rotFreq = 25
        self.cfg_r = 100
        self.cfg_theta0 = 20
        self.cfg_n = 1
        self.cfg_phase = 0

    def get_cfg(self):

        return self.makeCfg()

        
def makeTrapezoidGuide(length, x_neg, y_neg, x_pos, y_pos, m, 
                 thickness=1., outer_mateiral='solid::Cd/8.65gcm3',
                 inner_mat='freegas::N78O22/1.225e-3kgm3'):
    inner = Volume('inner', Trapezoid(x_neg, x_pos, y_neg, y_pos, length), matCfg=inner_mat)
    outer = Volume('outer', Trapezoid(x_neg+thickness, x_pos+thickness, 
                                      y_neg+thickness, y_pos+thickness, length), 
                                      matCfg=outer_mateiral, 
                                      surfaceCfg=f'physics=Mirror;m={m}')
    outer.placeChild('ininout', inner) 
    return outer

def makeDiskChopper(r_outer, r_inner, phase, num_slit, freq, theta):

    vol = Volume('chopper', Tube(0., r_outer, 1e-2, 0., 360.))
    chp = DiskChopper()
    chp.cfg_rotFreq = freq
    chp.cfg_n = num_slit
    chp.cfg_phase = phase
    chp.cfg_r = r_inner
    chp.cfg_theta0 = theta
    vol.setSurface(chp.get_cfg())

    return vol
    
