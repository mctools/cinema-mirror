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


from .solid import Trapezoid
from .geo import Volume

def makeTrapezoidGuide(length, x_neg, y_neg, x_pos, y_pos, m, 
                 thickness=1., outer_mateiral='solid::Cd/8.65gcm3',
                 inner_mat='freegas::N78O22/1.225e-3kgm3'):
    inner = Volume('inner', Trapezoid(x_neg, x_pos, y_neg, y_pos, length), matCfg=inner_mat)
    outer = Volume('outer', Trapezoid(x_neg+thickness, x_pos+thickness, 
                                      y_neg+thickness, y_pos+thickness, length), matCfg=outer_mateiral, surfaceCfg=f'physics=Mirror;m={m}')
    outer.placeChild('ininout', inner) 
    return outer
