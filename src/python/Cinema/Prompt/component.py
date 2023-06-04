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

import numpy as np
from .solid import Trapezoid, Tube, Box
from .geo import Volume, Transformation3D
from copy import deepcopy
from scipy.spatial.transform import Rotation as scipyRot

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


class Anchor:
    
    def __init__(self, refFrame = Transformation3D(), marker = '') -> None:
        '''
        An anchor is a point with spacial coordinates and reference frame.
        '''
        self.refFrame = refFrame
        self.marker = marker

    def setMarker(self, marker : str):
        self.marker = marker

    def setRefFrame(self, refFrame : Transformation3D):
        self.refFrame = deepcopy(refFrame)

class Array(Anchor):

    def __init__(self, refFrame = Transformation3D(), marker = '') -> None:
        super().__init__(refFrame, marker)
        self.members = []

    def setMemberMarker(self):
        for i_mem in self.members:
            i_mem.setMarker(f'{self.marker}|{i_mem.marker}')

    def get_member_abs_frame(self, anchor : Anchor) -> Transformation3D:
        return self.refFrame * anchor.refFrame
    
class EntityArray(Array):

    def __init__(self, element, 
                 array_size = None, spacings = None, refFrame = Transformation3D(), marker = 'Arr') -> None:
        super().__init__(refFrame, marker)
        if isinstance(element, EntityArray):
            element.refFrame = refFrame * element.refFrame # making all EntityArray refFrame as absolute frame
        self.element = element
        if hasattr(self.element, 'marker'):
            self.marker = self.marker + '|' + self.element.marker
        self.size = array_size
        self.spacing = spacings
        # self.vols = []
        self.eleAncs = []
        # self.__check()

    def __check(self):
        if not isinstance(self.element, Volume):
            raise TypeError("Object type f{self.element.__class__.__name__} do not match type 'Volume'!")
    
    def make(self):
        if self.size == None:
            anc = Anchor()
            anc.setRefFrame(self.element.refFrame)
            self.eleAncs.append((self.element, anc))
            self.members.append(anc)
        else:
            loc_x, loc_y = self.set_plane_location()
            for ix in loc_x:
                for iy in loc_y:
                    transf = Transformation3D(ix - loc_x[-1] * 0.5, iy - loc_y[-1] * 0.5, 0.)
                    anc = Anchor()
                    if self.refFrame != None:
                        anc.setRefFrame(self.refFrame * transf * self.element.refFrame)
                    else:
                        anc.setRefFrame(transf)
                    self.eleAncs.append((self.element, anc))
                    self.members.append(anc)
    
    def create(self, transf):
        anc = Anchor()
        anc.setRefFrame(transf)
        self.eleAncs.append((self.element, anc))
        self.members.append(anc)
        self.setMemberMarker()

    def repeat(self, direction = [0, 0, 1], gap = 1000., num = 1):
        origin_transf = deepcopy(self.element.refFrame)
        for i_num in np.arange(1, num):
            gap = gap * i_num
            transf = Transformation3D(direction[0] * gap, direction[1] * gap, direction[2] * gap) * origin_transf 
            self.create(transf)

    def reflect(self, plane = 'YZ'):
        new_anchors = []
        for i_anc in self.members:
            transl = deepcopy(i_anc.refFrame.translation)
            transl[0] = - transl[0]
            new_anc = Anchor()
            new_anc.setRefFrame(Transformation3D(transl[0], transl[1], transl[2]))
            euler_rot = i_anc.refFrame.sciRot.as_euler('xyz', True)
            euler_rot[1] = - euler_rot[1]
            new_anc.refFrame.applyRotxyz(euler_rot[0], euler_rot[1], euler_rot[2], True)
            new_anchors.append(new_anc)
        for i_new_anc in new_anchors:
            self.create(i_new_anc.refFrame)
            

    def make_plane(self, cur_h, cur_v):
        '''
        Make a plane double curve analyser component.
        '''
        loc_x, loc_y = self.set_plane_location()
        # self.anchor = Volume("Analyser", Box(loc_x[-1], loc_y[-1], 20))
        for ix in loc_x:
            for iy in loc_y:
                marker = f'{self.element}_r{ix}c{iy}'
                vol = Volume(f'{marker}', self.element.solid, self.element.matCfg, self.element.surfaceCfg)
                # if abs((iy - loc_y[-1] * 0.5) / cur_v) > 1.:
                #     raise ValueError(f'Too small cur_v: {cur_v}')
                # if abs((ix - loc_x[-1] * 0.5) / cur_h) > 1.:
                #     raise ValueError(f'Too small cur_v: {cur_h}')
                if cur_h == 0 and cur_v == 0:
                    # print('Using plane array!')
                    tilt_h = 0
                    tilt_v = 0
                    pass
                elif cur_h == 0:
                    # print('Using vertical single curved surface')
                    tilt_h = 0
                    tilt_v = - np.arcsin((iy - loc_y[-1] * 0.5) / cur_v) * np.rad2deg(1)
                elif cur_v == 0:
                    # print('Using horizontal single curved surface')
                    tilt_h = np.arcsin((ix - loc_x[-1] * 0.5) / cur_h) * np.rad2deg(1)
                    tilt_v = 0
                else:
                    # print('Using double curved surface')
                    tilt_h = np.arcsin((ix - loc_x[-1] * 0.5) / cur_h) * np.rad2deg(1)
                    tilt_v = - np.arcsin((iy - loc_y[-1] * 0.5) / cur_v) * np.rad2deg(1)
                transf = Transformation3D(ix - loc_x[-1] * 0.5, iy - loc_y[-1] * 0.5, 0,).applyRotxyz(tilt_v, tilt_h, 0)
                anc = Anchor()
                anc.setRefFrame(transf)
                anc.setMarker(f'{marker}')
                self.eleAncs.append((vol,anc))
                self.members.append(anc)
                # self.parent.placeChild(f"Arr_{marker}", vol, 
                #                        self.transformation * Transformation3D(ix - loc_x[-1] * 0.5, iy - loc_y[-1] * 0.5, 0,).applyRotxyz(tilt_v, tilt_h, 0))
        # self.parent.placeChild('Phy_Analyser', self.anchor, Transformation3D(500,0,17100,90,-90,-90))
                # print(f'Transformation is {Transformation3D(ix, iy, 18000, 0, 90, 0)}')

    def set_plane_location(self):
        xx = np.arange(0., self.size[0]) * self.spacing[0]
        yy = np.arange(0., self.size[1]) * self.spacing[1]
        return xx, yy
    
    # def set_global_transf(self, local_transf : Transformation3D):
    #     # raise ValueError('Stop')
    #     local_rot = self.rotation * local_transf.sciRot 
    #     local_transl = self.translation + self.rotation.apply(local_transf.getTranslation()) 
    #     print(f'running {local_transl}')
    #     transf = Transformation3D(local_transl[0], local_transl[1], local_transl[2])
    #     transf.sciRot = local_rot
    #     transf.setSciRot()
    #     return transf


def make2CurveAnalyser(nums = [20, 20], lengths = [0, 0], spacings = [0, 0], curves = [0, 0], matCfg = 'freegas::H1/1e-26kgm3'):
    crystal_plate = Volume("crystal", Box(lengths[0] * 0.5, lengths[1] * 0.5, 1), matCfg='solid::Cd/8.65gcm3', surfaceCfg='physics=Mirror;m=2')
    analyser = EntityArray(crystal_plate, [nums[0], nums[1]], [lengths[0] + spacings[0], lengths[1] + spacings[1]], curves[0], curves[1])
    return analyser


        
def makeTrapezoidGuide(length, x_neg, y_neg, x_pos, y_pos, m, 
                 thickness=800., outer_mateiral='solid::Cd/8.65gcm3',
                 inner_mat='freegas::H1/1e-26kgm3'):
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
    