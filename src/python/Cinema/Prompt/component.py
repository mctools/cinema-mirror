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

from Cinema.Prompt.geo import Transformation3D
from .solid import Trapezoid, Tube, Box
from .geo import Volume, Transformation3D
from copy import deepcopy
from scipy.spatial.transform import Rotation as scipyRot
from .configstr import ConfigString

class DiskChopper(ConfigString):
    def __init__(self) -> None:
        super().__init__({})
        self.cfg_physics = 'DiskChopper'
        self.cfg_rotFreq = 25
        self.cfg_r = 100
        self.cfg_theta0 = 20
        self.cfg_n = 1
        self.cfg_phase = 0


class Anchor:
    
    def __init__(self, refFrame = None, marker = '') -> None:
        '''
        An anchor is a point with spacial coordinates and reference frame.
        '''
        if refFrame == None:
            refFrame = Transformation3D()
        self.refFrame = refFrame
        self.marker = marker

    def setMarker(self, marker : str):
        self.marker = marker

    def setRefFrame(self, refFrame : Transformation3D):
        self.refFrame = deepcopy(refFrame)

    def getRelTransf(self, other):
        """Return the relative transformation from 'self' to 'other'.

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        transf = Anchor()
        transl = (other.refFrame.translation - self.refFrame.translation).dot(self.refFrame.sciRotMatrix)
        transl_ins = Transformation3D(transl[0], transl[1], transl[2])
        # transl_ins.sciRot =self.refFrame.sciRot.inv() *  other.refFrame.sciRot
        refMatrix = np.linalg.inv(self.refFrame.sciRotMatrix).dot(other.refFrame.sciRotMatrix)
        transl_ins.applyTrans(refMatrix)
        transf.setRefFrame(transl_ins)
        return transf
    
    # fixme: add inverse of Transformation 
    # fixme: add refresh of all members' anchor
    # def transform(self, frame: Transformation3D):
    #     self.refFrame = self.refFrame * frame

class Array(Anchor):

    def __init__(self, refFrame = Transformation3D(), marker = '') -> None:
        super().__init__(refFrame, marker)
        self.members = []

    def setMemberMarker(self):
        for i_mem in self.members:
            i_mem.setMarker(f'{self.marker}|{i_mem.marker}')
    
    def setAnchor(self, refFrame : Transformation3D):
        self.refFrame = refFrame
        for mem in self.members:
            mem.refFrame = self.refFrame * mem.refFrame

    # def getRelativeRefFrame(self):
    #     ancList = []
    #     for mem in self.members:
    #         anc = self.getRelTransf(mem)
    #         ancList.append(anc)
    #     return ancList    

class EntityArray(Array):

    def __init__(self, element, refFrame = Transformation3D(), marker = 'Entity'):
        super().__init__(refFrame, marker)
        self.element = element
        if hasattr(self.element, 'marker'):
            self.marker = self.marker + '|' + self.element.marker
        self.eleAncs = []
        
    def __check(self):
        if not isinstance(self.element, Volume):
            raise TypeError("Object type f{self.element.__class__.__name__} do not match type 'Volume'!")
    
    def selectByZ(self, zMin, zMax): # fixme: not working
        for member in self.members:
            loc = member.translation
            if loc[2] < zMin or loc[2] > zMax:
                self.members.pop(member)

    def setAnchor(self, refFrame : Transformation3D):
        super().setAnchor(refFrame)
        for eleAnc in self.eleAncs:
            eleAnc[1].refFrame = self.refFrame * eleAnc[1].refFrame

    def make(self):
        anc = Anchor()
        if isinstance(self.element, Volume):
            anc.setRefFrame(Transformation3D())
        else:
            anc.setRefFrame(self.element.refFrame)
        self.eleAncs.append((self.element, anc))
        self.members.append(anc)
    
    def create(self, transf):
        anc = Anchor()
        anc.setRefFrame(transf)
        self.eleAncs.append((self.element, anc))
        self.members.append(anc)
        self.setMemberMarker()

    def repeat(self, direction = [0, 0, 1], gap = 1000., num = 1):
        new_anchors = []
        for i_anc in self.members:
            origin_transf = deepcopy(i_anc.refFrame)
            for i_num in np.arange(1, num):
                gap = gap * i_num
                new_anc = Transformation3D(direction[0] * gap, direction[1] * gap, direction[2] * gap) * origin_transf 
                new_anchors.append(new_anc)
        for i_new_anc in new_anchors:
            self.create(i_new_anc)

    def reflect(self, plane = 'XY', plane_location = 0.):
        new_anchors = []
        for mem in self.members:
            absTrans = self.refFrame * mem.refFrame
            transl = absTrans.translation
            if plane == 'YZ':
                transl[0] = 2 * plane_location - transl[0]
                # rotvec[0] = - rotvec[0]
            elif plane == 'XY':
                transl[2] = 2 * plane_location - transl[2]
                refMatrix = scipyRot.identity().as_matrix() - 2 * np.outer(np.array([0,0,-1]), np.array([0,0,-1]))
                # rotvec[2] = - rotvec[2]
            elif plane == 'XZ':
                transl[1] = 2 * plane_location - transl[1]
                # rotvec[1] = - rotvec[1]
            else:
                raise ValueError(f'plane should be XY, YZ or XZ, but got {plane}.')
            new_anc = Anchor()
            new_anc.refFrame.translation = transl
            new_anc.refFrame.applyTrans(refMatrix.dot(absTrans.sciRotMatrix))
            # new_anc.setRefFrame(absTrans * new_anc.refFrame)
            relAnc = self.getRelTransf(new_anc)
            new_anchors.append(relAnc)
        for i_new_anc in new_anchors:
            self.create(i_new_anc.refFrame)
            

    def rotate_z(self, angle):
        self.setRefFrame(Transformation3D().applyRotZ(angle) * self.refFrame)


    def make_rotate_z(self, angle, num, x= 0, y = 0):
        for mem in self.members:
            new_anchors = []
            absTrans = self.refFrame * mem.refFrame
            for i_num in np.arange(1, num + 1):
                temp_anc = Anchor()
                transl = absTrans.translation
                angle_in = angle * i_num / 180 * np.pi
                temp_anc.refFrame = Transformation3D().applyRotZ(angle)
                u = np.array([0,0,1.])
                originTrans = np.array([x,y,0])
                v = transl - originTrans
                v_para = (v * u) * u
                vt = v - v_para
                asin = np.sin(angle_in)
                acons = np.cos(angle_in)
                transl = v_para + np.sin(angle_in) * np.cross(u, vt) + np.cos(angle_in) * vt
                transl = originTrans + transl
                skew_u = [[0, - u[2], u[1]],
                          [u[2], 0, - u[0]],
                          [- u[1], u[0], 0]]
                skew_u = np.array(skew_u)
                transl_mat = np.cos(angle_in) * scipyRot.identity().as_matrix() + (1-np.cos(angle_in)) * np.outer(u, u) + np.sin(angle_in) * skew_u
                absAnc = Anchor()
                absAnc.refFrame = Transformation3D(transl[0], transl[1], transl[2])
                absAnc.refFrame.applyTrans(transl_mat.dot(absTrans.sciRotMatrix))
                relAnc = self.getRelTransf(absAnc)
                new_anchors.append(relAnc)
        for i_new_anc in new_anchors:
            self.create(i_new_anc.refFrame)

class FlatAnalyser(EntityArray):

    def __init__(self, element, spacings = None, refFrame = Transformation3D(), marker = 'FlatAna'):
        super().__init__(element, refFrame, marker)
        self.spacing = spacings

    def make_trape_plane(self, upper_l, lower_l, height, cur_h=0, cur_v=0):
        points = self.set_trape_position(upper_l, lower_l, height)
        self.make_plane(cur_h, cur_v, points)

    def make_rect_plane(self, num_h, num_v, cur_h=0, cur_v=0):
        points = self.set_rect_position(num_h, num_v)
        self.make_plane(cur_h, cur_v, points)

    def set_rect_position(self, num_h, num_v):
        xx = np.arange(0, num_h) * self.spacing[0]
        yy = np.arange(0, num_v) * self.spacing[1]
        midx = xx[-1] * 0.5
        midy = yy[-1] * 0.5
        xx = xx - midx
        yy = yy - midy
        return zip(xx, yy)

    def set_trape_position(self, upper_l, lower_l, height): # fixme: need another algorithm 
        positions = []
        num_y = height // self.spacing[1]
        for row in np.arange(num_y):
            height_r = row * self.spacing[1]
            length_r = height_r / height * ((lower_l - upper_l)) + upper_l
            num_x = length_r // self.spacing[0]
            origin_x = (length_r - upper_l) * 0.5 // self.spacing[0]
            xx = np.arange(- origin_x, num_x-origin_x) * self.spacing[0]
            positions_r = []
            for x in xx:
                y = row * self.spacing[1]
                positions_r.append([x, y])
            positions.append(positions_r)
        midx = upper_l * 0.5
        midy = height * 0.5
        for pos_row in positions:
            for pos in pos_row:
                pos[0] = pos[0] - midx
                pos[1] = pos[1] - midy
        return positions

    def make_plane(self, cur_h, cur_v, points):
        '''
        Make a plane double curve analyser component.
        '''
        ir = 0
        ys = [row[0][1] for row in points]
        height = max(ys)
        for row in points:
            ir += 1
            xs = [point[0] for point in row]
            length = max(xs)
        # self.anchor = Volume("Analyser", Box(length, height, 20))
            ic = 0
            for col in row:
                ic += 1
                ix = col[0]
                iy = col[1]
                marker = f'{self.element}_r{ir}c{ic}'
                vol = Volume(f'{marker}', self.element.solid, self.element.matCfg, self.element.surfaceCfg)
                # if abs((iy - height * 0.5) / cur_v) > 1.:
                #     raise ValueError(f'Too small cur_v: {cur_v}')
                # if abs((ix - length * 0.5) / cur_h) > 1.:
                #     raise ValueError(f'Too small cur_v: {cur_h}')
                if cur_h == 0 and cur_v == 0:
                    # print('Using plane array!')
                    tilt_h = 0
                    tilt_v = 0
                    pass
                elif cur_h == 0:
                    # print('Using vertical single curved surface')
                    tilt_h = 0
                    tilt_v = - np.arcsin((iy - height * 0.5) / cur_v) * np.rad2deg(1)
                elif cur_v == 0:
                    # print('Using horizontal single curved surface')
                    tilt_h = np.arcsin((ix - length * 0.5) / cur_h) * np.rad2deg(1)
                    tilt_v = 0
                else:
                    # print('Using double curved surface')
                    tilt_h = np.arcsin(ix / cur_h) * np.rad2deg(1)
                    tilt_v = - np.arcsin(iy / cur_v) * np.rad2deg(1)
                transf = Transformation3D(ix, iy, 0,).applyRotxyz(tilt_v, tilt_h, 0)
                anc = Anchor()
                anc.setRefFrame(transf * self.refFrame)
                anc.setMarker(f'{marker}')
                self.eleAncs.append((vol,anc))
                self.members.append(anc)


def make2CurveAnalyser(nums = [20, 20], lengths = [0, 0], spacings = [0, 0], curves = [0, 0], matCfg = 'freegas::H1/1e-26kgm3'):
    crystal_plate = Volume("crystal", Box(lengths[0] * 0.5, lengths[1] * 0.5, 1), matCfg='solid::Cd/8.65gcm3', surfaceCfg='physics=Mirror;m=2')
    analyser = EntityArray(crystal_plate, [nums[0], nums[1]], [lengths[0] + spacings[0], lengths[1] + spacings[1]], curves[0], curves[1])
    return analyser


        
def makeTrapezoidGuide(length, front_x, front_y, rear_x, rear_y, m, 
                 thickness=20., outer_mateiral='solid::Cd/8.65gcm3',
                 inner_mat='freegas::H1/1e-26kgm3'):
    inner = Volume('inner', Trapezoid(front_x, rear_x, front_y, rear_y, length), matCfg=inner_mat)
    outer = Volume('outer', Trapezoid(front_x+thickness, rear_x+thickness, 
                                      front_y+thickness, rear_y+thickness, length), 
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
    