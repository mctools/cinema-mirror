#!/usr/bin/env python3

# MCstas Guide Data.
from abc import ABC, abstractmethod

from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Trapezoid
from jsonReader import JsonParser
import json, os
import numpy as np

class McstasGuideData():
    def __init__(self,
                 entry_width = 10,
                 entry_height = 5,
                 exit_width = 10,
                 exit_height = 5,
                 length = 20,
                 zlocation = 0) -> None:
        self.entry_width = entry_width
        self.entry_height = entry_height
        self.exit_width = exit_width
        self.exit_height = exit_height
        self.length = length
        self.zlocation = zlocation

    @classmethod
    def from_occdict(cls, adict):
        section_length = adict['length'] # full length, not demi length
        zlocation = adict['zlocation']

        entry_opening = adict['entryOpening']
        entry_width = entry_opening['halfwidth']
        entry_height = entry_opening['halfheight']

        exit_opening = adict['exitOpening']
        exit_width = exit_opening['halfwidth']
        exit_height = exit_opening['halfheight']
        return cls(entry_width, entry_height, exit_width, exit_height, section_length, zlocation)
    
    @property
    def volume_capacity(self):
        pass


class PhysicalVolume(ABC):
    def __init__(self, material, surface_physics = None):
        self.material = material
        self.surface_physics = surface_physics
        self.logicalVolume = self.makeLogicalVolume()

    @abstractmethod
    def makeLogicalVolume(self) -> Volume :
        pass

    @property
    @abstractmethod
    def dimension(self):
        pass

    @property
    @abstractmethod
    def translation(self):
        pass

    @property
    @abstractmethod
    def rotation(self):
        pass

    def add_scorer(self, scorer):
        self.logicalVolume.addScorer(scorer)

    def placeInto(self, volumeTo : Volume):
        tr = self.translation
        rot = self.rotation
        transformation = Transformation3D(tr[0], tr[1], tr[2]).applyRotxyz(rot[0], rot[1], rot[2])
        volumeTo.placeChild(f'{self.pname}', self.logicalVolume, transformation)


class PhysicalVolumeCollection():
    def __init__(self, pvc):
        self.pvc = pvc

    def placeInto(self, volume: Volume, wrap = False):
        for pv in self.pvc:
            pv.pname = f'{self.pname}_{pv.pname}'
            pv.placeInto(volume)

class TrapezoidGuideWall(PhysicalVolume):
    '''
    Implementation in Trapezoid
    From Entry, Exit, Length, Location (McstasGuideData) 
    to Dimension, Translation, Rotation (DTR)
    '''
    def __init__(self, 
                 lrtb, 
                 demi_entry, 
                 demi_exit, 
                 dist_entry_demi,
                 dist_exit_demi,
                 z_demi, 
                 zlocation,
                 thickness_demi=0.1,
                 m_value = 3
                 ) -> None:
        self.m_value = m_value
        surface_physics = f'physics=Mirror;m={self.m_value};threshold=1e-3'
        self.lrtb = lrtb
        self.edge_entry = demi_entry
        self.edge_exit = demi_exit
        self.dist_entry = dist_entry_demi
        self.dist_exit = dist_exit_demi
        self.z = z_demi
        self.thickness = thickness_demi
        self.zloc = zlocation
        self.lvname = f'wall_{self.lrtb}'
        self.pname = f'pv{self.lvname}'
        super().__init__('solid::Cd/8.65gcm3', surface_physics)

    def makeLogicalVolume(self):
        dm = self.dimension
        vol = Volume(self.lvname, Trapezoid(dm[0], dm[1], dm[2], dm[3], dm[4]), self.material, self.surface_physics)
        return vol

    @classmethod
    def from_McstasGuideData(cls, mc_gdata : McstasGuideData, lrtb, thickness=1):
        if lrtb in 'lr':
            return cls(lrtb, 
                       mc_gdata.entry_height, 
                       mc_gdata.exit_height, 
                       mc_gdata.entry_width,
                       mc_gdata.exit_width,
                       0.5*mc_gdata.length,
                       mc_gdata.zlocation,
                       thickness
                       )
        elif lrtb in 'tb':
            return cls(lrtb, 
                       mc_gdata.entry_width, 
                       mc_gdata.exit_width,
                       mc_gdata.entry_height, 
                       mc_gdata.exit_height, 
                       0.5*mc_gdata.length,
                       mc_gdata.zlocation,
                       thickness
                       )
        
    @property
    def dimension(self):
        if self.lrtb in 'lr':
            dx1 = self.thickness
            dx2 = self.thickness
            dy1 = self.edge_entry     #todo: temp fix guide leak
            dy2 = self.edge_exit    #todo: temp fix guide leak
            dz = self.get_dz_rotated()
        elif self.lrtb in 'tb':
            dx1 = self.edge_entry  
            dx2 = self.edge_exit 
            dy1 = self.thickness
            dy2 = self.thickness
            dz = self.get_dz_rotated()
        else:
            raise NotImplementedError()
        return [dx1, dx2, dy1, dy2, dz]

    @property
    def rotation(self):
        abscosine = self.z / self.get_dz_rotated()
        if self._get_sign() == 0:
            rot_degree = 0  # np.rad2deg(np.arccos(abscosine))
        else:
            rot_degree = self._get_sign() * np.rad2deg(np.arccos(abscosine))
        if self.lrtb == 'l':
            return [0,  rot_degree, 0]
        elif self.lrtb == 'r':
            return [0, - rot_degree, 0]
        elif self.lrtb == 't':
            return [- rot_degree, 0, 0]
        elif self.lrtb == 'b':
            return [rot_degree, 0, 0]

    @property
    def translation(self):
        cosine = self.z / self.get_dz_rotated()
        diff_dist = 0.5 * (self.dist_exit - self.dist_entry)
        if self._get_sign() == 0:
            diff_thickness = self.thickness
        else:
            diff_thickness = self.thickness * cosine 
        dist_midpoint = self.dist_entry + diff_dist + diff_thickness 
        sine = np.sqrt(1 - cosine **2)
        z_midpoint = self.z - self._get_sign() * sine * self.thickness + self.zloc
        if self.lrtb == 'l':
            return [dist_midpoint, 0, z_midpoint]
        elif self.lrtb == 'r':
            return [-dist_midpoint, 0, z_midpoint]
        elif self.lrtb == 't':
            return [0, dist_midpoint, z_midpoint]
        elif self.lrtb == 'b':
            return [0, -dist_midpoint, z_midpoint]
        else:
            raise NotImplementedError()
    
    def _get_sign(self):
        if abs(self.dist_entry - self.dist_exit) < 1e-3:
            sn = 0
        else:
            sn = np.sign((self.dist_exit - self.dist_entry))
        return sn
    
    def get_dz_rotated(self):
        return np.sqrt((0.5*(self.dist_exit - self.dist_entry))**2 + (self.z) ** 2)

class GuideSection(PhysicalVolumeCollection):
    section_id = 0

    def __init__(self, walls, wrap=False, material='void.ncmat') -> None:
        GuideSection.section_id += 1
        super().__init__(walls)
        self.demi_thickness = 1
        self.translation = [0,0,0]
        self.rotation = [0,0,0]
        self.material = material
        self.pname = f'section{GuideSection.section_id}'

    @classmethod
    def from_McstasGuideData(cls, mc_gdata : McstasGuideData, thickness=1):
        walls = []
        for wallside in 'ltrb':
            wall = TrapezoidGuideWall.from_McstasGuideData(mc_gdata, wallside, thickness)
            walls.append(wall)
        return cls(walls)

class GuideSectionCollection(PhysicalVolumeCollection):
    def __init__(self, collection) -> None:
        super().__init__(collection)

    @classmethod
    def from_json(cls, file):
        section_info = JsonParser(file).ginfo
        gsc = []
        for mcgd_dict in section_info:
            mcgd = McstasGuideData.from_occdict(mcgd_dict)
            gs = GuideSection.from_McstasGuideData(mcgd)
            gsc.append(gs)
        return cls(gsc)
    
    @property
    def pname(self):
        return f'Collection'
        
        
    def bound_zinfo(self):
        var_list = []
        for var in self.ginfo:
            var_list.append(var['zlocation'] + var['length'])
        return self.get_bounds(var_list)

    def bound_xinfo(self):
        var_list = []
        for var in self.ginfo:
            var_list.append(var['entryOpening']['halfwidth'])
        return self.get_bounds(var_list)
    
    def bound_zinfo(self):
        var_list = []
        for var in self.ginfo:
            var_list.append(var['zlocation'] + var['length'])
        return self.get_bounds(var_list)

    def bound_xinfo(self):
        var_list = []
        for var in self.ginfo:
            var_list.append(var['entryOpening']['halfwidth'])
        return self.get_bounds(var_list)

