#!/usr/bin/env python3

from Cinema.Prompt import Launcher, Visualiser
from Cinema.Prompt.solid import Box, Tessellated
from Cinema.Prompt.gun import PythonGun

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Sphere, Tube, Trapezoid, ArbTrapezoid, Cone
from Cinema.Prompt.gun import PythonGun, MCPLGun
from Cinema.Prompt.scorer import WlSpectrum, PSD
import numpy as np

from Cinema.Prompt import Prompt
from Cinema.Prompt.geo import Volume

import scipy
import json
import pyvista as pv
import os

from abc import ABC, abstractmethod

def read_stl_convert_polydata(fpath):
    dataset = pv.read(fpath)
    polydata = dataset
    return polydata

def pv_show(fpath):
    dataset = pv.read(fpath)
    print(dataset)
    plotter = pv.Plotter()
    plotter.add_mesh(dataset)
    plotter.show()

def x_bounds(polydata:pv.PolyData):
    bds = polydata.bounds
    return [bds[0], bds[1]]

def y_bounds(polydata:pv.PolyData):
    bds = polydata.bounds
    return [bds[2], bds[3]]

def z_bounds(polydata:pv.PolyData):
    bds = polydata.bounds
    return [bds[4], bds[5]]

def decode_to_ndarray(aStr):
    aStr = aStr.split(',')
    values = np.array([float(x) for x in aStr])
    return values

def read_box_obb(fname = 'box.json', ifprint = False):
    '''Read from obb mode'''
    fname = os.path.join(os.path.dirname(__file__), fname)
    with open(fname, 'r') as f:
        box_info = json.load(f)

    for boxid in box_info.keys():
        if ifprint:
            print(boxid)
        for info in box_info[boxid].keys():
            values = box_info[boxid][info]
            values = decode_to_ndarray(values)
            if ifprint:
                print(info, values, sep=': ')
        if ifprint:
            print()

    return box_info

def read_box_from_inner_info(fname = 'section.json', ifprint = False) -> dict:
    fname = os.path.join(os.path.dirname(__file__), fname)
    with open(fname, 'r') as f:
        info = json.load(f)
    if ifprint:
        print(info)
    return info


    # ----> Read obb case
    # numm = 1
    # box_info = read_box_obb()
    # # iterate on all box
    # for box_name in box_info.keys():
    #     infos = box_info[box_name]
    #     # print(infos['dimension'], type(infos['dimension']))
    #     dimension = decode_to_ndarray(infos['dimension'])
    #     location = decode_to_ndarray(infos['location'])
    #     xVector = decode_to_ndarray(infos['xVector'])
    #     guide_wall = Box(dimension[0] * 0.5, dimension[1] * 0.5, dimension[2] * 0.5)
    #     vol_guide_wall = Volume(box_name, guide_wall, 'solid::Cd/8.65gcm3', surfaceCfg=f'physics=Mirror;m={2}')
    #     transform = Transformation3D(location[0], location[1], location[2] - 0.5 * (zbds[0]+zbds[1])).setRotByAlignement( np.array([1,0,0]).reshape(1,3), xVector.reshape(1,3))
    #     gb.placeChild(f'pv_{box_name}', vol_guide_wall, transform)





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

class PhysicalVolume(ABC):
    def __init__(self, material, surface_physics = None):
        self.material = material
        self.surface_physics = surface_physics
        self.name = 'physicalVolume'

    @property
    @abstractmethod
    def volume(self):
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

    def placeInto(self, volumeTo : Volume):
        tr = self.translation
        rot = self.rotation
        transformation = Transformation3D(tr[0], tr[1], tr[2]).applyRotxyz(rot[0], rot[1], rot[2])
        volumeTo.placeChild(f'pv_{self.name}', self.volume, transformation)


class PhysicalVolumeCollection():
    def __init__(self, pvc):
        self.pvc = pvc
        self.name = 'collection'

    def placeInto(self, volume: Volume, wrap = False):
        for pv in self.pvc:
            pv.name = f'{self.name}_{pv.name}'
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
                 thickness_demi=1,
                 m_value = 2
                 ) -> None:
        self.m_value = m_value
        surface_physics = f'physics=Mirror;m={self.m_value}'
        super().__init__('solid::Cd/8.65gcm3', surface_physics)
        self.lrtb = lrtb
        self.edge_entry = demi_entry
        self.edge_exit = demi_exit
        self.dist_entry = dist_entry_demi
        self.dist_exit = dist_exit_demi
        self.z = z_demi
        self.thickness = thickness_demi
        self.zloc = zlocation
        self.name = f'wall_{self.lrtb}'

    @property
    def volume(self):
        dm = self.dimension
        vol = Volume(self.name, Trapezoid(dm[0], dm[1], dm[2], dm[3], dm[4]), self.material, self.surface_physics)
        return vol

    @classmethod
    def from_McstasGuideData(cls, mc_gdata : McstasGuideData, lrtb):
        if lrtb in 'lr':
            return cls(lrtb, 
                       mc_gdata.entry_height, 
                       mc_gdata.exit_height, 
                       mc_gdata.entry_width,
                       mc_gdata.exit_width,
                       0.5*mc_gdata.length,
                       mc_gdata.zlocation
                       )
        elif lrtb in 'tb':
            return cls(lrtb, 
                       mc_gdata.entry_width, 
                       mc_gdata.exit_width,
                       mc_gdata.entry_height, 
                       mc_gdata.exit_height, 
                       0.5*mc_gdata.length,
                       mc_gdata.zlocation,
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

    def __init__(self, walls, wrap=False, material=None) -> None:
        GuideSection.section_id += 1
        super().__init__(walls)
        self.name = f'section{GuideSection.section_id}'
        self.demi_thickness = 1
        self.translation = [0,0,0]
        self.rotation = [0,0,0]
        self.material = material

    @classmethod
    def from_McstasGuideData(cls, mc_gdata : McstasGuideData):
        walls = []
        for wallside in 'ltrb':
            wall = TrapezoidGuideWall.from_McstasGuideData(mc_gdata, wallside)
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

class JsonParser():
    def __init__(self, fname='section.json'):
        self.fname = fname
        self.ginfo = self.readIn()

    def readIn(self, ifprint=False):
        fname = os.path.join(os.path.dirname(__file__), self.fname)
        with open(fname, 'r') as f:
            eell = json.load(f)
        if ifprint:
            print(eell)
        return eell.values()

    def get_bounds(self, var_list):
        zinit = False
        for zz in var_list:
            if zinit == False:
                zmin = zmax = zz
                zinit = True
                continue
            if zz > zmax:
                zmax = zz
            if zz < zmin:
                zmin = zz
        zmidpoint = 0.5 * (zmin + zmax)
        return zmin, zmax, zmidpoint
        
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

    def bound_yinfo(self):
        var_list = []
        for var in self.ginfo:
            var_list.append(var['entryOpening']['halfheight'])
        return self.get_bounds(var_list)

class MySim(PromptMPI):
    def __init__(self, seed) -> None:
        super().__init__(seed)

    def make_psd(self, hx, hy):
        psd = PSD()
        psd.cfg_name = 'guide_out'
        psd.cfg_ptstate = 'ENTRY'
        psd.cfg_xmin = -hx
        psd.cfg_xmax = hx
        psd.cfg_ymin = -hy
        psd.cfg_ymax = hy
        psd.cfg_type = 'XY'
        psd.cfg_numbin_x =100
        psd.cfg_numbin_y =100

        sol_dtt = Box(hx, hy, 1e-2)
        vol_dtt = Volume('guide_out', sol_dtt)
        vol_dtt.addScorer(psd.cfg)
        return vol_dtt

    def makeWorld(self):
        # worldx, worldy, worldz, xbds, ybds, zbds = self.get_world_size()
        guideinfo = JsonParser()
        zmin,zmax,zmid = guideinfo.bound_zinfo()
        _,xmax,_ = guideinfo.bound_xinfo()
        _,ymax,_ = guideinfo.bound_yinfo()
        zlength = zmax - zmin
        world = Volume('world', Box(3*xmax, 3*ymax, zmax+2000))

        gb = Volume('guide_box', Box(2*xmax, 2*ymax, zmax+1000))
        vol_mon = self.make_psd(50, 50)
        # vol_mon = self.make_psd(xbds[1], ybds[1])

        gsc = GuideSectionCollection.from_json('section.json')
        gsc.placeInto(gb)
        
        gb.placeChild('pv_mon', vol_mon, Transformation3D(0,0, zmax + 200))

        world.placeChild('detectorPhy', gb, Transformation3D(0,0,0))

        self.setWorld(world)

if __name__ == "__main__":

    # >>>> Transport run
    sim = MySim(seed=102)
    sim.makeWorld()
    guideinfo = JsonParser()
    zmin,zmax,zmid = guideinfo.bound_zinfo()
    sec1_entry = [15.1 * 2, 47.7 *2]
    gunCfg = f"gun=MaxwellianGun;src_w={sec1_entry[0]};src_h={sec1_entry[1]};src_z={zmin-5000};slit_w={sec1_entry[0]};slit_h={sec1_entry[1]};slit_z={zmin};temperature=293;"
    gunfile = 'bl9_cut.mcpl'
    # gun = MCPLGun(gunfile)
    if False:
        sim.show(gunCfg, 10, zscale=0.01)
    else:
        sim.simulate(gunCfg, 1e6)
        mon = sim.gatherHistData('guide_out')
        if sim.rank == 0:
            plt = mon.plot(show=False,log=True)
            plt.savefig('test.svg')

    # >>>> print cad info
    # read_box_from_inner_info(ifprint=1)

    # >>>> inspect geo
    # fpath = 'guides_simple.stl'
    # pv_show(fpath)




