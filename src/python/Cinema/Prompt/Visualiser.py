#!/usr/bin/env python3

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

from ..Interface import *

import pyvista as pv
import random
import matplotlib.colors as mcolors
from .Mesh import Mesh

class Visualiser():
    def __init__(self, blacklist, printWorld=False, nSegments=30, mergeMesh=False, dumpMesh=False):
        self.color =  list(mcolors.CSS4_COLORS.keys())
        self.worldMesh = Mesh()
        self.blacklist = blacklist
        if printWorld:
            self.worldMesh.printMesh()

        self.plotter = pv.Plotter()
        self.loadMesh(nSegments, dumpMesh, mergeMesh)
        self.trj=pv.MultiBlock()
        self.redpoints=pv.MultiBlock()

        self.plotter.show_bounds()
        self.plotter.view_zy()
        self.plotter.show_axes()

        self.plotter.show_grid()
        self.plotter.enable_mesh_picking(callback=self.callback, left_clicking=False, show_message=False)
        self.plotter.add_key_event('s', self.save)

    def save(self):
        print('save screenshot.png')
        self.plotter.screenshot('screenshot.png')

    def addTrj(self, data):
        line = pv.lines_from_points(data)
        line.add_field_data(['a neutron trajectory'], 'mesh_info')
        #draw the first and last position as red dots
        if data.size>2:
            point_cloud = pv.PolyData(data[1:-1])
            self.redpoints.append(point_cloud)
            point_cloud.add_field_data(['a neutron trajectory'], 'mesh_info')
        self.trj.append(line)

    def loadMesh(self, nSegments=30, dumpMesh=False, combineMesh=False):
        count = 0
        if combineMesh:
            allmesh = pv.MultiBlock()

        for am in self.worldMesh:
            name = am.getMeshName()
            if self.blacklist is not None:
                if any(srchstr in name for srchstr in self.blacklist):
                    continue

            name, points, faces = am.getMesh(nSegments)
            name=f'{count}_{name}'
            if points.size==0:
                continue
            
            mesh = pv.PolyData(points, faces)
            if combineMesh:
                allmesh.append(mesh)
            else:
                rcolor = random.choice(self.color)
                mesh.add_field_data([' Volume name: '+name, ' Infomation: '+am.getLogVolumeInfo()], 'mesh_info')
                self.plotter.add_mesh(mesh, color=rcolor, opacity=0.3)
            # 

            if dumpMesh:
                fn=f'{name}.ply'
                print(f'saving {fn}')
                mesh.save(fn, False)
            count+=1

        if combineMesh:
            g = allmesh.combine()
            g.add_field_data(['Combined geometry'], 'mesh_info')
            self.plotter.add_mesh(g, color=random.choice(self.color), opacity=0.3)


    def callback(self, mesh):
        print(f'\nPicked volume info:')
        for info in mesh['mesh_info']:
            print(info)
        # self.plotter.add_point_scalar_labels(mesh.cast_to_pointset(), 'mesh_name')

    def show(self):
        if self.trj.keys()!=[]:
            self.plotter.add_mesh(self.trj.combine(), color='blue', opacity=0.2, line_width=2 )

        if self.redpoints.keys()!=[]:
            self.plotter.add_mesh(self.redpoints.combine(), color='red', opacity=0.3, point_size=8 )

        self.plotter.show(title='Cinema Visualiser')
