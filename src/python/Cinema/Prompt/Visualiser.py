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
    def __init__(self, blacklist, printWorld=False):
        self.color =  list(mcolors.CSS4_COLORS.keys())
        self.plotter = pv.Plotter()
        self.worldMesh = Mesh()
        self.blacklist = blacklist
        if printWorld:
            self.worldMesh.printMesh()
        self.loadMesh()


    def addLine(self, data):
        line = pv.lines_from_points(data)
        self.plotter.add_mesh(line, color='blue', opacity=0.2, line_width=2)
        if data.size>2:
            point_cloud = pv.PolyData(data[1:-1])
            self.plotter.add_mesh(point_cloud, color='red', opacity=0.3)


    def loadMesh(self):
        for am in self.worldMesh:
            name = am.getMeshName()
            if any(srchstr in name for srchstr in self.blacklist):
                continue

            print(f'loading mesh {name}')
            if name!='World':
                name, points, faces = am.getMesh(10)
                if points.size==0:
                    continue
                rcolor = random.choice(self.color)
                mesh = pv.PolyData(points, faces)
                self.plotter.add_mesh(mesh, color=rcolor, opacity=0.3)

    def show(self):
        self.plotter.show()
