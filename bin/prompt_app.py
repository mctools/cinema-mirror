#!/usr/bin/env python3

from prompt import Launcher, Mesh
import matplotlib.pyplot as plt
import trimesh
import numpy as np
import pyvista as pv


l = Launcher()
l.loadGeometry("../gdml/first_geo.gdml");

plotter = pv.Plotter()

m = Mesh()
print(f'printMesh {m.printMesh()}')

for am in m:
    name = am.getMeshName()
    print(name)
    if name!='World':
        name, points, faces = am.getMesh(100)

        face3p = []
        face4p = []
        for face in faces:
            if face.size == 3:
                face3p.append(face)
            elif face.size == 4:
                face4p.append(face)

        if len(face4p) !=0 :
            tmesh4 = pv.wrap(trimesh.Trimesh(points, faces=face4p, process=False))
            plotter.add_mesh(tmesh4, color='r')

        if len(face3p) !=0 :
            tmesh3 = pv.wrap(trimesh.Trimesh(points, faces=face3p, process=False))
            plotter.add_mesh(tmesh3, color='r')


plotter.show()
