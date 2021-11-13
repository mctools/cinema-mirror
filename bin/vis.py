#!/usr/bin/env python3

import trimesh
import numpy as np
import pyvista as pv
# points = np.array([[-2500, 500, 2500],[-2500, 500, -2500],[2500, 500, -2500],[2500, 500, 2500],[-2500, 500, 2500],[2500, 500, 2500],[2500, 500, -2500],[-2500, 500, -2500]])
#
# cloud = pv.PolyData(points)
# cloud.plot(point_size=15)
#
# surf = cloud.delaunay_2d()
# cloud.plot(show_edges=True)
#







points = np.array([[-2500, 500, 2500],[-2500, 500, -2500],[2500, 500, -2500],[2500, 500, 2500],[-2500, 500, 2500],[2500, 500, 2500],[2500, 500, -2500],[-2500, 500, -2500]])
faces = [[0, 1, 2, 3],[0, 1, 2, 3]]
tmesh = trimesh.Trimesh(points, faces=faces, process=False)
mesh = pv.wrap(tmesh)
print(mesh)


p = pv.Plotter()
p.add_mesh(mesh)
p.show()
