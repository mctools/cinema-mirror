#import pyvista as pv
#mesh = pv.read('reconstruction/bun_zipper.ply')


import pyvista as pv
import tetgen
import numpy as np
pv.set_plot_theme('document')

bunny = pv.read('reconstruction/bun_zipper_res4.ply')
tet = tetgen.TetGen(bunny)
tet.make_manifold()
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
grid = tet.grid #grid is of UnstructuredGrid type
# grid.show()

# celltypes``VTK_TETRA = 10``
# See   https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
# for all cell types.
if np.all(grid.celltypes==10) == False:
    raise RuntimeError('not all cells are tegrahedron')


points = grid.points 
connectivity=grid.cell_connectivity
connectivity=connectivity.reshape(grid.n_cells,-1)

#now the "connectivity" is a numpy array with size [n_cells, 4]

for pointsid in connectivity:
    print(f'There should be 4 points for this cell, they are {points[pointsid[0]]} {points[pointsid[1]]} {points[pointsid[2]]} {points[pointsid[3]]}')

