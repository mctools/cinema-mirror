#!/usr/bin/env python3


import pyvista as pv
import tetgen
import numpy as np
from lxml import etree

class GdmlElement():
    '''
    
    '''
    def __init__(self, world_size = [100.0, 100.0, 100.0]):
        
        self.solid_counter = 0
        self.gdml = etree.Element('gdml')
        self.userinfo = etree.SubElement(self.gdml, 'userinfo')
        self.materials = etree.SubElement(self.gdml, 'materials')
        self.define = etree.SubElement(self.gdml, 'define')
        self.solids = etree.SubElement(self.gdml, 'solids')
        self.structure = etree.SubElement(self.gdml, 'structure')
        self.__set_world(world_size)

    def __set_world(self, xyz):

        sol_world = etree.SubElement(self.solids, 'box')
        sol_world.set('lunit', 'mm')
        sol_world.set('name', 'sol_world')
        sol_world.set('x', f'{xyz[0]}')
        sol_world.set('y', f'{xyz[1]}')
        sol_world.set('z', f'{xyz[2]}')

        self.world_vol = self.set_logical('Vacuum', sol_world, return_vol=True)
        self.setup = etree.SubElement(self.gdml, 'setup')
        self.setup.set('name', 'Default')
        self.setup.set('version', '1.0')
        self.setup_world = etree.SubElement(self.setup, 'world')
        self.setup_world.set('ref', f'vol_{sol_world.get("name")}')
        self.structure.append(self.world_vol)

    def set_position(self, name, xyz = [0.0, 0.0, 0.0], element = None):

        position = etree.Element('position')
        position.set('name', name)
        position.set('x', f'{"%.6f"%xyz[0]}')
        position.set('y', f'{"%.6f"%xyz[1]}')
        position.set('z', f'{"%.6f"%xyz[2]}')
        if not element:
            self.define.append(position)
        elif isinstance(element, etree.Element):
            element.append(position)
        else:
            raise TypeError('Setting position subelement failed. Please check element.')
    
    def set_rotation(self, name, xyz = [0.0, 0.0, 0.0], unit = 'deg'):

        rot = etree.Element('rotation')
        rot.set('name', name)
        rot.set('x', f'{xyz[0]}')
        rot.set('y', f'{xyz[1]}')
        rot.set('z', f'{xyz[2]}')
        rot.set('unit', str(unit))
        self.define.append(rot)

    def set_tetrahedron(self, cells, coordinates):
        
        self.solid_counter = self.solid_counter + 1
        from copy import deepcopy
        
        i_cell = 0
        sol = etree.Element('solids')
        for point in range(len(coordinates)):
            self.set_position(f'point{point + 1}', coordinates[point])

        for cell in cells:
            i_cell = i_cell + 1
            i_point = 0
            tetrahedron = etree.Element('tet')
            tetrahedron.set('name', f'cell{i_cell}')
            for point in cell:
                i_point = i_point + 1
                tetrahedron.set(f'vertex{i_point}', f'point{point+1}')

            self.solids.append(tetrahedron)
            sol.append(deepcopy(tetrahedron))
        return sol

    def set_logical(self, mat_name, sol = None, return_vol = False):

        if not sol:
            iterator = self.solids.iterchildren(tag=etree.Element)
        else:
            iterator = sol.iterchildren(tag=etree.Element)

        for sol in iterator:
            vol = etree.Element('volume')
            vol.set('name', f'vol_{sol.get("name")}')

            solid = etree.SubElement(vol, 'solidref')
            solid.set('ref', sol.get("name"))

            mat = etree.SubElement(vol, 'materialref')
            mat.set('ref', mat_name)

            if return_vol:
                return vol
            else:
                self.structure.append(vol)

    def set_material(self, name, value):

        mat = etree.Element('material')
        mat.set('name', name)
        atom = etree.SubElement(mat, 'atom')
        atom.set('value', value)
        self.materials.append(mat)

    def set_physical(self, sol, position, rotation, parent = None):

        if not sol:
            iterator = self.solids.iterchildren(tag=etree.Element)
        else:
            iterator = sol.iterchildren(tag=etree.Element)

        self.set_position(f'p_sol_{self.solid_counter}', xyz=position)
        self.set_rotation(f'r_sol_{self.solid_counter}', xyz=rotation)
        for child in iterator:
            pVol = etree.Element('physvol')
            volRef = etree.SubElement(pVol, 'volumeref')
            volRef.set('ref', f'vol_{child.get("name")}')
            positionVol = etree.SubElement(pVol, 'positionref')
            positionVol.set('ref', f'p_sol_{self.solid_counter}')
            rotationVol = etree.SubElement(pVol, 'rotationref')
            rotationVol.set('ref', f'r_sol_{self.solid_counter}')
        
            if not parent:
                self.world_vol.append(pVol)
            else:
                parent.append(pVol)
    
    def export_gdml(self, file = 'user.gdml'):

        self.sort_before_export()
        output = etree.tostring(self.gdml, pretty_print=True)
        with open(file, 'wb') as file:
            file.write(output)

    def sort_before_export(self):

        if self.structure[0].get('name') == "vol_sol_world":
            self.structure.append(self.structure[0])

# def closest_pair_naive(points):

#     import numpy as np
#     points = np.array(points)
#     points

pv.set_plot_theme('document')

bunny = pv.read('../../files/bunny/reconstruction/bun_zipper.ply')
# bunny.plot()
tet = tetgen.TetGen(bunny)
tet.make_manifold()
tet.tetrahedralize(order=1, mindihedral=40, minratio=1.5)
grid = tet.grid #grid is of UnstructuredGrid type
# grid = grid.explode() # explode view
# grid.plot(show_axes=True)

# pter = pv.Plotter()
# pter.add_mesh(grid)
# pter.show_axes()
# pter.add_mesh(exploded, show_edges=True)
# pter.save_graphic('rabbit.svg', title='Stanford Bunny')
# pter.show()

# celltypes``VTK_TETRA = 10``
# See   https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
# for all cell types.
if np.all(grid.celltypes==10) == False:
    raise RuntimeError('not all cells are tegrahedron')


points = grid.points 
connectivity=grid.cell_connectivity
connectivity=connectivity.reshape(grid.n_cells,-1)

#now the "connectivity" is a numpy array with size [n_cells, 4]
# print(grid.n_cells)
# XML processor
Gdml = GdmlElement([1.0, 1.0, 60.0])
Solid = Gdml.set_tetrahedron(connectivity, points)
Gdml.set_material('Vacuum', 'vacuum.ncmat')
Gdml.set_material('B', 'B4C.ncmat')
Gdml.set_logical('B', Solid)
Gdml.set_physical(Solid, position=[0.0, -0.1, 29.0], rotation=[0.0, 180.0, 0.0])
Gdml.export_gdml('res1.gdml')

# for pointsid in connectivity:
#     c_cell = c_cell + 1
#     print(f'There should be 4 points for cell {c_cell}, they are {points[pointsid[0]]} {points[pointsid[1]]} {points[pointsid[2]]} {points[pointsid[3]]}')

print(
    'The following sections should be worked out manually:\n - gun\n - scorer\n'
)