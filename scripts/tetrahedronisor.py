#import pyvista as pv
#mesh = pv.read('reconstruction/bun_zipper.ply')


import pyvista as pv
import tetgen
import numpy as np
from lxml import etree

class GdmlElement(etree.Element):
    '''
    
    '''
    def __init__(self):
        
        self.gdml = etree.Element('gdml')
        self.userinfo = etree.SubElement(self.gdml, 'userinfo')
        self.define = etree.SubElement(self.gdml, 'define')
        self.materials = etree.SubElement(self.gdml, 'materials')
        self.solids = etree.SubElement(self.gdml, 'solids')
        self.structure = etree.SubElement(self.gdml, 'structure')
        self.set_world()

    def set_world(self, xyz = [100.0, 100.0, 100.0]):

        sol_world = etree.SubElement(self.solids, 'box')
        sol_world.set('lunit', 'mm')
        sol_world.set('name', 'sol_world')
        sol_world.set('x', f'{xyz[0]}')
        sol_world.set('y', f'{xyz[1]}')
        sol_world.set('z', f'{xyz[2]}')

        self.set_logical('Vacuum', sol_world)
        self.setup = etree.SubElement(self.gdml, 'setup')
        self.setup.set('name', 'Default')
        self.setup.set('version', '1.0')
        self.setup_world = etree.SubElement(self.setup, 'world')
        self.setup_world.set('ref', f'vol_{sol_world.get("name")}')

    def set_position(self, name, xyz = [], element = None):

        position = etree.Element('position')
        position.set('name', name)
        position.set('x', f'{xyz[0]}')
        position.set('y', f'{xyz[1]}')
        position.set('z', f'{xyz[2]}')
        if not element:
            self.define.append(position)
        elif isinstance(element, etree.Element):
            element.append(position)
        else:
            raise TypeError('Setting position subelement failed. Please check element.')
            
    def set_tetrahedron(self, cells):
        
        i_cell = 0
        for cell in cells:
            i_cell = i_cell + 1
            i_point = 0
            for point in cell:
                i_point = i_point + 1
                self.set_position(f'cell{i_cell}_v{i_point}', point)
            tetrahedron = etree.Element('tet')
            tetrahedron.set('name', f'cell{i_cell}')
            tetrahedron.set('vertex1', f'cell{i_cell}_v1')
            tetrahedron.set('vertex2', f'cell{i_cell}_v2')
            tetrahedron.set('vertex3', f'cell{i_cell}_v3')
            tetrahedron.set('vertex4', f'cell{i_cell}_v4')
            self.solids.append(tetrahedron)

    def set_logical(self, mat_element, sol = None):

        if not sol:
            iterator = self.solids.iter()
        else:
            iterator = sol.iter()

        for solid in iterator():
            vol = etree.Element('volume')
            vol.set('name', f'vol_{solid.get("name")}')

            solid = etree.SubElement(vol, 'solidref')
            solid.set('ref', solid.get("name"))

            mat = etree.SubElement(vol, 'materialref')
            mat.set('ref', mat_element)
            self.structure.append(vol)

    def set_physical(self, sol, position = [0.0, 0.0, 0.0], parent = None):

        if not sol:
            iterator = self.solids.iter()
        else:
            iterator = sol.iter()

        for child in iterator:
            pVol = etree.Element('physvol')
            volRef = etree.SubElement(pVol, 'volumeref')
            volRef.set('ref', f'vol_{child.get("name")}')
            positionVol = etree.SubElement(pVol, 'position')
            positionVol.set('name', f'p_{child.get("name")}')
            positionVol.set('unit', 'mm')
            positionVol.set('x', f'{position[0]}')
            positionVol.set('y', f'{position[1]}')
            positionVol.set('z', f'{position[2]}')
        
            if not parent:
                self.world.append(pVol)
            else:
                parent.append(pVol)


pv.set_plot_theme('document')

bunny = pv.read('../../files/bunny/reconstruction/bun_zipper_res4.ply')
# bunny.plot()
tet = tetgen.TetGen(bunny)
tet.make_manifold()
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
grid = tet.grid #grid is of UnstructuredGrid type
exploded = grid.explode() # explode view
# exploded.plot(show_axes=True)

# pter = pv.Plotter()
# pter.add_mesh(grid, style='wireframe', opacity=0.5)
# pter.add_mesh(exploded, show_edges=True)
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
gdml = etree.Element('gdml')
c_cell = 0 
define = etree.SubElement(gdml, 'define')
solids = etree.SubElement(gdml, 'solids')
structure = etree.SubElement(gdml, 'structure')
volWorld = set_logical_volume('volWorld', 'solidWorld', 'Vacuum')
structure.append(volWorld)

for pointsid in connectivity:
    c_cell = c_cell + 1
    # print(f'There should be 4 points for cell {c_cell}, they are {points[pointsid[0]]} {points[pointsid[1]]} {points[pointsid[2]]} {points[pointsid[3]]}')
    for vertex in [0,1,2,3]:
        position = etree.Element('position')
        position.set('name', f'cell{c_cell}_v{vertex+1}')
        position.set('x', f'{points[pointsid[vertex]][0]}')
        position.set('y', f'{points[pointsid[vertex]][1]}')
        position.set('z', f'{points[pointsid[vertex]][2]}')
        define.append(position)
    tetrahedron = etree.Element('tet')
    tetrahedron.set('name', f'cell{c_cell}')
    tetrahedron.set('vertex1', f'cell{c_cell}_v1')
    tetrahedron.set('vertex2', f'cell{c_cell}_v2')
    tetrahedron.set('vertex3', f'cell{c_cell}_v3')
    tetrahedron.set('vertex4', f'cell{c_cell}_v4')
    solids.append(tetrahedron)
    structure.append(set_logical_volume(f'vol{c_cell}', f'cell{c_cell}', 'B'))
    volWorld.append(set_physical_volume(f'vol{c_cell}', f'p_{c_cell}'))

output = etree.tostring(gdml, pretty_print=True)
with open('solid.gdml', 'wb') as file:
    file.write(output)

print(
    'The following sections should be worked out manually:\n - materials\n - userinfo\n - setup\n'
)