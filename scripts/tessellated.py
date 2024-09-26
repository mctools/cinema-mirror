#!/usr/bin/env python3

from Cinema.Prompt import Launcher, Visualiser
from Cinema.Prompt.solid import Box, Tessellated
from Cinema.Prompt.gun import PythonGun

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Sphere, Tube, Trapezoid, ArbTrapezoid, Cone
from Cinema.Prompt.gun import PythonGun
from Cinema.Prompt.scorer import WlSpectrum
import numpy as np
import pyvista

from Cinema.Prompt import Prompt
from Cinema.Prompt.geo import Volume

import pyvista as pv


def is_degenerate(facet, vertices, min_length=1e-3):
    """
    Check if a facet (triangle) is degenerate based on the minimum side length.
    """
    for i in range(3):
        v1 = vertices[facet[i]]
        v2 = vertices[facet[(i + 1) % 3]]
        length = np.linalg.norm(v2 - v1)
        if length < min_length:
            return True
    return False

def remove_degenerate_facets(polydata, min_length=0.02):
    """
    Remove degenerate facets from a pyvista.PolyData object.
    """
    # Extract the faces and points from the polydata
    faces = polydata.faces.reshape((-1, 4))[:, 1:]
    points = polydata.points

    # Identify non-degenerate faces
    valid_faces = []
    for face in faces:
        if not is_degenerate(face, points, min_length):
            valid_faces.append(face)
    
    # Create a new PolyData with only valid faces
    valid_faces = np.hstack([[3] + list(face) for face in valid_faces])
    cleaned_polydata = pv.PolyData(points, valid_faces)
    
    return cleaned_polydata

class MySim(Prompt):
    def __init__(self, seed) -> None:
        super().__init__(seed)

    def makeWorld(self):
        world = Volume('world', Box(10, 10, 20))

        dtt = Volume('detector', Box(2, 2, 2))

        bunny = pyvista.read('./bun_zipper.ply').extract_surface()
        bunny.points *= 10
        bunny = bunny.triangulate()
        bunny = bunny.decimate_pro(0.90, preserve_topology=True)  # Reduce by 90%
        bunny = bunny.fill_holes(1000)  
        bunny = bunny.clean(tolerance=1e-4, absolute=False)


        t = Tessellated(bunny)

        # t = Tessellated(pyvista.Sphere(radius=1, theta_resolution=35, phi_resolution=35))
        tes = Volume('T', t) 
        dtt.placeChild("Tessellated_TP", tes, Transformation3D(0., 0, 0))

        world.placeChild('detectorPhy', dtt, Transformation3D(0,0,0))

        self.setWorld(world)


sim = MySim(seed=4096)
sim.makeWorld()
gunCfg = "gun=MaxwellianGun;src_w=2;src_h=2;src_z=-10;slit_w=2;slit_h=2;slit_z=1e99;temperature=293;"
# sim.show(gunCfg, 100)
sim.simulate(gunCfg, 1e4)



