#!/usr/bin/env python3

import numpy as np
import mcpl
from scipy.spatial.transform import Rotation
from Cinema.Prompt.files import MCPLParticle, MCPLBinaryWrite
import argparse

class MCPLProcessor4KdSrc:
    """This class transforms the surface source center to the global original and normal to z-axis
    """
    def __init__(self, inputfile, outputfile, cutoff_eV, cutoff_deg, initial_center, initial_norm, norm_in_deg=True):
        self.inputfile = inputfile
        self.outputfile = outputfile
        self.cutoff_eV = cutoff_eV
        self.cutoff_deg = cutoff_deg
        self.deg = np.pi / 180
        self.sensitivity = 1e-4
        self.initial_center = np.asarray(initial_center)
        self.norm_in_new_frame = np.array([0, 0, 1.])
        if norm_in_deg:
            self.initial_norm = np.cos(np.asarray(initial_norm) * self.deg)
        else:
            self.initial_norm = np.asarray(initial_norm)

    def isparallel(self, vec1, vec2, tolerance=1e-7):
        """
        Check if two vectors are parallel.

        Parameters:
        vec1 (array-like): First vector.
        vec2 (array-like): Second vector.
        tolerance (float): A small tolerance value to account for floating-point precision.

        Returns:
        bool: True if vectors are parallel, False otherwise.
        """
        # Calculate the cross product
        cross_product = np.cross(vec1, vec2)
        
        # Check if the magnitude of the cross product is close to zero
        return np.linalg.norm(cross_product) < tolerance


    def find_plane_normal_and_angles(self, points):
        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid

        # Use Singular Value Decomposition (SVD) to find the normal vector
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[-1]

        # Normalize the normal vector
        normal_norm = np.linalg.norm(normal)
        normal = normal / normal_norm

        # Calculate angles with x, y, and z axes
        angles = [np.arccos(np.abs(np.dot(normal, axis_vector))) for axis_vector in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        angles = [angle * 180 / np.pi for angle in angles]

        return normal, angles

    def process_particles(self):
        n_par_input = mcpl.MCPLFile(self.inputfile).nparticles
        pl = mcpl.MCPLFile(self.inputfile, n_par_input)

        for pb in pl.particle_blocks:
            all_pos = pb.position
            all_dir = pb.direction

        cal_nor, angles = self.find_plane_normal_and_angles(all_pos[:3])
        print('normal and angles', cal_nor, angles)
        print('position mean', all_pos.mean(axis=0))

        if np.linalg.norm(self.initial_norm - cal_nor) > self.sensitivity and np.linalg.norm(self.initial_norm + cal_nor) > self.sensitivity:
            print(np.linalg.norm(self.initial_norm - cal_nor), np.linalg.norm(self.initial_norm + cal_nor))
            raise RuntimeError('given surface normal is not compatible with the data in the MCPL file')
        
        
        addi_vec = np.array([0, 1, 0.])
        if self.isparallel(addi_vec, self.norm_in_new_frame):
            raise RuntimeError('norm_in_new_frame can not be the y axis')

        ncmp_frame = np.array([self.initial_norm, addi_vec])
        rot, _ = Rotation.align_vectors([self.norm_in_new_frame, addi_vec], ncmp_frame)

        new_position = rot.apply(all_pos) - rot.apply(self.initial_center)
        new_dir = rot.apply(all_dir)

        print('rotated and translated results:')
        print('  position mean', new_position.mean(axis=0))
        posmin = new_position.min(axis=0)
        posmax = new_position.max(axis=0)
        print('  position x min and max', posmin[0], posmax[0])
        print('  position y min and max', posmin[1], posmax[1])
        print('  position z min and max', posmin[2], posmax[2])

        mu = np.einsum('i , ji -> j', self.norm_in_new_frame, new_dir)
        print('  mean cos(theta) with the z-axis', mu.mean())

        wrt = MCPLBinaryWrite(self.outputfile)
        mu_cut = np.cos(self.cutoff_deg * self.deg)
        write2file = int(0)

        for par in pl.particles:
            if par.ekin * 1e6 > self.cutoff_eV:
                continue

            rot_dir = rot.apply(par.direction)
            if rot_dir.dot(self.norm_in_new_frame) < mu_cut:
                continue
            
            rot_cen = rot.apply(self.initial_center) 
            rot_position = rot.apply(par.position) - rot_cen
            rot_position[2] += np.linalg.norm(rot_cen)
            ptpar = MCPLParticle(par.ekin, par.polx, par.poly, par.polz,
                                 rot_position[0], rot_position[1], rot_position[2],
                                 rot_dir[0], rot_dir[1], rot_dir[2], par.time, par.weight, par.pdgcode)
            wrt.write(ptpar)
            write2file += 1
        wrt.close()

        print('input particle number:', n_par_input)
        print('output particle number:', write2file)
        return write2file, n_par_input

def main():
    parser = argparse.ArgumentParser(description='Process MCPL files and transform particle data.')
    parser.add_argument('-i', '--inputfile', type=str, default='bl6.mcpl', help='Input MCPL file (default: %(default)s)')
    parser.add_argument('-o', '--outputfile', type=str, default='bl6_cut', help='Output file prefix (default: %(default)s)')
    parser.add_argument('-ce', '--cutoff_eV', type=float, default=10, help='Energy cutoff in eV (default: %(default)s)')
    parser.add_argument('-ca', '--cutoff_deg', type=float, default=5, help='Angle cutoff in degrees (default: %(default)s)')
    parser.add_argument('--initial_center', nargs=3, type=float, default=[32.59,  13.5, 227.22],
                        help='Initial center coordinates (default: %(default)s)')
    parser.add_argument('--initial_norm', nargs=3, type=float, default=[41, 90, 49],
                        help='Initial normal angles. If they are given in degree, the plane normal is calculated using numpy.cos (default: %(default)s)')
    parser.add_argument('--norm_in_deg', type=bool, default=True,
                        help='Specify if the initial normal is given in degrees (default: %(default)s)')

    args = parser.parse_args()

    processor = MCPLProcessor4KdSrc(args.inputfile, args.outputfile, args.cutoff_eV, args.cutoff_deg,
                                    args.initial_center, args.initial_norm, args.norm_in_deg)
    processor.process_particles()

if __name__ == '__main__':
    main()
