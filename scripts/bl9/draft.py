#!/usr/bin/env python3
import pyvista as pv
import json
import os
import numpy as np

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
