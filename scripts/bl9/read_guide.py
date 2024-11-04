from OCC.Core.Tesselator import ShapeTesselator
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer, stlapi_Write
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, topods_Face, TopoDS_Shape
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_XYZ
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Ax1
from OCC.Core.Message import Message_ProgressRange
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.TopoDS import topods_Face
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_AbscissaPoint, GCPnts_UniformAbscissa
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_OBB

try:
    import numpy as np

    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

DISPLAY_BOOL = False
if DISPLAY_BOOL:
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.SetSelectionModeVertex()


    def display_show(shape):
        display.EraseAll()
        # display.DisplayShape(shape)
        display.DisplayShape(shape, update=True)


# create the shape
# box_s = BRepPrimAPI_MakeBox(10, 20, 30).Shape()
input_file  = 'guides_simple.step'   # input STEP (AP203/AP214 file)
output_file = 'tessellated_guide.stl'   # output X3D file

def read_step(input_file):
    step_reader = STEPControl_Reader()
    step_reader.ReadFile( input_file )
    step_reader.TransferRoot()
    myshape = step_reader.Shape()
    print("File readed")
    return myshape

def getobb(shape):
    obb2 = Bnd_OBB()
    brepbndlib.AddOBB(shape, obb2, True, True, True)
    obb_shape2, center, zvec = convert_bnd_to_shape(obb2)
    return obb_shape2, center, zvec
    # display.DisplayShape(shape)
    # display.DisplayShape(obb_shape2, transparency=0.5, update=True)

def convert_bnd_to_shape(the_box):
    """Converts a bounding box to a box shape."""
    barycenter = the_box.Center()
    x_dir = the_box.XDirection()
    y_dir = the_box.YDirection()
    z_dir = the_box.ZDirection()
    half_x = the_box.XHSize()
    half_y = the_box.YHSize()
    half_z = the_box.ZHSize()

    x_vec = gp_XYZ(x_dir.X(), x_dir.Y(), x_dir.Z())
    y_vec = gp_XYZ(y_dir.X(), y_dir.Y(), y_dir.Z())
    z_vec = gp_XYZ(z_dir.X(), z_dir.Y(), z_dir.Z())
    point = gp_Pnt(barycenter.X(), barycenter.Y(), barycenter.Z())
    axes = gp_Ax2(point, gp_Dir(z_dir), gp_Dir(x_dir))
    axes.SetLocation(
        gp_Pnt(point.XYZ() - x_vec * half_x - y_vec * half_y - z_vec * half_z)
    )
    box = BRepPrimAPI_MakeBox(axes, 2.0 * half_x, 2.0 * half_y, 2.0 * half_z).Solid()
    return box, point.Coord(), z_vec.Coord()

def extract_edge_direction(edge) -> gp_Vec:  
    ex = TopologyExplorer(edge)
    vertex = list(ex.vertices_from_edge(edge))
    if len(vertex) !=2:
        raise ValueError("Check number of vertex is 2 !")
    start_point = BRep_Tool.Pnt(vertex[0])  
    end_point = BRep_Tool.Pnt(vertex[1])  
    direction_vector = gp_Vec(start_point, end_point)  
    return direction_vector.Normalized()

def recognize_edge(edge, matched_dir, degree_tolerance = 2):
    dir = extract_edge_direction(edge).XYZ().Coord()
    dir = np.array(dir)
    cosine = get_cosine(dir, matched_dir)
    angle = np.rad2deg(np.arccos(cosine))
    tolerance_cos = np.cos(np.deg2rad(degree_tolerance))
    if not angle_is_between_tolerance(cosine, tolerance_cos):
        return False
    # print(f"------> Edge direction angle: {angle}")
    return True


def explore(myshape, verbose = False):
    ex = TopologyExplorer(myshape)
    box_dict = {}
    box_order = 1
    for solid in ex.solids():
    # solids = list(ex.solids())
    # solid = solids[0]
        box_name = f'Solid_{box_order}'
        box_dict[box_name] = {}
        if verbose:
            print(f'--> {box_name}: ')
        obb, center, zvec = getobb(solid)
        box_dimensions = get_box_dimensions(obb, verbose)
        box_dict[box_name]['dimension'] = box_dimensions
        box_dict[box_name]['location'] = np.round(np.array(center), 6)
        box_dict[box_name]['zVector'] = np.round(np.array(zvec), 6)
        box_order += 1
    return box_dict

        # for face in ex.faces_from_solids(solid):
        #     found = recognize_face(face)
        #     if found:
        #         for edge in ex.edges_from_face(face):
        #             print(length_from_edge(edge))

def get_box_dimensions(box, verbose):
    ex_obb = TopologyExplorer(box)
    faces_num = ex_obb.number_of_faces()
    if not faces_num == 6:
        raise ValueError(f"Box with {faces_num} faces, 6 expected")
    dimensions = []
    x_length = []
    y_length = []
    z_length = []

    bool_xy_done = False
    bool_z_done = False
    for face in ex_obb.faces():
        edge_num = ex_obb.number_of_edges_from_face(face)
        if edge_num != 4:
            raise ValueError(f"Face with {edge_num} edges, 4 expected")

        # if face normal to X
        if face_normal_xyz(face, 5) == (1,0,0) and not bool_xy_done:
            for edge in ex_obb.edges_from_face(face):
                if recognize_edge(edge, [0,0,1], 5):
                    z_length.append(length_from_edge(edge))
                elif recognize_edge(edge, [0,1,0], 5):
                    y_length.append(length_from_edge(edge))
                else:
                    raise ValueError(f"Edge direction not matching Y or Z")
            bool_xy_done = True
        # if face normal to Y
        elif face_normal_xyz(face, 5 ) == (0,1,0) and not bool_z_done:
            for edge in ex_obb.edges_from_face(face):
                if recognize_edge(edge, [1,0,0], 5):
                    x_length.append(length_from_edge(edge))
            bool_z_done =True
        
        if len(x_length) ==2 and len(y_length) == 2 and len(z_length) == 2:
            break
            
    dimensions.append(x_length)
    dimensions.append(y_length)
    dimensions.append(z_length)
    box_dimensions = []
    for dim in dimensions:
        if len(dim) != 2:
            raise ValueError("Edge numbers should be both 2")
        dim = np.round(np.array(dim), 6)
        box_dim = np.unique(dim)
        if len(box_dim) != 1:
            raise ValueError("Final box dimension numbers should be respectively 1")
        box_dimensions.append(box_dim[0])
    return np.array(box_dimensions)

        


def length_from_edge(edg):
    curve_adapt = BRepAdaptor_Curve(edg)
    length = GCPnts_AbscissaPoint().Length(
        curve_adapt, curve_adapt.FirstParameter(), curve_adapt.LastParameter(), 1e-6
    )
    return length

def mesh(shape):
    # compute the tessellation
    tess = ShapeTesselator(shape)
    tess.Compute()
    # get the triangles
    triangle_count = tess.ObjGetTriangleCount()
    triangles = []

    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    for i_triangle in range(0, triangle_count):
        i1, i2, i3 = tess.GetTriangleIndex(i_triangle)
        triangles.append([tess.GetVertex(i1), tess.GetVertex(i2), tess.GetVertex(i3)])
        for j in range(1, 4):
            if j == 1:
                m = i1
                n = i2
            elif j == 2:
                n = i3
            elif j == 3:
                m = i2
            p1 = tess.GetVertex(m)
            p2 = tess.GetVertex(n)
            me = BRepBuilderAPI_MakeEdge(gp_Pnt(p1[0],p1[1],p1[2]), gp_Pnt(p2[0],p2[1],p2[2]))
            if me.IsDone():
                builder.Add(comp, me.Edge())

    return comp

def face_normal_xyz(face, degree_tolerance):
    dirs = [[1,0,0],
            [0,1,0],
            [0,0,1]]
    is_normal_x = recognize_face(face, dirs[0], degree_tolerance)
    is_normal_y = recognize_face(face, dirs[1], degree_tolerance)
    is_normal_z = recognize_face(face, dirs[2], degree_tolerance)
    return int(is_normal_x), int(is_normal_y), int(is_normal_z)

def recognize_face(a_face, matched_dir: list, degree_tolerance):
    """Takes a TopoDS shape and tries to identify its nature
    whether it is a plane a cylinder a torus etc.
    if a plane, returns the normal
    if a cylinder, returns the radius
    """
    surf = BRepAdaptor_Surface(a_face, True)
    surf_type = surf.GetType()
    if surf_type == GeomAbs_Plane:
        # print("--> plane")
        # look for the properties of the plane
        # first get the related gp_Pln
        gp_pln = surf.Plane()
        location = gp_pln.Location()  # a point of the plane
        normal = gp_pln.Axis().Direction()  # the plane normal
        # then export location and normal to the console output
        # print(
        #     "--> Location (global coordinates)",
        #     location.X(),
        #     location.Y(),
        #     location.Z(),
        # )
        # print("--> Normal (global coordinates)", normal.X(), normal.Y(), normal.Z())
        dir1 = np.array(matched_dir)
        dir2 =  np.array([normal.X(), normal.Y(), normal.Z()])
        cosine = get_cosine(dir1, dir2)
        tolerance_cos = np.cos(np.deg2rad(degree_tolerance))
       
        if angle_is_between_tolerance(cosine, tolerance_cos) :
            # print(f"----> Face Normal Angle: {np.rad2deg(np.arccos(cosine))}")
            return True
        else:
            return False
    else:
        raise ValueError("Plane not found!")

def angle_is_between_tolerance(angle_cos, tolerance_cos):
    return abs(angle_cos) > abs(tolerance_cos)

def get_cosine(dir1, dir2):
    dir1 = np.array(dir1)
    dir2 = np.array(dir2)
    return np.dot(dir1, dir2) / (np.linalg.norm(dir1)*np.linalg.norm(dir2))


def geo_transform(shape:TopoDS_Shape):
    trns = gp_Trsf()
    # trns.SetTranslation(gp_Vec(-50, 0, 0))
    rotation_axe = gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,1,0))
    trns.SetRotation(rotation_axe, np.deg2rad(-90))
    transformed_shape = BRepBuilderAPI_Transform(shape, trns, False).Shape()
    return transformed_shape

def export(shape, numMark):
    import os
    pr = Message_ProgressRange()
    stl_writer = StlAPI_Writer()
    stl_writer.SetASCIIMode(True)
    # print(shape)
    fpath = f'{numMark}_{output_file}'
    success = stl_writer.Write(shape, fpath, pr)
    if success:
        print(f'File written to {fpath}')
    else:
        raise IOError(f"File {fpath} not written.")


if __name__ == "__main__":
    myshape = read_step(input_file)
    transformed_shape = geo_transform(myshape)
    explore(transformed_shape)
    # shape = mesh(transformed_shape)
    # if DISPLAY_BOOL:
    #     display_show(shape)
    #     start_display()
    # export(transformed_shape)
