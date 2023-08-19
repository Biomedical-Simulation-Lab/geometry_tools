"""
This file includes a method to create network centerlines for a surface mesh, which contains the VMTK
attributes, as well as the cross-sectional area and perimeter of the surface mesh.
Commented out lines may be useful. Two other ways to generate centerlines included.

Call this using:

make_centerlines.py proj_dir case_name

where proj_dir is the directory you are looking for the surface file in (should end in "cl_mapped.vtp"),
and the case_name is what you want your centerline to be called ("case_name_centerline.vtp")
"""

import numpy as np
import pyvista as pv
from geometry_tools.meshing import Mesher
from geometry_tools import vmtk_wrapper as vmtk
from geometry_tools import common as cc
from scipy.spatial import cKDTree as KDTree 
from pathlib import Path
import sys

def make_cl(proj_dir, case_name):
    out_dir = proj_dir
    surf_file = sorted(proj_dir.glob('*cl.vtp'))[0]
    surf=pv.read(surf_file)

    m = Mesher(
            surf,
            include_aneurysms=False
            )
    #m.set_inlets_outlets()
    #m.generate_centerlines(include_aneurysms=False, endpoints=1)
    
    centers = m.get_open_profiles()
    inlet_id = m._pick_points(pv.wrap(centers), 'Pick Major Inlet')
    outlet_id = m._pick_points(pv.wrap(centers), 'Pick Major Outlet')
    m.inlet_ids = [inlet_id]
    m.inlet_points = centers[inlet_id]
    m.outlet_ids = [outlet_id]
    m.outlet_points = centers[outlet_id]
    m.generate_centerlines(include_aneurysms=False, endpoints=1)
    #_ , graph = vmtk.network_extractor(surf, ratio = 1.01)
    #graph.save(out_dir/('graph.vtp'))
    graph = pv.read(out_dir/('graph.vtp'))
    inlet_point = graph.points[2]
    tree2 = KDTree(m.centerlines.points)
    out_point = graph.points[5]
    out_id=tree2.query(out_point, k=1)[1]
    outlet_point = m.centerlines.points[out_id]

    surf_capped = pv.PolyData()
    surf_capped.copy_structure(vmtk.surface_capper(surf))
    tree = KDTree(surf_capped.points)
    inlet_ids = tree.query(inlet_point, k=1)[1]
    
    outlet_ids = tree.query(outlet_point, k=1)[1]
    centerlines_seg = vmtk.centerlines(
        surf_capped, 
        seed_selector='idlist', 
        resampling_step_length = 1,
        src_ids=[inlet_ids],
        target_ids=[outlet_ids]
        )
    #make sure to set the first and last points to the inlet and outlet points
    centerlines_seg.points[0]=m.centerlines.points[tree2.query(outlet_point, k=1)[1]]
    centerlines_seg.points[-1]=inlet_point
    centerlines = centerlines_seg

    def polyline_from_points(points):
        poly = pv.PolyData()
        poly.points = points
        the_cell = np.arange(0, len(points), dtype=np.int_)
        the_cell = np.insert(the_cell, 0, len(points))
        poly.lines = the_cell
        return poly

    points = m.centerlines.points[0:out_id]
    line = polyline_from_points(points)
    centerlines +=line
    points2 = m.centerlines.points[out_id:]
    line2 = polyline_from_points(points2)
    centerlines +=line2
    centerlines = vmtk.centerline_geometry(centerlines)
    '''
    m.centerlines, _ = vmtk.network_extractor(m.surf)
    m.centerlines = vmtk.resample_cl(m.centerlines, length=1.5) #make sure this matches with other files!
    m.centerlines = vmtk.centerline_geometry(m.centerlines)

    tree1 = KDTree(m.centerlines.points)
    tree2 = KDTree(m.surf.points)
    dist, idx = tree2.query(m.centerlines.points) #closest dist to centerline point
    for i in range(len(idx)):
        for j in range(len(idx)):
            if (idx[j]==idx[i]) and (i != j):
                closest, _ = tree1.query(m.surf.points[idx[i]]) #closest centerline distance to the point
                dist[j]=closest
    m.centerlines.point_data['MaximumInscribedSphereRadius']=dist
    #p = pv.Plotter()
    #p.add_mesh(surf, color="gray", opacity=0.5)
    #p.add_mesh(centerline, color="red", point_size=20)
    #p.add_mesh(m.centerlines, color="blue")
    #p.show()
    #print(centerline)
    #Use the centerline points to create planes
    m.centerlines.point_data['CSA']=np.array(m.centerlines.n_points)
    m.centerlines.point_data['perimeter']=np.array(m.centerlines.n_points)
    points = m.centerlines.points
    normals = m.centerlines.point_data['FrenetTangent'] 
    for ndx, pt in enumerate(points):
        plane=pv.Plane(center = pt, direction = normals[ndx], i_size=20, j_size=20, i_resolution=100, j_resolution=100).triangulate()
        plane_split = plane.clip_surface(surf)
        split = plane_split.split_bodies()
        if len(split)>1:
            cm = np.zeros((len(split),3))
            for i in range(len(split)):
                cm[i, :] = split[i].center_of_mass()
            tree = KDTree(cm)
            _, j = tree.query(pt) #closest center of mass to the centerline point
            plane_split = split[j]
        CSsurf = plane_split.extract_surface() 
        area = CSsurf.area
        edges = CSsurf.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
        #p=pv.Plotter()
        #p.add_mesh(CSsurf)
        #p.add_mesh(edges, color='red')
        #p.show()
        sized = edges.compute_cell_sizes()
        #print(sized)
        perimeter = sum(sized['Length'])
        m.centerlines.point_data['CSA'][ndx]=area
        m.centerlines.point_data['perimeter'][ndx]=perimeter
    '''
    centerlines.save(out_dir/(case_name+'_centerline.vtp'))

if __name__ == "__main__":
    proj_dir = Path(sys.argv[1])
    case_name = sys.argv[2] 
    make_cl(proj_dir=proj_dir, case_name=case_name)
