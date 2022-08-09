""" Clip boundaries, mark inlets, outlets, and aneurysm locations.
"""

from configparser import NoOptionError
import sys
from pathlib import Path 
import pyvista as pv 
from geometry_tools.meshing import Mesher
from geometry_tools import common as cc

import time
from datetime import timedelta
from tubeclipper import TubeClipper

def surface_process(proj_dir, surf_type, endpoints_pv=None):
    proj_dir = Path(proj_dir)

    surf_file = sorted(proj_dir.glob('*.stl'))[0]

    point_file = proj_dir / (surf_file.stem + '_endpoints.vtm')
    #print(point_file)
    assert point_file.exists()
    
    if surf_type=='a':    
        neckpoints_file = proj_dir / (surf_file.stem + '_neckpoints.vtm')    
        assert neckpoints_file.exists()
        anubool=True

    surf_output_file = proj_dir / (surf_file.stem + '_pr.vtp')
    points_output_file = proj_dir / (surf_file.stem + '_pr_endpoints.vtm')
    endpoints_output_file = proj_dir / (surf_file.stem + '_pr_endpoints.vtp')

    start = time.time()
    #print('\n' + surf_file.stem)

    if not surf_output_file.exists():
        surf = pv.read(surf_file)
        # Make option for smoothing here
        # surf = cc.vtk_taubin_smooth(surf, pass_band=0.05, iterations=50)
        if surf_type=='pt':
            neck_geodesic_points = None
        else:
            neck_geodesic_points = pv.read(neckpoints_file)

            if neck_geodesic_points is not None:
                s = cc.SelectGeodesic(surf, scalars='Mask')
                s.use_stored_points(neck_geodesic_points)
                surf = s.mesh

        points = pv.read(point_file)
        in_points = points['inlets'].points
        out_points = points['outlets'].points
        if surf_type=='a':   
            an_points = points['aneurysms'].points
            anubool=True
        else:
            an_points = None
            anubool = False

        if 'normals' in points['inlets'].point_arrays:
            # Clip
            for pdx in range(points['inlets'].n_points):
                t = TubeClipper(surf)
                t.clip(points['inlets'].points[pdx], points['inlets'].point_arrays['normals'][pdx])
                surf = t.far_side
        
        if 'normals' in points['outlets'].point_arrays:
            # Clip
            for pdx in range(points['outlets'].n_points):
                t = TubeClipper(surf)
                t.clip(points['outlets'].points[pdx], -points['outlets'].point_arrays['normals'][pdx])
                surf = t.far_side

        if endpoints_pv is not None:
            print('Clipping with endpoints')
            surf_closed = surf.fill_holes(20.0) #not sure about this, may have to make a flag
            select = endpoints_pv.select_enclosed_points(surf_closed, tolerance=0.00001) #this doesn't work with a MuliBlock object
            
            for sdx in range(select.n_points):
                check = select.point_arrays['SelectedPoints'][sdx]
                if check == True:
                    print('t')
                    origin = select.points[sdx]
                    normal = select.point_arrays['normals'][sdx]
                    t = TubeClipper(surf)
                    t.clip(origin, normal)
                    surf = t.far_side

        m = Mesher(
            surf,
            inlet_points=in_points, 
            outlet_points=out_points,
            aneurysm_points=an_points,
            include_aneurysms=anubool
            )

        m.update_inlets_outlets()

        if m.surf.n_points > 1000:
            m.decimate_surface(target_edge_length=0.5)
        
        if surf_type=='a': 
            m.generate_centerlines()
        m.generate_centerlines(include_aneurysms=False)
        if surf_type=='a': 
            m.branch_centerlines()

        # outlet_clip = m.clip_endpoints_with_spheres()
        endpoints_pv = m.clip_endpoints_with_tubeclipper(endpoints_pv=endpoints_pv, include_aneurysms=anubool) #includes aneurysms by default
        endpoints_pv.save(endpoints_output_file)

        m.update_inlets_outlets()
        if surf_type=='a': 
            m.get_bifurcation_ref_systems_vectors()
        m.surface_preparation(neck_points=neck_geodesic_points, min_edge_size=0.1, max_edge_size=0.4, sac_size=0.15, misr_min=0.1, misr_max=2.5)
        m.update_inlets_outlets()
            
            #going to need to alter surface_preparation to accomodate PT, but maybe already good?

        m.surf.save(surf_output_file)

        # # This stuff is for getting plc points and 
        # # parent regions for SCI
        # m.get_mean_segments()
        # m.branch_centerlines()
        # m.update_aneurysm_group_ids()
        # m.get_branch_endpoints()

        # n_spheres=4
        # m.mark_distance_from_sacs(n_spheres)
        # m.mark_near_vessel_regions(m.surf, n_spheres=n_spheres)
        # # plc_points = m.get_plc_points() 

        # m.surf.save(surf_output_file)

        m.generate_centerlines(include_aneurysms=anubool)
        if surf_type=='a': 
            m.generate_centerlines(include_aneurysms=False)
            m.branch_centerlines()
            m.update_inlets_outlets()
            m.get_bifurcation_ref_systems_vectors()
            m.update_inlets_outlets()
            m.get_mean_segments()
            m.update_aneurysm_group_ids()
            m.get_branch_endpoints()
            n_spheres=4
            m.mark_distance_from_sacs(n_spheres)
            m.mark_near_vessel_regions(m.surf, n_spheres=n_spheres)

        # m.surf.plot(scalars='sac_zones')
        m.surf.save(surf_output_file)
        m.update_inlets_outlets()
        m.save_inlet_outlet_points(points_output_file, include_aneurysms=anubool)


        print('\n Case done', timedelta(seconds=time.time() - start))

    else:
        print('Output surface exists.')



if __name__ == "__main__":
    proj_dir = Path(sys.argv[1])

    if len(sys.argv) == 3:
        endpoints_f = Path(sys.argv[2])
        if endpoints_f.exists():
            endpoints_pv = pv.read(endpoints_f) #have to be PolyData type (vtp) to work?
        else:
            endpoints_pv = None
            surf_type=sys.argv[2]    
    elif len(sys.argv) > 3:
        endpoints_f = Path(sys.argv[2])
        endpoints_pv = pv.read(endpoints_f)
        surf_type=sys.argv[3]

    surface_process(proj_dir, surf_type, endpoints_pv)
