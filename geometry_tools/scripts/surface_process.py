""" Clip boundaries, mark inlets, outlets, and aneurysm locations.
"""

from pathlib import Path 
import pyvista as pv 
from geometry_tools.meshing import Mesher
from geometry_tools import common as cc
import sys 

import time
from datetime import timedelta
from tubeclipper import TubeClipper

def surface_process(proj_dir, endpoints_pv=None):
    proj_dir = Path(proj_dir)

    clipped_dir = proj_dir / '01_clipped' 
    surf_file = sorted(clipped_dir.glob('*.vtp'))[0]

    points_dir = proj_dir / '01_points' 
    neckpoints_dir = proj_dir / '01_neckpoints' 

    point_file = points_dir / (surf_file.stem + '_endpoints.vtm')
    neckpoints_file = neckpoints_dir / (surf_file.stem + '_neckpoints.vtm')
    print(point_file)
    assert point_file.exists()
    assert neckpoints_file.exists()

    surf_output_dir = proj_dir / '02_processed' 
    points_output_dir = proj_dir / '02_points' 
    endpoints_output_dir = proj_dir / '02_endpoints' # For pre-extension points

    for f in [surf_output_dir, points_output_dir, endpoints_output_dir]:
        if not f.exists():
            f.mkdir(parents=True, exist_ok=True)

    surf_output_file = surf_output_dir / (surf_file.stem + '_pr.vtp')
    points_output_file = points_output_dir / (surf_file.stem + '_pr_endpoints.vtm')
    endpoints_output_file = endpoints_output_dir / (surf_file.stem + '_pr_endpoints.vtp')

    start = time.time()
    # print('\n' + surf_file.stem)

    if not surf_output_file.exists():
        surf = pv.read(surf_file)
        # Make option for smoothing here
        # surf = cc.vtk_taubin_smooth(surf, pass_band=0.05, iterations=50)

        neck_geodesic_points = pv.read(neckpoints_file)

        if neck_geodesic_points is not None:
            s = cc.SelectGeodesic(surf, scalars='Mask')
            s.use_stored_points(neck_geodesic_points)
            surf = s.mesh

        points = pv.read(point_file)
        in_points = points['inlets'].points
        out_points = points['outlets'].points
        an_points = points['aneurysms'].points

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
            surf_closed = surf.fill_holes(20.0)
            select = endpoints_pv.select_enclosed_points(surf_closed, tolerance=0.00001)
            
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
            )

        m.update_inlets_outlets()

        if m.surf.n_points > 1000:
            m.decimate_surface(target_edge_length=0.5)

        m.generate_centerlines()
        m.generate_centerlines(include_aneurysms=False)
        m.branch_centerlines()

        # outlet_clip = m.clip_endpoints_with_spheres()
        endpoints_pv = m.clip_endpoints_with_tubeclipper(endpoints_pv=endpoints_pv)
        endpoints_pv.save(endpoints_output_file)

        m.update_inlets_outlets()
        
        m.get_bifurcation_ref_systems_vectors()

        m.surface_preparation(neck_points=neck_geodesic_points, max_size=0.4, min_size=0.15)
        m.update_inlets_outlets()

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

        m.generate_centerlines()
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
        m.save_inlet_outlet_points(points_output_file)


        print('\n Case done', timedelta(seconds=time.time() - start))

    else:
        print('Output surface exists.')



if __name__ == "__main__":
    proj_dir = Path(sys.argv[1])

    if len(sys.argv > 2):
        endpoints_f = Path(sys.argv[2])
        endpoints_pv = pv.read(endpoints_f)
    else:
        endpoints_pv = None

    surface_process(proj_dir, endpoints_pv)
