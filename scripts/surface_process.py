""" Clip boundaries, mark inlets, outlets, and aneurysm locations.
"""

from pathlib import Path 
import pyvista as pv 
from geometry_tools.meshing import Mesher
from geometry_tools import common as cc
import sys 

import time
from datetime import timedelta


if __name__ == "__main__":
    # surf_file = Path(sys.argv[1])
    proj_dir = Path(sys.argv[1]) #surf_file.parents[1]

    clipped_dir = proj_dir / '01_clipped' 
    surf_file = sorted(clipped_dir.glob('*.vtp'))[0]

    points_dir = proj_dir / '01_points' 
    neckpoints_dir = proj_dir / '01_neckpoints' 

    point_file = points_dir / (surf_file.stem + '_endpoints.vtm')
    neckpoints_file = neckpoints_dir / (surf_file.stem + '_neckpoints.vtm')
    assert point_file.exists()
    assert neckpoints_file.exists()

    surf_output_dir = proj_dir / '02_processed' 
    points_output_dir = proj_dir / '02_points' 

    for f in [surf_output_dir, points_output_dir]:
        if not f.exists():
            f.mkdir(parents=True, exist_ok=True)

    surf_output_file = surf_output_dir / (surf_file.stem + '_pr.vtp')
    points_output_file = points_output_dir / (surf_file.stem + '_pr_endpoints.vtm')
    start = time.time()
    # print('\n' + surf_file.stem)

    if not surf_output_file.exists():
        try:
            surf = pv.read(surf_file)
            neck_geodesic_points = pv.read(neckpoints_file)

            points = pv.read(point_file)
            in_points = points['inlets'].points
            out_points = points['outlets'].points
            an_points = points['aneurysms'].points
                    
            m = Mesher(
                surf,
                inlet_points=in_points, 
                outlet_points=out_points,
                aneurysm_points=an_points,
                )

            m.update_inlets_outlets()

            m.decimate_surface(target_edge_length=0.5)

            m.generate_centerlines()
            m.generate_centerlines(include_aneurysms=False)
            m.branch_centerlines()

            # outlet_clip = m.clip_endpoints_with_spheres()
            m.clip_endpoints_with_tubeclipper()
            
            m.update_inlets_outlets()
            
            m.get_bifurcation_ref_systems_vectors()

            # Here, use the previous neck geodesics.
            s = cc.SelectGeodesic(m.surf)
            s.use_stored_points(neck_geodesic_points)
            
            m.surf = s.mesh

            # m.get_mean_segments()

            m.surface_preparation(neck_geodesic_points)
            m.update_inlets_outlets()

            m.surf.save(surf_output_file)
            m.save_inlet_outlet_points(points_output_file)

            print('\n Case done', timedelta(seconds=time.time() - start))

        except Exception as e:
            print(e)
            print(surf_file, '*'*50, 'FAILED')
                    
    else:
        print('Output surface exists.')


