""" Clip boundaries, mark inlets, outlets, and aneurysm locations.
"""

from pathlib import Path 
import pyvista as pv 
from geometry_tools.meshing import Mesher
from geometry_tools import common as cc
import time
from datetime import timedelta
import sys 

def surface_prep(surf_file, proj_dir):
    """ Basic surface prep.
    """
    surf_file = Path(surf_file)
    proj_dir = Path(proj_dir)
    surf_output_dir = proj_dir / '01_clipped' 
    points_output_dir = proj_dir / '01_points' 
    neckpoints_output_dir = proj_dir / '01_neckpoints' 

    for f in [surf_output_dir, points_output_dir, neckpoints_output_dir]:
        if not f.exists():
            f.mkdir(parents=True)

    # Output files
    surf_file_out = surf_output_dir / (surf_file.stem + '_cl.vtp')
    neck_file_out = neckpoints_output_dir / (surf_file.stem + '_cl_neckpoints.vtm')
    points_file_out = points_output_dir / (surf_file.stem + '_cl_endpoints.vtm')

    if not surf_file_out.exists():
        case_start = time.time()

        surf = pv.read(surf_file)
        surf = surf.compute_normals(auto_orient_normals=False)
        
        m = Mesher(surf)

        # Uses m.clip_boundaries uses cc.ClickDragDelete to delete boundaries.
        flag_inspect = m.clip_boundaries()

        # Delete any sharp edges or small branches
        while flag_inspect == True:
            s = cc.SelectGeodesic(m.surf, scalars='Delete', )
            s.interact(title='Isolate region to delete and fill.')
            m.surf = s.mesh

        m.set_inlets_outlets()
        m.pick_aneurysm()
        m.copy_structure()

        # Select aneurysms
        s = cc.SelectGeodesic(m.surf)
        s.interact(title='Isolate aneurysms.')
        s.save_stored_points(neck_file_out)

        m.surf = s.mesh 
        m.surf.save(surf_file_out)
        m.generate_centerlines()
        m.save_inlet_outlet_points(points_file_out)

        time_spent = time.time() - case_start
        print('Case done', timedelta(seconds=time_spent))

    else:
        print('Output file exists.')


if __name__ == "__main__":
    surf_file = sys.argv[1]
    proj_dir = sys.argv[2]
    surface_prep(surf_file, proj_dir)