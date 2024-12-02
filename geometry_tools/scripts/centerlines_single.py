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
    surf_file = sorted(proj_dir.glob('*_cl.vtp'))[0]
    surf=pv.read(surf_file)

    m = Mesher(
            surf,
            include_aneurysms=False
            )
    centers = m.get_open_profiles()
    inlet_id = m._pick_points(pv.wrap(centers), 'Pick Major Inlet')
    outlet_id = m._pick_points(pv.wrap(centers), 'Pick Major Outlet')
    m.inlet_ids = [inlet_id]
    m.inlet_points = centers[inlet_id]
    m.outlet_ids = [outlet_id]
    m.outlet_points = centers[outlet_id]
    m.generate_centerlines(include_aneurysms=False, endpoints=1)
    m.centerlines = vmtk.centerline_geometry(m.centerlines)
    m.centerlines = vmtk.resample_cl(m.centerlines,0.75)
    m.centerlines.save('{}_centerline_single.vtp'.format(case_name))

if __name__ == "__main__":
    proj_dir = Path(sys.argv[1])
    case_name = sys.argv[2] 
    make_cl(proj_dir=proj_dir, case_name=case_name)