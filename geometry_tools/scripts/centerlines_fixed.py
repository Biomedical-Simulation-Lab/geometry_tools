"""
This file includes a method to create vmtk centerlines for a surface mesh, which contains the VMTK
attributes.

Call this using:

centerlines_fixed.py proj_dir case_name

where proj_dir is the directory you are looking for the surface file in (should end in "cl_mapped.vtp"),
and the case_name is what you want your centerline to be called ("case_name_centerline.vtp")
"""

import numpy as np
import pyvista as pv
from geometry_tools.meshing import Mesher
from geometry_tools import vmtk_wrapper as vmtk
from scipy.spatial import cKDTree as KDTree 
from pathlib import Path
import sys

def make_cl(proj_dir, case_name):
    out_dir = proj_dir
    surf_file = sorted(proj_dir.glob('*cl.vtp'))[0]
    surf=pv.read(surf_file)
    remeshed_file = out_dir/(case_name+'_remeshed.vtp')
    graphed_cl_file = out_dir/(case_name+'_centerline_graph_vmtk.vtp')
    resampled_file =  out_dir/(case_name+'_centerline_resampled.vtp')
    if not remeshed_file.exists():
        surf = vmtk.surface_remeshing(surf, edgelength=0.5)
        surf.save(remeshed_file)
        
    m = Mesher(
            surf,
            include_aneurysms=False
            )
    
    if not graphed_cl_file.exists():
        centerlines, graph = vmtk.network_extractor(m.surf)
        #m.centerlines = vmtk.resample_cl(m.centerlines, length=0.2)
        #m.centerlines = vmtk.centerline_geometry(m.centerlines)
        #m.centerlines.save(out_dir/(case_name+'_centerline.vtp'))
        #m.centerlines = vmtk.centerlines_smooth(m.centerlines, iterations=1, sm_factor=0.1)
        centerlines.save(out_dir/(case_name+'_centerline_graph.vtp'))
        graph.save(out_dir/(case_name+'_graph.vtp'))
        #use vmtk for each segment
        centerline = pv.PolyData()
        for idx in range(1, len(graph.points), 2):
                inlet_id = [idx-1]
                m.inlet_points = graph.points[inlet_id]
                outlet_id = [idx]
                m.outlet_points = graph.points[outlet_id]
                m.generate_centerlines(include_aneurysms=False)
                '''
                surf = surf.compute_normals()
                surf_perturb = surf.copy()
                perturbed_vec = np.einsum(
                    'ij,i->ij', 
                    surf_perturb.point_arrays['Normals'], 
                    np.random.normal(0, 0.005, surf_perturb.n_points)
                    )
                surf_perturb.points = surf_perturb.points + perturbed_vec
                m.centerlines = vmtk.centerlines(surf_perturb, seed_selector = 'pointlist', src_pts = m.inlet_points, target_pts = m.outlet_points)
                '''
                if idx == 1:
                    centerline = m.centerlines
                else:
                    centerline += m.centerlines
        m.centerlines = centerline
        centerline.save(graphed_cl_file)
    else:
        m.centerlines =  pv.read(graphed_cl_file)
        centerline_resampled = vmtk.resample_cl(m.centerlines, length=1.5)
        centerline_resampled.save(resampled_file)
if __name__ == "__main__":
    proj_dir = Path(sys.argv[1])
    case_name = sys.argv[2] 
    make_cl(proj_dir=proj_dir, case_name=case_name)
