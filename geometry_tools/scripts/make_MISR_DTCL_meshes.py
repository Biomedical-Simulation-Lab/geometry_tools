"""
This script contains a method to generate DTCL and MISR volume meshes in vtu format, and you can uncomment
some things if you want to then run the cases.

Call this script using:
make_MISR_DTCL_meshes.py proj_dir proj_name min_el max_el multi_inlets ref

Where 
-proj_dir is the directory where the surface mesh is stored,
-proj_name is the name you want all of your files to start with,
-min_el and max_el are your min and max edge lengths that you want,
-multi_inlets indicates whether there are more than one inlet (options are 'Multi' and 'single'),
-ref indicates if you want a refinement patch, defined as a mask in map_DTCL_MISR.py, and is a float 
indicating the level of refinement (eg. 0.8) you want
"""

import numpy as np
import pyvista as pv
from geometry_tools.meshing import Mesher
import geometry_tools.vmtk_wrapper as vmtk
import geometry_tools.common as cc
from geometry_tools.make_submission_file import SubmissionTemplate
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree as KDTree 
from pathlib import Path
import sys

def make_mesh(proj_dir, proj_name, min_el, max_el, multi_inlets,ref):
    if not proj_dir.exists():
        proj_dir.mkdir()
    main_dir = proj_dir.parent
    mapped_file = proj_dir/(proj_name+'_mapped.vtp')
    #mesh_out_dir = proj_dir / 'mesh' 
    #data_out_dir = proj_dir / 'data' 
    vtufile1 = proj_dir/(proj_name + '_DTCL.vtu')
    vtufile2 = proj_dir/(proj_name + '_MISR.vtu')
    #meshfile = data_out_dir / (proj_name + '.h5')
    #infofile = data_out_dir / (proj_name + '.info')
    #fcoeffsfile = data_out_dir / ('FC_VENOUS')
    #xmlgzfile = data_out_dir / (proj_name + '.xml.gz')
    #for f in [mesh_out_dir, data_out_dir]:
    #    if not f.exists():
    #        f.mkdir(parents=True, exist_ok=True)

    surf=pv.read(mapped_file)
    m = Mesher(
            surf,
            include_aneurysms=False
            )
    m.centerlines = pv.read(main_dir/(proj_name.split('_')[0] +'_centerline.vtp'))
    if not vtufile1.exists():
        m.surf, m.centerlines = vmtk.distance_to_centerlines(m.surf, m.centerlines)
        dtcl_max=np.max(m.surf.point_data['DistanceToCenterlinesArray'])
        dtcl_min =np.min(m.surf.point_data['DistanceToCenterlinesArray'])
        m.surf.clean()
        #check for refinement points
        if ref != 'False':
            m.surf.point_data['RefinementPoints']=m.surf.point_data['ref']
        m.surf = cc.create_edge_size_array(m.surf, fix_centerline='reg', min_edge_size=min_el, max_edge_size=max_el, misr_min=dtcl_min, misr_max=dtcl_max)
        m.surf.clean()

        #send surface to vmtk for distance to centerline meshing
        m.generate_volume_mesh()
        m.mesh.save(vtufile1)

    m.surf=surf
    m.surf, m.centerlines = vmtk.distance_to_centerlines(m.surf, m.centerlines)
    #check for refinement points
    if ref != 'False':
        m.surf.point_data['RefinementPoints']=m.surf.point_data['ref']
    m.surf.point_data['centerline_map']=np.array(surf.n_points)
    m.surf.point_data['misr']=np.array(surf.n_points)
    tree = KDTree(m.centerlines.points)
    _, idx_c = tree.query(surf.points)
    #this assigns the closest centerline point id to each point on the surf
    m.surf.point_data['centerline_map']=idx_c 
    #this assigns the misr associated with the centerline to each point
    tree2 = KDTree(m.surf.points)
    dist, _ = tree2.query(m.centerlines.points) #closest dist to centerline point
    m.centerlines.point_data['MaximumInscribedSphereRadius']=dist
    m.surf.point_data['misr']=m.centerlines.point_data['MaximumInscribedSphereRadius'][idx_c] 
    #fix around the ends
    _, idx_d = tree.query(m.surf.points[m.surf.point_data['misr']==0], k=10)
    max_neigh_misr = np.amax(m.centerlines.point_data['MaximumInscribedSphereRadius'][idx_d], axis=1)
    m.surf.point_data['misr'][m.surf.point_data['misr']==0]=max_neigh_misr
    
    misr_max = np.max(m.surf.point_data['misr'])
    misr_min = np.min(m.surf.point_data['misr'])
    m.surf = cc.create_edge_size_array(m.surf, fix_centerline='fix', min_edge_size=min_el, max_edge_size=max_el, misr_min=misr_min, misr_max=misr_max)
    m.surf.clean()

    #send surface to vmtk for MISR meshing
    #if not vtufile2.exists():
    #    m.generate_volume_mesh()
    #    m.mesh.save(vtufile2)
    m.generate_volume_mesh()
    m.mesh.save(vtufile2)

    '''
    m.set_inlets_outlets()
    m.centerlines, _ = vmtk.network_extractor(m.surf)
    m.centerlines = vmtk.centerline_geometry(m.centerlines)

    m.update_inlets_outlets()
    if not meshfile.exists():
        m.generate_h5_file(meshfile)
    if multi_inlets == 'single':
        m.outlet_flow_divisions = {} #not sure why generate flow rates not working. Prob something not done
    m.generate_info_file(infofile, fcoeffsfile, multi_inlets, inlet_vel=False, inlet_flowrates=[5.578888889, 2.034444444, 0.63534717171], waveform='FC_VENOUS')  
    if not xmlgzfile.exists():
        m.generate_xml_gz_file(xmlgzfile)

    # Create submission file
    min_EL = np.min(m.surf.point_data['Size'])
    max_vel = 3*np.max(m.surf.point_data['mean_velocity'])
    tstep_per_cycle = int(60*np.round((max_vel*915/min_EL)/60)) #round to nearest multiple of 60
    s = SubmissionTemplate(proj_name, timesteps_per_cycle=tstep_per_cycle, save_frequency=1)
    s.save_script(proj_dir)
    '''


if __name__ == "__main__":
    proj_dir = Path(sys.argv[1]) 
    proj_name=sys.argv[2]
    min_el=sys.argv[3]
    max_el=sys.argv[4]
    multi_inlets = sys.argv[5]
    if len(sys.argv)>6:
        ref=sys.argv[6]
    else: 
        ref='False'
    make_mesh(proj_dir=proj_dir, proj_name=proj_name, min_el=min_el, max_el=max_el, multi_inlets=multi_inlets, ref=ref)
