"""
This file contains a method to produce all the outputs you would need to run a case using Mehdi's framework
for a PT-specific case. Requires a surface mesh with a 'Size' array and also with a 'Ve' array (unsmoothed)

Call this script using:
make_mesh.py proj_dir proj_name multi_inlets ref

Where 
-proj_dir is the directory where the surface mesh is stored,
-proj_name is the name you want all of your files to start with,
-multi_inlets indicates whether there are more than one inlet (options are 'Multi' and 'single'),
- ref is a number between 0-1 that you want to multiply your highest Ve region with

"""

import numpy as np
import pyvista as pv
from geometry_tools.meshing import Mesher
from vmtk import vmtkscripts
import geometry_tools.vmtk_wrapper as vmtk
import geometry_tools.common as cc
from geometry_tools.make_submission_file import SubmissionTemplate
from scipy.interpolate import interp1d
from pathlib import Path
import sys

def create_ref_array(surf, min_ref):
    min_Ve=min(surf.point_data['logVe'])
    max_Ve=max(surf.point_data['logVe'])
    print('Sizing array...')
    
    interp = interp1d([min_Ve, max_Ve], [1, min_ref], #smallest Ve will keep its element edge length, largest will be reduced in size by a factor of ref
        kind='linear',
        bounds_error=False,
        fill_value=(1, min_ref),
        )
    surf.point_data['ref'] = interp(surf.point_data['logVe'])
    print('Smoothing ref array...')
    surf, neighbour_pts = cc.smooth_mesh_data_local_alt(surf, array='ref', iterations = 10)
    surf.point_data['oldSize']=surf.point_data['Size']
    surf.point_data['Size']=surf.point_data['Size']*surf.point_data['ref']
    print('Smoothing Size array....')
    surf, _ = cc.smooth_mesh_data_local_alt(surf, array='Size', neighbour_pt_ids = neighbour_pts, iterations = 10)
    #surf, _ = cc.smooth_mesh_data_local(surf, array='Size', func=np.mean, iterations=5) #more aggressive
    
    return surf

def make_ref_mesh(proj_dir, proj_name, multi_inlets,ref, mapped_file=None):
    if not proj_dir.exists():
        proj_dir.mkdir()
    main_dir = proj_dir.parent
    if mapped_file==None:
        mapped_file = sorted(main_dir.glob('*_mappedsys.vtp'))[0] #('*_mappedsys.vtp'))[0]
    #else:
    #    mapped_file = sorted(main_dir.glob('*_{}_mapped.vtp'.format(mapped_file)))[0]
    ref_file = sorted(main_dir.glob('*_ref.vtp'))[0]
    
    ref_out_file = proj_dir / (proj_name + '_surf.vtp')
    mesh_out_dir = proj_dir / 'mesh' 
    data_out_dir = proj_dir / 'data' 
    vtufile = mesh_out_dir / (proj_name + '.vtu')
    meshfile = data_out_dir / (proj_name + '.h5')
    infofile = data_out_dir / (proj_name + '.info')
    fcoeffsfile = data_out_dir / ('FC_VENOUS')
    xmlgzfile = data_out_dir / (proj_name + '.xml.gz')
    for f in [mesh_out_dir, data_out_dir]:
        if not f.exists():
            f.mkdir(parents=True, exist_ok=True)
    m = Mesher(
            include_aneurysms=False
            )
    #m.surf = pv.read('PTSeg028_frommesh_0p{}.vtp'.format(min_ref.split('.')[-1]))
    
    surf_size = pv.read(mapped_file)

    #Get all of the things we need together on one surface
    if 'Size' in surf_size.point_data:
        surf_ref = pv.read(ref_file)
        #project the size surface onto the refinement surface:
        projection = vmtkscripts.vmtkSurfaceProjection()
        projection.Surface = surf_size
        projection.ReferenceSurface = surf_ref
        projection.Execute()
        #save the projection
        surf = pv.wrap(projection.Surface) #should now have both ref and Size on one surface
        #Get the new Size array
        surf = create_ref_array(surf, min_ref=float(ref))
        surf.clean()
        surf_rm=vmtk.surface_remeshing(surf, element_size_mode='edgelengtharray', edgearray='Size', iterations=3)
        projection = vmtkscripts.vmtkSurfaceProjection()
        projection.Surface = surf_rm
        projection.ReferenceSurface = surf
        projection.Execute()
        surf = pv.wrap(projection.Surface)

        m.surf=surf
    else:
        m.surf=surf_size
    
    m.surf.save(ref_out_file) #save refinement surface with new size array
    
    #Remesh the surface
    surf_og=m.surf.copy()
    m.surf=vmtk.surface_remeshing(m.surf, element_size_mode='edgelengtharray', edgearray='Size', iterations=10)
    projection = vmtkscripts.vmtkSurfaceProjection()
    projection.Surface = m.surf
    projection.ReferenceSurface = surf_og
    projection.Execute()
    m.surf = pv.wrap(projection.Surface)

    m.surf.save(ref_out_file) 
    #m.surf=surf_size
    m.set_inlets_outlets()
    #send surface to vmtk
    if not vtufile.exists():
        m.generate_volume_mesh(True)
        #print(m.mesh.n_cells)
        m.mesh.save(vtufile)
    else:
        m.mesh = pv.read(vtufile)
    #print(m.surf)
    m.centerlines, _ = vmtk.network_extractor(m.surf)
    m.centerlines = vmtk.centerline_geometry(m.centerlines)

    m.update_inlets_outlets()
    if not meshfile.exists():
        m.generate_h5_file(meshfile)
    if multi_inlets == 'single':
        m.outlet_flow_divisions = {} #not sure why generate flow rates not working. Prob something not done
    #NOTE: WILL NEED TO CORRECT THESE FLOWRATES
    m.generate_info_file(infofile, fcoeffsfile, multi_inlets, inlet_vel=False, inlet_flowrates=[5.578888888888,0.1859629629,0.1859629629,0.1859629629], waveform='FC_VENOUS')  
    if not xmlgzfile.exists():
        m.generate_xml_gz_file(xmlgzfile)

    # Create submission file
    min_EL = np.min(m.surf.point_data['Size'])
    max_vel = 3*np.max(m.surf.point_data['mean_velocity'])
    tstep_per_cycle = int(60*np.round((max_vel*915/min_EL)/60)) #round to nearest multiple of 60
    s = SubmissionTemplate(proj_name, timesteps_per_cycle=tstep_per_cycle, save_frequency=1)
    s.save_script(proj_dir)
    

if __name__ == "__main__":
    proj_dir0 = sys.argv[1] 
    min_ref=sys.argv[4]
    dec=min_ref.split('.')[-1]
    proj_name=sys.argv[2]+'_'+'0p'+dec
    multi_inlets = sys.argv[3]
    if len(sys.argv)>5:
        mapped_file = sys.argv[5] 
    else:
        mapped_file = None
    proj_dir = Path(proj_dir0 + '_0p' + dec)
    make_ref_mesh(proj_dir=proj_dir, proj_name=proj_name, multi_inlets=multi_inlets, ref=min_ref, mapped_file=mapped_file)
