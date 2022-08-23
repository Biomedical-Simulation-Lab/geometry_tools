""" Generate volume mesh and relevant submission files.
"""

from pathlib import Path 
import pyvista as pv 
from geometry_tools.meshing import Mesher
from geometry_tools.make_submission_file import SubmissionTemplate
import sys
import time
from datetime import timedelta

def volume_meshing(proj_dir, surf_type, multi_inlets):
    
    if surf_type == 'pt':
        anubool=False
    else:
        anubool = True
        
    surf_file = sorted(proj_dir.glob('*_pr.vtp'))[0]

    point_file = proj_dir / (surf_file.stem + '_endpoints.vtm') 
    assert point_file.exists()

    mesh_out_dir = proj_dir / 'mesh' 
    data_out_dir = proj_dir / 'data'  
    submission_out_dir = proj_dir / 'submissions' 

    for f in [mesh_out_dir, data_out_dir, submission_out_dir]:
        if not f.exists():
            f.mkdir(parents=True, exist_ok=True)

    vtufile = mesh_out_dir / (surf_file.stem + '.vtu')
    meshfile = data_out_dir / (surf_file.stem + '.h5')
    infofile = data_out_dir / (surf_file.stem + '.info')
    xmlgzfile = data_out_dir / (surf_file.stem + '.xml.gz')

    start = time.time()
    print('\n' + surf_file.stem)

    if not xmlgzfile.exists():
        # try:
        surf = pv.read(surf_file)

        points = pv.read(point_file)
        in_points = points['inlets'].points
        out_points = points['outlets'].points
        if surf_type == 'a':
            an_points = points['aneurysms'].points
        else:
            an_points = None

        m = Mesher(
            surf,
            inlet_points=in_points, 
            outlet_points=out_points,
            aneurysm_points=an_points,
            include_aneurysms=anubool
            )
        m.update_inlets_outlets()

        if not vtufile.exists():
            m.generate_volume_mesh()
            m.mesh.save(vtufile)
        else:
            m.mesh = pv.read(vtufile)

        m.update_inlets_outlets()
        if surf_type == 'a':
            m.generate_centerlines()
        m.generate_centerlines(include_aneurysms=False)
        #WARNING: does not work with multiple inlets
        if multi_inlets == 'single':
            m.generate_flow_rates()
        m.generate_h5_file(meshfile)
        #m.generate_flow_rates_legacy() #why call?
        m.update_inlets_outlets()
        m.generate_info_file(infofile, multi_inlets, inlet_vel=0.27, waveform='FC_MCA_10')  
        m.generate_xml_gz_file(xmlgzfile)

        # Create submission file
        s = SubmissionTemplate(meshfile.stem)
        s.save_script(submission_out_dir)

        print('\n Case done', timedelta(seconds=time.time() - start))

    else:
        print('Output file exists.')



if __name__ == "__main__":
    proj_dir = Path(sys.argv[1]) 
    surf_type = sys.argv[2]
    if len(sys.argv) > 3:
        multi_inlets = sys.argv[3]
    else:
        multi_inlets = 'single'
    volume_meshing(proj_dir=proj_dir, surf_type=surf_type, multi_inlets=multi_inlets)
