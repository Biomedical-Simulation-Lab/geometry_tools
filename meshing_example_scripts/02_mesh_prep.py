""" Example: generate volume mesh and relevant submission files.
"""

from pathlib import Path 
import pyvista as pv 
from meshing_tools.mesh_generation import Mesher
from meshing_tools.make_submission_file import SubmissionTemplate
import h5py

if __name__ == "__main__":
    cwd = Path(__file__).parent.absolute()
    surf_file = Path(cwd / 'data' / 'Pair10a_processed.vtp')
    in_out_points = Path(cwd / 'data' / 'Pair10a_endpoints.h5')
    
    dest = surf_file.parents[0]

    vtufile = dest / (surf_file.stem + '.vtu')
    meshfile = dest / (surf_file.stem + '.h5')
    infofile = dest / (surf_file.stem + '.info')
    xmlgzfile = dest / (surf_file.stem + '.xml.gz')

    surf = pv.read(surf_file)
    points = h5py.File(in_out_points, 'r')

    m = Mesher(surf, inlet_points=points['inlets'], outlet_points=points['outlets'])
    m.generate_volume_mesh()
    m.update_inlets_outlets()

    m.mesh.save(vtufile)
    m.generate_centerlines()
    m.generate_flow_rates()
    m.generate_h5_file(meshfile)
    m.update_inlets_outlets()
    m.generate_info_file(infofile)  
    m.generate_xml_gz_file(xmlgzfile)

    # Create submission file
    s = SubmissionTemplate(meshfile.stem)
    s.save_script(dest)

