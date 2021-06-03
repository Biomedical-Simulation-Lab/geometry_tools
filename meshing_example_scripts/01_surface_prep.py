""" Example: prepare a surface for volume meshing.
"""

from pathlib import Path 
import pyvista as pv 
from meshing_tools.mesh_generation import Mesher

if __name__ == "__main__":
    cwd = Path(__file__).parent.absolute()
    surf_file = Path(cwd / 'data' / 'Pair10a.stl')

    dest = surf_file.parents[0]

    surf = pv.read(surf_file)

    m = Mesher(surf)
    m.clip_boundaries()
    m.set_inlets_outlets()
    m.select_refinement_regions()
    m.generate_centerlines()
    m.surface_preparation()
    m.update_inlets_outlets()
    m.generate_centerlines()
    m.refine_picked_regions()

    m.surf.save(dest / (surf_file.stem + '_processed.vtp'))
    m.save_inlet_outlet_points(dest / (surf_file.stem + '_endpoints.h5'))
    m.centerlines.save(dest / (surf_file.stem + '_centerlines.vtp'))