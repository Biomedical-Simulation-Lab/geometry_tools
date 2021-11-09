""" Fix a meshes using resampling and pymeshfix.
""" 

from pathlib import Path 
import pyvista as pv 
import numpy as np
from geometry_tools.resample_surface import Resampler
import sys

if __name__ == "__main__":
    surf_file = Path(sys.argv[1])
    
    if len(sys.argv) > 2:
        out_file = Path(sys.argv[2])
    else:
        out_file = Path(surf_file.parents[0] / (surf_file.stem + '_rs.vtp'))

    if not out_file.exists():
        surf = pv.read(surf_file)
        r = Resampler(surf, resample_spacing=0.05)
        r.remove_junk_points()
        r.fix()
        r.surf.save(out_file)
    else:
        print('Output file exists.')