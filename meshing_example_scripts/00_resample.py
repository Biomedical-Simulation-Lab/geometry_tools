""" Example: fix a messy mesh using resampling and pymeshfix.
""" 

from pathlib import Path 
import pyvista as pv 
from meshing_tools.resample_surface import Resampler

if __name__ == "__main__":
    cwd = Path(__file__).parent.absolute()
    surf_file = Path(cwd / 'data' / 'Pair10a.stl')
    dest = surf_file.parents[0]

    surf = pv.read(surf_file)
    r = Resampler(surf)
    r.fix()
    r.surf.save(dest / (surf_file.stem + '_fixed.vtp'))