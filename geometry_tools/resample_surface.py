import pyvista as pv 
import numpy as np 
from pathlib import Path
import pymeshfix
from geometry_tools import common as cc

class Resampler():
    """ Fix a messy mesh using resampling and pymeshfix
    """
    def __init__(self, surf, resample_spacing=0.05):
        self.surf = surf 
        self.resample_spacing = resample_spacing

    def remove_junk_points(self, thresh=1e4):
        junk_points = np.unique(np.where(np.abs(self.surf.points) > thresh)[0])
        self.surf, _ = self.surf.remove_points(junk_points)
        self.surf = self.surf.extract_largest()

    def fix(self):
        # Decimate the surface
        surf = self.surf 

        if surf.n_points > 75000:
            decimate_ratio = 1 - 75000 / surf.n_points
        else:
            decimate_ratio = 0.05
        surf = surf.decimate(decimate_ratio)
        surf = surf.clean(tolerance=1e-4)

        # Resample the surface to a grid, recontour
        self.grid = cc.vtk_generate_img_stencil(surf.fill_holes(15.), spacing=self.resample_spacing)

        surf_r = self.grid.contour([0.5])
        surf_smooth = cc.vtk_taubin_smooth(surf_r, pass_band=0.03, iterations=100)

        # Decimate again
        if surf_smooth.n_points > 75000:
            decimate_ratio = 1 - 75000 / surf_smooth.n_points
        else:
            decimate_ratio = 0.05
        surf_decimate = surf_smooth.decimate(decimate_ratio)
        
        # Use pymeshfix to get rid of junk
        surf = surf_decimate.triangulate()
        meshfix = pymeshfix.MeshFix(surf)
        meshfix.repair(verbose=True)
        surf = meshfix.mesh
        surf = surf.clean()
        self.surf = surf
