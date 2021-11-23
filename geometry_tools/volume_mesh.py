import pyvista as pv 
import numpy as np
from geometry_tools import vmtk_wrapper as vmtk
from geometry_tools.surface import Surfer

from pathlib import Path 
from geometry_tools import common as cc 
from scipy.spatial import cKDTree as KDTree 
import h5py 

class VolumeMesh(Surfer):
    """ Basic analysis tools based on VMTK. 

    Attributes:
        mesh : volumetric UnstructuredGrid or None
        inlet_points : array of inlet points
        outlet_points : array of outlet points
        centerlines : PolyData centerlines of surf
        centerlines_branched : PolyData centerlines of surf with branch IDs
    """

    def __init__(self, surf=None, mesh=None, inlet_points=None, outlet_points=None, aneurysm_points=None):
        self.mesh = mesh

        entity_ids = np.unique(self.mesh.cell_arrays['CellEntityIds'])
        pieces = [self.mesh.extract_cells(self.mesh['CellEntityIds']==v) for v in entity_ids]

        self.pieces = dict(zip(entity_ids, pieces))
        surf = self.pieces[1]
        assert np.all(surf.cells.reshape(-1,4)[:,0] == 3), 'Surface is not triangulated'
        self.surf = pv.PolyData(surf.points, surf.cells)
        self.surf = self.surf.clean()
        self.surf = self.surf.compute_normals(auto_orient_normals=True)

        self.inner_mesh = self.pieces[0]

    def create_sac_mask(self, hole_size=20.0):
        """ Create sac mask array in mesh and surf.
        """
        self.extract_sacs()

        for sac_id in self.sacs.keys():
            sac = self.sacs[sac_id].fill_holes(hole_size)
            sac = sac.compute_normals(auto_orient_normals=True)
            sac_inflate = sac.copy()
            sac_inflate.points = sac.points + 0.1*sac.point_arrays['Normals']
        
            mask_arr_name = 'sac_mask_{}'.format(sac_id)
            self.mesh = self.mesh.select_enclosed_points(sac_inflate, check_surface=False)
            self.mesh.point_arrays[mask_arr_name] = self.mesh.point_arrays['SelectedPoints'].copy().astype(bool)
            self.mesh, _ = cc.smooth_mesh_data_local(self.mesh, array=mask_arr_name)

            self.surf.point_arrays[mask_arr_name] = np.zeros(self.surf.n_points, dtype=bool)
            self.surf.point_arrays[mask_arr_name][self.surf.point_arrays['GroupIds'] == sac_id] = True

        return self