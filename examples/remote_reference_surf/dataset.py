import sys
import os
from pathlib import Path
import numpy as np
#import scipy.linalg as la
import math
import h5py
import pyvista as pv
import vtk

class Dataset():
    """ Load BSL-specific data and common ops. 
    """
    def __init__(self, folder, mesh_folder = None, file_glob_key=None, file_stride=1, case_name=None, wss_files=False):
        self.folder = Path(folder)
        if case_name is not None:
            self.case_name=case_name

        if file_glob_key is None:
            file_glob_key = '*_curcyc_*up.h5'

        mesh_glob_key = '*h5'
        if mesh_folder is None:
            #get from data path instead of results path
            self.folder_data=self.folder.parents[0] / ('data')
            print(self.folder_data)
            self.mesh_file = sorted(self.folder_data.glob(mesh_glob_key), key=lambda x: len(x.stem))[0]
            self.ts = '_ts='
            self.tssplit = '_'
            self.different_folders=False
        else:
            self.mesh_file = sorted(mesh_folder.glob(mesh_glob_key), key=lambda x: len(x.stem))[0]   
            self.ts='_tstep=' 
            self.tssplit='.'
            self.different_folders=True

        self.up_files = sorted(self.folder.glob(file_glob_key), key=self._get_ts)[::int(file_stride)]
        self.tsteps = len(sorted(self.folder.glob(file_glob_key), key=self._get_ts)[::int(file_stride)])
        self.times = sorted(self.folder.glob(file_glob_key), key=self._get_time)[::int(file_stride)]
        if wss_files:
            wss_folder = folder / ('wss_files')
            self.wss_files = sorted(self.folder.glob(file_glob_key), key=self._get_ts)[::int(file_stride)]


    def __call__(self, idx=0, array='u', file=None):
        """ Return velocity in u_file. """
        if array in ['u', 'p']:
            if file==None:
                h5_file = self.up_files[idx]
            else:
                h5_file=file
            with h5py.File(h5_file, 'r') as hf:
                if self.different_folders:
                    val = np.array(hf[array])
                else:
                    val = np.array(hf['Solution'][array])
        else:
            if array in ['wss', 'qcriterion']:
                if file==None:
                    h5_file = self.wss_files[idx]
                else:
                    h5_file=file
                with h5py.File(h5_file, 'r') as hf:
                    val = np.array(hf['Computed'][array])
            else:
                h5_file = file
                with h5py.File(h5_file, 'r') as hf:
                    val = np.array(hf[array])
        return val

    def _get_ts(self, h5_file):
        """ Given a simulation h5_file, get ts. """
        return int(h5_file.stem.split(self.ts)[1].split(self.tssplit)[0])
    
    def _get_ts_swirl(self, h5_file):
        """ Given a different h5_file, get ts. """
        return int(h5_file.stem.split('_')[1].split('.')[0])
        
    def _get_time(self, h5_file):
        """ Given a simulation h5_file, get time. """
        return float(h5_file.stem.split('_t=')[1].split('_')[0]) / 1000.0
        
    def check_cells(self):
        #print(self.mesh.n_cells)
        cellsize=self.mesh.compute_cell_sizes().cell_data['Volume']
        neg=np.flatnonzero(cellsize<0)
        #mark the boundary cells:
        self.mesh.cell_data['boundary']=np.zeros(self.mesh.n_cells)
        self.mesh.cell_data['boundary'][neg]=1
        #print(np.sum(self.mesh.cell_data['boundary']))
        self.mesh=self.mesh.cell_data_to_point_data()
        #now swap the last two indices of those cells:
        n3=5*neg+2
        n4=5*neg+3
        cells = self.mesh.cells.copy()
        cells_n3=cells[n3].copy()
        cells_n4=cells[n4].copy()
        cells[n3]=cells_n4
        cells[n4]=cells_n3
        self.mesh.cells = cells
        cellsize=self.mesh.compute_cell_sizes().cell_data['Volume']
        new_neg=np.flatnonzero(cellsize<0)
        if len(new_neg)>0:
            print("check_cells didn't work!")
            sys.exit()
        return self
    
    def assemble_mesh(self):
        """ Create UnstructuredGrid from h5 mesh file. """
        assert self.mesh_file.exists(), 'mesh_file does not exist.'
        #self.mesh_file='../mesh_rez/cases/case_C/case_106_ultraultralow/data/PTSeg106_ultraultralow.h5'
        
        with h5py.File(self.mesh_file, 'r') as hf:
            points = np.array(hf['Mesh']['coordinates'])*(10**-3)
            cells = np.array(hf['Mesh']['topology'])

            celltypes = np.empty(cells.shape[0], dtype=np.uint8)
            celltypes[:] = vtk.VTK_TETRA
            cell_type = np.ones((cells.shape[0], 1), dtype=int) * 4
            cells = np.concatenate([cell_type, cells], axis = 1)
            self.mesh = pv.UnstructuredGrid(cells.ravel(), celltypes, points)
            self=self.check_cells()

            w_points = np.array(hf['Mesh']['Wall']['coordinates'])*(10**-3)
            w_cells = np.array(hf['Mesh']['Wall']['topology'])

            w_cell_type = np.ones((w_cells.shape[0], 1), dtype=int) * 3
            w_cells = np.concatenate([w_cell_type, w_cells], axis = 1)
            self.surf = pv.PolyData(w_points, w_cells)
    
        # self.assemble_surface()
        return self