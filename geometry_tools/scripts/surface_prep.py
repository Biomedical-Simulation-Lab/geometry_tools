#!/usr/bin/env python3
""" Clip boundaries, mark inlets, outlets, and aneurysm locations.
"""

from pathlib import Path 
import pyvista as pv 
from geometry_tools.meshing import Mesher
from geometry_tools import common as cc
import time
from datetime import timedelta
import sys 
import numpy as np

def surface_prep(surf_file, proj_dir, surf_type):
    """ Basic surface prep.
    """
    surf_file = Path(surf_file)
    proj_dir = Path(proj_dir)
    if not proj_dir.exists():
        proj_dir.mkdir()
    
    '''
    surf_output_dir = proj_dir / '01_clipped' 
    points_output_dir = proj_dir / '01_points' 
    neckpoints_output_dir = proj_dir / '01_neckpoints' 

    for f in [surf_output_dir, points_output_dir, neckpoints_output_dir]:
        if not f.exists():
            f.mkdir(parents=True)
    '''
    # Output files
    surf_file_out = proj_dir / (surf_file.stem + '_cl.vtp')
    if surf_type=='a':
        neck_file_out = proj_dir / (surf_file.stem + '_cl_neckpoints.vtm')
    #else:
        #we may want to choose some points to identify some important pt features (eg torcula)
    points_file_out = proj_dir / (surf_file.stem + '_cl_endpoints.vtm') #Multiblock object

    if not surf_file_out.exists():
        case_start = time.time()

        surf = pv.read(surf_file)
        surf = surf.compute_normals(auto_orient_normals=False)
        if surf_type=='a': 
            m = Mesher(surf, include_aneurysms=True)
        else:
            m = Mesher(surf) 

        # Delete any sharp edges or small branches
        # if flag_inspect == True:
        # s = cc.SelectGeodesic(m.surf, scalars='Delete', )
        # s.interact(title='Isolate region to delete and fill.')
        # m.surf = s.mesh
        # m.surf = m.surf.fill_holes(10.0)
 
        # Uses m.clip_boundaries uses cc.ClickDragDelete to delete boundaries.
        flag_inspect = m.clip_boundaries()

        m.set_inlets_outlets()
        if surf_type=='a':
            m.pick_aneurysm()
            m.copy_structure() #this makes the surface mesh (m.surf) into a pv.PolyData object    
        
        m.surf.save(surf_file_out) #can only save in vtk, ply, or stl format (not vtp)
           
        if surf_type=='a':
            # Select aneurysms
            s = cc.SelectGeodesic(m.surf)
            s.interact(title='Isolate aneurysms.')
            s.save_stored_points(neck_file_out)
            m.surf = s.mesh 
            

        if surf_type=='a':
            anubool=True
        else:
            anubool=False   
        m.generate_centerlines(include_aneurysms=anubool)#probably need something here for torcula
        m.save_inlet_outlet_points(points_file_out, include_aneurysms=anubool)

        time_spent = time.time() - case_start
        print('Case done', timedelta(seconds=time_spent))

    else:
        print('Output file exists.')


if __name__ == "__main__":
    surf_file = sys.argv[1]
    proj_dir = sys.argv[2]
    if len(sys.argv) > 3:
        surf_type = sys.argv[3] #options: a or pt
    else:
        surf_type = 'a' #default is an aneurysm surface file
    surface_prep(surf_file, proj_dir, surf_type)
