"""
This script contains a method for obtaining a surface of your geometry or a number of surface 
mesh spheres that can later be used to compute spectrograms within the volume contained by the surface(s)

Just comment out/uncomment what you want - surface or spheres

Call this script using:
make_spectro_points.py proj_dir step radius

Where 
-proj_dir is the directory where the surface mesh is stored,
-step is how many centerline points to skip to make the spheres (eg. 5),
-radius is the radius of the spheres you want (eg. 10) this is in mm
"""

import numpy as np
import pyvista as pv
import geometry_tools.vmtk_wrapper as vmtk
import geometry_tools.common as cc
from scipy.spatial import cKDTree as KDTree
from pathlib import Path
import sys

def make_spectro(proj_dir, step, rad):
    out_dir = proj_dir
    surf_file = sorted(proj_dir.glob('*_cl_mapped.vtp'))[0]
    cl_file = sorted(proj_dir.glob('*_centerline.vtp'))[0]
    point_file = surf_file.stem + '_spectrospheres.vtm'
    surf = pv.read(surf_file)
    cent = pv.read(proj_dir.parent / cl_file)

    select = cc.RefinementSelection_OLD(surf, title = 'Select region for spectrograms')
    newsurf = select.temprefsurf
    '''
    cent_selected=cent.select_enclosed_points(newsurf, tolerance=0.01)
    cent.point_data['spectro_pt']=cent_selected.point_data['SelectedPoints']
    new_cl=pv.PolyData()
    new_cl.points = cent.points[cent.point_data['spectro_pt']==1]
    new_cl.point_data['radius']=cent.point_data['MaximumInscribedSphereRadius'][cent.point_arrays['spectro_pt']==1]*3/4

    newer_cl=pv.PolyData()
    newer_cl.points = new_cl.points[::int(step)].copy()
    newer_cl.point_data['radius']=new_cl.point_data['radius'][::int(step)].copy()

    spheres = pv.MultiBlock()

    p = pv.Plotter()
    for idx,pt in enumerate(newer_cl.points):
        sph =  pv.Sphere(radius = float(rad), center = pt) #newer_cl.point_data['radius'][idx]
        p.add_mesh(sph)
        spheres.append(sph)

    p.add_mesh(newsurf, color="gray", opacity=0.5)
    #p.add_points(newer_cl.points, color='r', point_size=10)
    p.show()

    spheres.save(out_dir/point_file)
    '''
    newsurf.save(out_dir/('spectro_sigmoid.vtp'))

if __name__ == "__main__":
    proj_dir = Path(sys.argv[1]) 
    step = sys.argv[2]
    radius = sys.argv[3]
    make_spectro(proj_dir=proj_dir, step=step, rad = radius)
