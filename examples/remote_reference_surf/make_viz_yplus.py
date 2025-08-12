""" Generate visualizations using BSL tools.
"""

import sys
import os
from pathlib import Path
import re
os.environ['MPLCONFIGDIR'] = '/scratch/s/steinman/ahaleyyy/.config/mpl'
import matplotlib.pyplot as plt 
import imageio
import pyvista as pv
import numpy as np 
import vtk
import math

def images_to_movie(imgs, outpath, fps=30):
    """ Write images to movie.
    Format inferred from outpath extension, 
    see imageio docs.
    """
    writer = imageio.get_writer(outpath, format='FFMPEG',fps=fps)

    for im in imgs:
        writer.append_data(imageio.imread(im))
    writer.close()

def vtk_taubin_smooth(mesh, pass_band=0.1, feature_angle=60.0, iterations=20):
    """ Smooth mesh using Taubin method. """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(mesh) 
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff() 
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return pv.wrap(smoother.GetOutput())

def viz_yplus(ref, array_name, mesh, output_folder, cpos=None, window_size=[768, 768], clim=None):
    out = output_folder
    output_folder = Path(output_folder)
    
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    silhouette = dict(color='black', line_width=3.0, decimate=None)
    
    p=pv.Plotter(off_screen=True, window_size=window_size)
    p.camera_position = cpos
    
    p.add_mesh(mesh,
        cmap='plasma',
        clim=clim,
        smooth_shading=True,
        name='y_plus',
        show_scalar_bar=False,
        silhouette=silhouette,
        scalars=array_name,
        lighting=True,
        specular=0.0,
        diffuse=1.0,
        ambient=1.0,
        )
    p.add_scalar_bar(
        title='y+ maximum',
        n_labels=clim[1],
        width=0.1,
        height=0.2,
        position_x=0.05,
        position_y=0.1,
        vertical=True,
        fmt='%1.f',
        )
    p.show(screenshot=output_folder / ('{}_y_plus_max.png'.format(ref)))
    p.close()

def viz_domains(ref, mesh, output_folder, cpos=None, window_size=[768, 768], clim=None):
    out = output_folder
    output_folder = Path(output_folder)
    
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    silhouette = dict(color='black', line_width=3.0, decimate=None)
    
    p=pv.Plotter(off_screen=True, window_size=window_size)
    p.camera_position = cpos

    p.add_mesh(mesh,
        cmap='plasma',
        smooth_shading=True,
        show_scalar_bar=False,
        name='surface',
        silhouette=silhouette,
        scalars='Ve',
        clim = clim,
        lighting=True,
        specular=0.0,
        diffuse=1.0,
        ambient=1.0,
        )
    p.add_scalar_bar(
        title='$V_e$',
        n_labels=4,
        width=0.1,
        height=0.2,
        position_x=0.05,
        position_y=0.1,
        vertical=True,
        fmt='%1.4f',
        )
    p.show(screenshot=output_folder / ('{}_Ve_surface.png'.format(ref)))
    p.close()

def viz_process(ref, mesh, output_folder, array_name = None, cpos=None, window_size=[768, 2304], clim=None):
    out = output_folder
    output_folder = Path(output_folder)
    
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    silhouette = dict(color='black', line_width=3.0, decimate=None)
    
    p=pv.Plotter(off_screen=True, window_size=window_size)
    p.camera_position = cpos
    fmt='%1.2f'
    cmap='plasma'
    s_name=array_name
    if array_name == 'Ve':
        s_name='$V_e$'
        fmt='%1.4f'
    if array_name=='oldSize':
        s_name='Initial EL'
        fmt='%1.2f'
    if array_name=='Size':
        s_name ='New EL'
        fmt='%1.2f'
    if array_name == 'ref':
        s_name='Refinement Map'
        fmt='%1.2f'
        cmap='plasma_r'
    if array_name == None:
        multi_colors=True
    else:
        multi_colors=False

    p.add_mesh(mesh,
        cmap=cmap,
        smooth_shading=True,
        show_scalar_bar=False,
        name='surface',
        silhouette=silhouette,
        scalars=array_name,
        clim = clim,
        lighting=True,
        specular=0.0,
        diffuse=1.0,
        ambient=1.0,
        multi_colors=multi_colors
        )
    p.add_scalar_bar(
        title=s_name,
        n_labels=7,
        width=0.2,
        height=0.27,
        position_x=0.05,
        position_y=0.03,
        vertical=True,
        fmt=fmt,
        )
    p.show(screenshot=output_folder / ('{}_{}_surface.png'.format(ref,array_name)))
    p.close()


if __name__ == "__main__":
    case = sys.argv[1] #eg. A
    ref = sys.argv[2] #eg. case_028_low
    out_folder = 'DPQ_files/figs'.format(case)
    
    if case == 'A' or case == 'A_uul' or case == 'A_base':
        cpos_yplus = [(0.12803017759126217, 0.010063464259694701, 0.049640455634675146),
 (0.016244555918878363, -0.03161410656734622, -0.004106068266565774),
 (-0.4292163376882493, -0.013398001079371114, 0.9031023358560004)]
        clim = [0,10]
    elif case == 'B' or case == 'B_uul' or case == 'B_base':
        cpos_yplus = [(0.183100267732579, -0.008745433620431063, 0.02737021257654537),
 (0.04408511030886814, -0.022585534405651315, 0.04289240689841815),
 (0.04764643173133199, 0.49466537435711466, 0.8677764602450373)]
        clim = [0,7]
    else:
        cpos_yplus=[(0.05310108024811892, 0.03713962279086022, -0.5691060180848868),
 (0.03922270854722861, -0.03322778846462496, -0.5961091575726692),
 (0.36346066310722874, -0.39538944053650216, 0.8435422554240454)]
        clim = [0,8]
    
    mesh = pv.read('DPQ_files/case_{}/{}_yplus.vtp'.format(case,ref))
    viz_yplus(ref,'y_max', mesh, out_folder, cpos=cpos_yplus, clim=clim)
    '''
    if case == 'A':
        cpos_reg='-376.4742110859654,-21.99532990350684,102.56016377833636,19.39754056930542,-35.10753917694092,-5.915561676025391,0.2658040837345268,0.07175655506646969,0.9613527894977167'
        clim = [0,0.003]
    elif case == 'B':
        cpos_reg='-313.59959986624835,-57.60343722861809,68.04773278206676,48.24593544006348,-31.467658519744873,31.001827239990234,0.06274053306500921,0.41676708294829407,0.9068455348522638'
        clim = [0,0.001]
    else:
        cpos_reg='-247.82827430783206,178.7470347423291,-466.14256977558534,4.279509302599797,-22.407039460313538,-609.5624261084848,0.30466564575397603,-0.26949853368792437,0.9135367450942362'
        clim = [0,0.002]

    cpos_str = cpos_reg.split(',')[:]
    cpos_flt = [float(x) for x in cpos_str]
    it = iter(cpos_flt)
    cpos_reg=list(zip(it,it, it))
    
    mesh_dom = pv.read('case_{}/{}_ref.vtp'.format(case,ref))
    viz_domains(ref, mesh_dom, out_folder, cpos=cpos_reg, clim=clim)
    
    cpos_map=[(68.4717290169933, 352.42942834173185, 170.1222506041669),
 (19.391900000000004, -35.11385, -5.914849999999992),
 (-0.21694130741814782, -0.3808144617738715, 0.8988419298406041)]
    
    mesh_map = pv.read('case_{}/PTSeg028_uul_0p8/PTSeg028_uul_0p8_surf.vtp'.format(case))
    #for array_name in mesh_map.array_names:
    #    viz_process(ref, mesh_map, out_folder, array_name = array_name, cpos=cpos_map)
    viz_process(ref, mesh_map, out_folder, array_name = 'ref', cpos=cpos_map)
    viz_process(ref, mesh_map, out_folder, array_name = 'Ve', cpos=cpos_map)
    viz_process(ref, mesh_map, out_folder, array_name = 'Size', cpos=cpos_map)
    
    cpos_dom=[(0.0684717290169933, 0.35242942834173185, 0.1701222506041669),
 (0.019391900000000004, -0.03511385, -0.005914849999999992),
 (-0.21694130741814782, -0.3808144617738715, 0.8988419298406041)]

    mesh_heat = pv.read('case_{}/case_028_ultraultralow_domains.vtm'.format(case))
    viz_process(ref, mesh_heat, out_folder, array_name = 'T', cpos=cpos_dom)
    viz_process(ref, mesh_heat, out_folder, array_name = None, cpos=cpos_dom)
    '''