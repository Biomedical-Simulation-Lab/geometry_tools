""" Clip boundaries, mark inlets, outlets, and aneurysm locations.
"""

from configparser import NoOptionError
import sys
from pathlib import Path 
import pyvista as pv 
from geometry_tools.meshing import Mesher
from geometry_tools import common as cc
import geometry_tools.vmtk_wrapper as vmtk
import numpy as np

import time
from datetime import timedelta
from tubeclipper import TubeClipper

def surface_process(proj_dir, proc_dir, surf_type, multi_inlets, ND, min_EL, max_EL, ref, fix_centerline, endpoints_pv=None):
    proj_dir = Path(proj_dir)
    if not proj_dir.exists():
        proj_dir.mkdir()

    proc_dir = Path(proc_dir) 

    surf_file = sorted(proc_dir.glob('*.vtp'))[0]
    #print(proj_dir)
    point_file = proc_dir / (surf_file.stem + '_endpoints.vtm')
    #print(point_file)
    assert point_file.exists()
    
    if surf_type=='a':    
        neckpoints_file = proc_dir / (surf_file.stem + '_neckpoints.vtm')    
        assert neckpoints_file.exists()
        anubool=True

    surf_output_file = proj_dir / (surf_file.stem + '_pr.vtp')
    points_output_file = proj_dir / (surf_file.stem + '_pr_endpoints.vtm')
    #endpoints_output_file = proj_dir / (surf_file.stem + '_pr_endpoints.vtp')
    centerlines_output_file=proj_dir / (surf_file.stem + '_pr_centerlines.vtp')
    #print(surf_output_file, points_output_file, centerlines_output_file)
    ref_defined_outfile = proj_dir/('surf_refdefined.vtp')
    ND_defined_outfile = proj_dir/('surf_NDdefined.vtp')

    start = time.time()
    #print('\n' + surf_file.stem)

    if not surf_output_file.exists():
        surf = pv.read(surf_file)
        surf.clean()
        # Make option for smoothing here
        # surf = cc.vtk_taubin_smooth(surf, pass_band=0.05, iterations=50)
        if surf_type=='pt':
            neck_geodesic_points = None
        else:
            neck_geodesic_points = pv.read(neckpoints_file)

            if neck_geodesic_points is not None:
                s = cc.SelectGeodesic(surf, scalars='Mask')
                s.use_stored_points(neck_geodesic_points)
                surf = s.mesh

        points = pv.read(point_file)
        in_points = points['inlets'].points
        out_points = points['outlets'].points
        if surf_type=='a':   
            an_points = points['aneurysms'].points
            anubool=True
        else:
            an_points = None
            anubool = False
        '''
        #Commenting this section out because it is not necessary. We already have the
        #flow extensions from the preparation script!
        if 'normals' in points['inlets'].point_data:
            # Clip
            for pdx in range(points['inlets'].n_points):
                t = TubeClipper(surf)
                t.clip(points['inlets'].points[pdx], points['inlets'].point_data['normals'][pdx])
                surf = t.far_side
        
        if 'normals' in points['outlets'].point_data:
            # Clip
            for pdx in range(points['outlets'].n_points):
                t = TubeClipper(surf)
                t.clip(points['outlets'].points[pdx], -points['outlets'].point_data['normals'][pdx])
                surf = t.far_side

        if endpoints_pv is not None:
            print('Clipping with endpoints')
            surf_closed = surf.fill_holes(20.0) #not sure about this, may have to make a flag
            select = endpoints_pv.select_enclosed_points(surf_closed, tolerance=0.00001) #this doesn't work with a MuliBlock object
            
            for sdx in range(select.n_points):
                check = select.point_data['SelectedPoints'][sdx]
                if check == True:
                    print('t')
                    origin = select.points[sdx]
                    normal = select.point_data['normals'][sdx]
                    t = TubeClipper(surf)
                    t.clip(origin, normal)
                    surf = t.far_side
        '''
        m = Mesher(
            surf,
            inlet_points=in_points, 
            outlet_points=out_points,
            aneurysm_points=an_points,
            include_aneurysms=anubool
            )

        m.update_inlets_outlets()

        if m.surf.n_points > 1000:
            m.decimate_surface(target_edge_length=0.5)

        #Get any refinement regions for the PT mesh
        if (surf_type=='pt') and (ref == 'refine') and (not ref_defined_outfile.exists()):
            s_pt = cc.RefinementSelection(m.surf, title='Choose Refinement Zone')
            s_pt.select() #Selects the region of interest
            s_pt.define_surface() #Adds data attribute to point array called 'RefinementPoints'            
            m.surf= s_pt.surf
            '''
            This goes with the old method

            m.surf.point_data['vtkOGIds'] = list(range(m.surf.n_points))
            submesh = m.surf.extract_points(s_pt.surf.point_data['RefinementPoints']==1)
            
            #Check that the ids still match
            #p=pv.Plotter()
            #p.add_mesh(m.surf,color = 'blue', style='wireframe', show_edges=True)
            #p.add_points(submesh, color='r')
            #p.show()

            submesh_ids = submesh.point_data['vtkOGIds'].copy()
            submesh_array = np.zeros(m.surf.n_points, dtype=int)
            submesh_array[submesh_ids] = 1
            m.surf.point_data['RefinementPoints'] = submesh_array.astype(bool)
            '''
            m.surf.save(proj_dir/('surf_refdefined.vtp'))    
            #print(m.surf.point_data)
        elif (ref_defined_outfile.exists()):
            m.surf = pv.read(ref_defined_outfile)

        if (ND == 'nd') and (not ND_defined_outfile.exists()):
            e_pt = cc.RefinementSelection(m.surf, name='Enlarge_Cells', title = 'Mark Non-Dominant Side')
            e_pt.select()
            e_pt.define_surface()  
            m.surf.point_data['Enlarge_Cells'] = e_pt.surf.point_data['Enlarge_Cells']
            m.surf.save(proj_dir/('surf_NDdefined.vtp'))    
        elif ND_defined_outfile.exists():
            #NOTE: ref_defined_outfile must have been generated in the same script as this file!!
            nd_surf = pv.read(ND_defined_outfile)
            m.surf.point_data['Enlarge_Cells'] = nd_surf.point_data['Enlarge_Cells']

        if surf_type=='a': 
            m.generate_centerlines()
            m.generate_centerlines(include_aneurysms=False) 

        if multi_inlets == 'multi':
            m.generate_centerlines_multi(proj_dir)               
        else:
            m.generate_centerlines(include_aneurysms=False)

        if surf_type=='a': 
            m.branch_centerlines()
            endpoints_pv = m.clip_endpoints_with_tubeclipper(endpoints_pv=endpoints_pv, include_aneurysms=anubool) #includes aneurysms by default
            m.update_inlets_outlets()
            m.get_bifurcation_ref_systems_vectors()

        #These need to be optional, based on the flowrate and the size of the vessel
        #For Dan's aneurysm cases, he appears to have used the following, which is probably too fine for the PT cases:
        # min_edge_size=0.1, max_edge_size=0.4, sac_size=0.15, misr_min=0.1, misr_max=2.5
        #I am going to mess with the defaults here, but keep the aneurysm defaults on the actual function
        m.surface_preparation(proj_dir, multi_inlets, ref, ND, fix_centerline, neck_points=neck_geodesic_points, min_edge_size=min_EL, max_edge_size=max_EL, sac_size=0.15, misr_min=1, misr_max=7)
        m.surf.clean()
        m.update_inlets_outlets()
   
        # # This stuff is for getting plc points and 
        # # parent regions for SCI
        # m.get_mean_segments()
        # m.branch_centerlines()
        # m.update_aneurysm_group_ids()
        # m.get_branch_endpoints()

        # n_spheres=4
        # m.mark_distance_from_sacs(n_spheres)
        # m.mark_near_vessel_regions(m.surf, n_spheres=n_spheres)
        # # plc_points = m.get_plc_points() 

        # m.surf.save(surf_output_file)
        if (multi_inlets == 'multi'):
            #NOTE: Always check the result of this!!!
            m.generate_centerlines_multi(proj_dir)
        elif (multi_inlets == 'single') :
            m.generate_centerlines(include_aneurysms=anubool)

        if surf_type=='a': 
            m.generate_centerlines(include_aneurysms=False)
            m.branch_centerlines()
            m.centerlines.save(centerlines_output_file)
        elif (fix_centerline == 'reg') and (multi_inlets == 'multi'):
            m.branch_centerlines_pt()
            m.centerlines.save(centerlines_output_file)
        m.update_inlets_outlets()
        
        if surf_type=='a':                            
            m.get_bifurcation_ref_systems_vectors()
            m.update_inlets_outlets()
            m.get_mean_segments()
            m.update_aneurysm_group_ids()        
            m.get_branch_endpoints()
            n_spheres=4
            m.mark_distance_from_sacs(n_spheres)
            m.mark_near_vessel_regions(m.surf, n_spheres=n_spheres)
            m.update_inlets_outlets()
        
        # m.surf.plot(scalars='sac_zones')
        
        m.surf.save(surf_output_file)
        m.save_inlet_outlet_points(points_output_file, include_aneurysms=anubool)


        print('\n Case done', timedelta(seconds=time.time() - start))

    else:
        print('Output surface exists.')



if __name__ == "__main__":
    """
    Call this script using the following formula:
    surface_process.py path/to/case surf_type multi_inlets ref fix_centerline min_EL max_EL ND
    
    Info about the input arguments:

    path/to/case: must be a path with a target separated with min two/max three underscores, eg. ./case_1_proc or ./case_1_proc_refined.
    It will look in the directory ./case_1 for your prepared surface.

    surf_type: options are a for aneurysm or pt for pulsatile tinnitus
    
    multi_inlets: options are multi for multiple inlet cases or single for single inlet cases
    
    ref: options are refine or no_ref. Use if you want a refinment patch
    
    fix_centerline: options are reg, fix. Use the 'fix' option for if you want to use MISR instead of distance to centerlines
    
    min_EL/max_EL: specify floats that correspond to the minimum and maximum desired edgelengths
    
    ND: options are none or nd for whether or not you want to enlarge certain edge lengths (eg. the non-dominant side in a PT case)
    
    A full example of a command to put in a terminal that would call this script is as follows:
    surface_process.py ./case_1_low pt multi no_ref fix 0.4 0.5 nd

    NOTE: you do not need to precede this command with python because it already knows it is a python script.
    """
    proj_dir = Path(sys.argv[1])  

    if len(sys.argv[1].split('_')) == 4:
        proc_dir = sys.argv[1].split('_')[0], '_', sys.argv[1].split('_')[1], '_',sys.argv[1].split('_')[2] 
    else:
        proc_dir = sys.argv[1].split('_')[0], '_', sys.argv[1].split('_')[1]

    proc_dir = Path(''.join(str(i) for i in proc_dir))
    #print(proc_dir)
    if len(sys.argv) == 3:
        endpoints_f = Path(sys.argv[2])
        ref = 'no_ref' 
        if endpoints_f.exists():
            endpoints_pv = pv.read(endpoints_f) 
            surf_type = 'a'
        else:
            endpoints_pv = None
            surf_type = 'a'
        min_EL = 0.1
        max_EL = 0.4
        multi_inlets = 'single'
        ND = 'none'
        fix_centerline = 'reg'
    elif len(sys.argv) > 3:
        endpoints_f = Path(sys.argv[2])
        if endpoints_f.exists():
            endpoints_pv = pv.read(endpoints_f) #have to be PolyData type (vtp) to work?
            surf_type=sys.argv[3]
            multi_inlets = sys.argv[4] #options are 'single' and 'multi'
            ref = sys.argv[5] #options are 'refine' or 'no_ref' 
            fix_centerline = sys.argv[6] #options are 'reg', 'fix'
            min_EL = float(sys.argv[7])
            max_EL = float(sys.argv[8])  
            ND = sys.argv[9]
        else:
            endpoints_pv = None
            surf_type=sys.argv[2] 
            multi_inlets = sys.argv[3] #options are 'single' and 'multi'
            ref = sys.argv[4] #options are 'refine' or 'no_ref' 
            fix_centerline = sys.argv[5] #options are 'reg', 'fix'
            min_EL = float(sys.argv[6])
            max_EL = float(sys.argv[7])
            ND = sys.argv[8] # options are 'none' and 'nd'

    #print(proc_dir)
    surface_process(proj_dir, proc_dir, surf_type, multi_inlets, ND, min_EL, max_EL, ref, fix_centerline, endpoints_pv)
