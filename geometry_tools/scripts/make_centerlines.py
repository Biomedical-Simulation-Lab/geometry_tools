"""
This file includes a method to create network centerlines for a surface mesh, which contains the VMTK
attributes, as well as the cross-sectional area and perimeter of the surface mesh.
Commented out lines may be useful. Two other ways to generate centerlines included.

Call this using:

make_centerlines.py proj_dir case_name

where proj_dir is the directory you are looking for the surface file in (should end in "cl_mapped.vtp"),
and the case_name is what you want your centerline to be called ("case_name_centerline.vtp")
"""

import numpy as np
import pyvista as pv
from geometry_tools.meshing import Mesher
from geometry_tools import vmtk_wrapper as vmtk
from geometry_tools import common as cc
from scipy.spatial import cKDTree as KDTree 
from pathlib import Path
import sys

def polyline_from_points(points):
        poly = pv.PolyData()
        poly.points = points
        the_cell = np.arange(0, len(points), dtype=np.int_)
        the_cell = np.insert(the_cell, 0, len(points))
        poly.lines = the_cell
        return poly

def cleanup_lines(surf, poly, centerlines, graph):
    graph_pts=graph.points
    p = pv.Plotter()
    p.add_mesh(surf, opacity=0.3)
    p.add_mesh(centerlines.points, color='b', render_points_as_spheres=True, point_size = 20)
    ids=range(0,len(poly.points))
    #gids = range(0,len(graph_pts))
    p.add_mesh(poly, color='r')
    p.add_point_labels(poly.points, ids, point_color='r', render_points_as_spheres=True, point_size=20, font_size=18)
    #p.add_point_labels(graph_pts, gids, point_color='g', render_points_as_spheres=True, point_size=30, font_size=18)
    p.add_title('Inspect for bad points and record ids!')
    p.show()
    info = input('Any bad points[y/n]?')
    if info == 'y':
        lst = []
        n = int(input("Enter number of bad points: "))
        for i in range(0, n):
            ele = int(input('Enter id of bad point:'))
            lst.append(ele)
        points = np.delete(poly.points, lst, axis=0)
        poly = polyline_from_points(points)
    '''
    end = input('Does the endpoint need to be reset [y/n]?')
    if end == 'y':
        gpt = input('Enter the graph point id you want to reset to:')
        poly.points[0]=graph_pts[int(gpt)]
    beg = input('Does the start point need to be reset [y/n]?')
    if beg == 'y':
        gpt = input('Enter the graph point id you want to reset to:')
        poly.points[-1]=graph_pts[int(gpt)]
    '''
    return poly

def fenest(tree2, surf, graph, centerline, centerlines, origin):
    print('There is a fenestration present! ')
    '''
    select = cc.ClickDragSelect(graph_lines, title = 'Select first side of fenestration')
    graph_lines.cell_data['fen1'] = select.mesh.cell_data['PickedMask']
    select = cc.ClickDragSelect(graph_lines, title = 'Select second side of fenestration')
    graph_lines.cell_data['fen2'] = select.mesh.cell_data['PickedMask']
    graph_lines = graph_lines.cell_data_to_point_data()
    line1 = polyline_from_points(graph_lines.points[graph_lines.point_data['fen1']==1])
    line2 = polyline_from_points(graph_lines.points[graph_lines.point_data['fen2']==1])
    return line1+line2
    '''
    val_out1 = input("What is the point of the fenestration that meets with the SS?")
    centerlines_seg1, out_id = one_branch(tree2, surf, graph, centerline, origin, val_out1, ss_or_sss=True, fen = True)
    centerline_seg1 = cleanup_lines(surf, centerlines_seg1, centerlines, graph)
    val_out2 = input("\nWhat is the point of the fenestration that meets with the SSS?")
    centerlines_seg2, out_id = one_branch(tree2, surf, graph, centerline, origin, val_out2, ss_or_sss=True, fen = True)
    centerline_seg2 = cleanup_lines(surf, centerlines_seg2, centerlines, graph)
    ss_ept = centerline_seg1.points[0]
    sss_ept = centerline_seg2.points[0]
    val_out = input("\nWhat is the patient left point of the fenestration?")
    centerlines_seg3, out_id = one_branch(tree2, surf, graph, centerline, ss_ept, val_out, ipt_given=True)
    centerline_seg3 = cleanup_lines(surf, centerlines_seg3, centerlines+centerlines_seg1, graph)
    centerlines_seg4, out_id = one_branch(tree2, surf, graph, centerline, sss_ept, val_out, ipt_given=True)
    centerline_seg4 = cleanup_lines(surf, centerlines_seg4, centerlines+centerlines_seg2, graph)
    return centerlines_seg1+centerlines_seg2+centerlines_seg3+centerlines_seg4, ss_ept, sss_ept

def one_branch(tree2, surf, graph, centerline, val_in, val_out, ss_or_sss=False, fen = False, ipt_given=False, ept_given=False):
    if fen == False:
        if ipt_given==True:
            inlet_point=val_in
        else:    
            inlet_point = graph.points[int(val_in)]
    else:
        in_id = val_in
        inlet_point = centerline.points[in_id]
    if ss_or_sss == True: #set this point equal to the graph point, not the centerline point!
        out_point = graph.points[int(val_out)]
        outlet_point = out_point
        out_id = None
    else:    
        if ept_given==True:
            outlet_point = val_out
            out_id = None
        else:
            out_point = graph.points[int(val_out)]
            out_id=tree2.query(out_point, k=1)[1]
            outlet_point = centerline.points[out_id]

    surf_capped = pv.PolyData()
    surf_capped.copy_structure(vmtk.surface_capper(surf))
    tree = KDTree(surf_capped.points)
    inlet_ids = tree.query(inlet_point, k=1)[1]
    
    outlet_ids = tree.query(outlet_point, k=1)[1]
    centerlines_seg = vmtk.centerlines(
        surf_capped, 
        seed_selector='idlist', 
        resampling_step_length = 1,
        src_ids=[inlet_ids],
        target_ids=[outlet_ids]
        )
    #make sure to set the first and last points to the inlet and outlet points
    if ept_given==True:
        centerlines_seg.points[0]=outlet_point
    else:
        centerlines_seg.points[0]=centerline.points[tree2.query(outlet_point, k=1)[1]]
    centerlines_seg.points[-1]=inlet_point

    return centerlines_seg, out_id


def iter_branches(surf, graph, centerline, fen='n'):
    tree2 = KDTree(centerline.points)
    nbranches = input('\nHow many branches? NOTE: In bilateral models, the SS is not counted as a branch!')
    if fen=='y':
        nb = input('How many branches up from the patient right outlet before the fenestration?')
        i_seg = input('Is there a main branch segment below it [y/n]?')
        if i_seg == 'y':
            fb_p1 = input('What is the lower id of the main branch segment below it?')
            fb_p2 = input('What is the upper id of the main branch segment below it?')
        o_seg = input('Is there a main branch segment above it (if only the left outlet, say n) [y/n]?')
        if o_seg == 'y':
            fa_p1 = input('What is the lower id of the main branch segment above it?')
            fa_p2 = input('What is the upper id of the main branch segment above it?')
    #Branches are ordered from the bottom up!!
    for b in range(int(nbranches)):
        val_in = input("What is the inlet point id of branch {}?".format(b))
        val_out = input("What is the outlet point id of branch {}?".format(b))
        centerlines_seg, out_id = one_branch(tree2, surf, graph, centerline, val_in, val_out)
        ss_ept=None
        sss_ept=None
        if b == 0:
            #The branch segment
            centerlines = centerlines_seg
            #From the outlet to the last branch
            old_outid = 0
        else:
            #The branch segment
            centerlines += centerlines_seg
        #The segment in between two branches
        if (fen == 'y' and b == int(nb)):
            print('Building pre-fen centerline')
            if i_seg == 'y':
                inlet_point = graph.points[int(fb_p1)]
                out_point = graph.points[int(fb_p2)]
                out_id=tree2.query(out_point, k=1)[1]
                inlet_id = tree2.query(inlet_point, k=1)[1]
                p = centerline.points[inlet_id:out_id]
                centerline_segfb = polyline_from_points(p)

                centerlines += centerline_segfb
            print('Building fenestration centerline')
            fc, ss_ept, sss_ept = fenest(tree2, surf, graph, centerline, centerlines, origin = out_id)
            centerlines +=fc

            print('Building post-fen centerline')
            if o_seg == 'y':
                inlet_point = graph.points[int(fa_p1)]
                out_point = graph.points[int(fa_p2)]
                out_id=tree2.query(out_point, k=1)[1]
                inlet_id = tree2.query(inlet_point, k=1)[1]
                p = centerline.points[inlet_id:out_id]
                centerline_segfa = polyline_from_points(p)

                centerlines += centerline_segfa
        else:
            points = centerline.points[old_outid:out_id]
            line = polyline_from_points(points)
            centerlines +=line
        old_outid = out_id
    return centerlines, out_id, ss_ept, sss_ept

def make_cl(proj_dir, case_name):
    out_dir = proj_dir
    surf_file = sorted(proj_dir.glob('*cl.vtp'))[0]
    surf=pv.read(surf_file)

    m = Mesher(
            surf,
            include_aneurysms=False
            )
    #m.set_inlets_outlets()
    #m.generate_centerlines(include_aneurysms=False, endpoints=1)
    
    graph_lines , graph = vmtk.network_extractor(surf, ratio = 1.01)
    #graph.save(out_dir/('graph.vtp'))
    p = pv.Plotter()
    p.add_mesh(surf, opacity=0.3)
    labels = [str(i) for i in range(len(graph.points))]
    p.add_point_labels(graph.points, labels, point_size=30, font_size=20, always_visible=True, render_points_as_spheres=True)
    p.add_mesh(graph_lines)
    p.add_text("Look for inlet and outlet labels for branches and fenestration, if present",position='upper_left', font_size = 14)
    p.add_text("Note: graph points in the main branch will have multiple values, Choose one.",position='lower_left', font_size = 12)
    p.show()
    
    centerlines = pv.PolyData()
    bilat = input('Is this a bilateral model [y/n]?')
    torc = input('Is there a fenestration at the torcula [y/n]?')
    fen = input('Is there a fenestration somewhere else [y/n]?')
    if fen =='y':
        print('Fenestrations not at the torcula not currently supported! Use centerlines_fixed and resample the line.')
        exit(1)
    centers = m.get_open_profiles()
    if bilat == 'n':
        inlet_id = m._pick_points(pv.wrap(centers), 'Pick Major Inlet')
        outlet_id = m._pick_points(pv.wrap(centers), 'Pick Major Outlet')
        m.inlet_ids = [inlet_id]
        m.inlet_points = centers[inlet_id]
        m.outlet_ids = [outlet_id]
        m.outlet_points = centers[outlet_id]
        m.generate_centerlines(include_aneurysms=False, endpoints=1)
        centerlines, out_id, _, _ = iter_branches(surf, graph, m.centerlines, fen=fen)

        #from the last branch to the inlet
        points2 = m.centerlines.points[out_id:]
        line2 = polyline_from_points(points2)
        centerlines +=line2
        if torc == 'y':
            ss = input('Is there an SS [y/n]?')
            if ss =='y':
                val_in = input("What is the inlet point id of the SS?")
                val_out = input("What is the outlet point id of the SS?")
                centerlines_seg, _ = one_branch(tree2, surf, graph, val_in, val_out)
                centerlines += centerlines_seg
                
    else:
        inlet_id = m._pick_points(pv.wrap(centers), 'Pick Patient Left Outlet')
        outlet_id = m._pick_points(pv.wrap(centers), 'Pick Patient Right Outlet')
        m.inlet_ids = [inlet_id]
        m.inlet_points = centers[inlet_id]
        m.outlet_ids = [outlet_id]
        m.outlet_points = centers[outlet_id]
        m.generate_centerlines(include_aneurysms=False, endpoints=1)
        if torc == 'y':
            fen = 'y'
        centerlines, out_id, ss_ept, sss_ept = iter_branches(surf, graph, m.centerlines, fen=fen)
        #from the last branch to the inlet
        points2 = m.centerlines.points[out_id:]
        line2 = polyline_from_points(points2)
        centerlines +=line2
        ss = input('Is there an SS [y/n]?')
        if ss =='y':
            val_in = input("What is the inlet point id of the SS?")
            if torc == 'y':
                tree2 = KDTree(m.centerlines.points)
                centerlines_seg, _ = one_branch(tree2, surf, graph, m.centerlines, val_in, ss_ept, ept_given=True)
                centerlines_seg  = cleanup_lines(surf, centerlines_seg, centerlines, graph)
            else:
                val_out = input("What is the outlet point id of the SS?")
                centerlines_seg, _ = one_branch(tree2, surf, graph, m.centerlines, val_in, val_out)
            centerlines += centerlines_seg
        #There has to be an SSS
        val_in = input("What is the inlet point id of the SSS?")
        if torc == 'y':
            centerlines_seg, _ = one_branch(tree2, surf, graph, m.centerlines, val_in, sss_ept, ept_given=True)
            centerlines_seg  = cleanup_lines(surf, centerlines_seg, centerlines, graph)
        else:
            val_out = input("What is the outlet point id of the SSS?")
            centerlines_seg, _ = one_branch(tree2, surf, graph, m.centerlines, val_in, val_out)
        centerlines += centerlines_seg

    centerlines = vmtk.centerline_geometry(centerlines)
    '''
    m.centerlines, _ = vmtk.network_extractor(m.surf)
    m.centerlines = vmtk.resample_cl(m.centerlines, length=1.5) #make sure this matches with other files!
    m.centerlines = vmtk.centerline_geometry(m.centerlines)

    tree1 = KDTree(m.centerlines.points)
    tree2 = KDTree(m.surf.points)
    dist, idx = tree2.query(m.centerlines.points) #closest dist to centerline point
    for i in range(len(idx)):
        for j in range(len(idx)):
            if (idx[j]==idx[i]) and (i != j):
                closest, _ = tree1.query(m.surf.points[idx[i]]) #closest centerline distance to the point
                dist[j]=closest
    m.centerlines.point_data['MaximumInscribedSphereRadius']=dist
    #p = pv.Plotter()
    #p.add_mesh(surf, color="gray", opacity=0.5)
    #p.add_mesh(centerline, color="red", point_size=20)
    #p.add_mesh(m.centerlines, color="blue")
    #p.show()
    #print(centerline)
    #Use the centerline points to create planes
    m.centerlines.point_data['CSA']=np.array(m.centerlines.n_points)
    m.centerlines.point_data['perimeter']=np.array(m.centerlines.n_points)
    points = m.centerlines.points
    normals = m.centerlines.point_data['FrenetTangent'] 
    for ndx, pt in enumerate(points):
        plane=pv.Plane(center = pt, direction = normals[ndx], i_size=20, j_size=20, i_resolution=100, j_resolution=100).triangulate()
        plane_split = plane.clip_surface(surf)
        split = plane_split.split_bodies()
        if len(split)>1:
            cm = np.zeros((len(split),3))
            for i in range(len(split)):
                cm[i, :] = split[i].center_of_mass()
            tree = KDTree(cm)
            _, j = tree.query(pt) #closest center of mass to the centerline point
            plane_split = split[j]
        CSsurf = plane_split.extract_surface() 
        area = CSsurf.area
        edges = CSsurf.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
        #p=pv.Plotter()
        #p.add_mesh(CSsurf)
        #p.add_mesh(edges, color='red')
        #p.show()
        sized = edges.compute_cell_sizes()
        #print(sized)
        perimeter = sum(sized['Length'])
        m.centerlines.point_data['CSA'][ndx]=area
        m.centerlines.point_data['perimeter'][ndx]=perimeter
    '''
    centerlines.save(out_dir/(case_name+'_centerline.vtp'))

if __name__ == "__main__":
    proj_dir = Path(sys.argv[1])
    case_name = sys.argv[2] 
    make_cl(proj_dir=proj_dir, case_name=case_name)
