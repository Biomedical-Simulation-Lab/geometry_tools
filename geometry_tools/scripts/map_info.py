"""
This file contains a method for preparing a segmented surface mesh that has been run through 'surface_prep.py'
for creating the mappings for a PT surface mesh. Currently only does unilateral.

Call this file using:
map_info.py prep_dir sss ss lab fen syl trol emissary condylar

Where
-prep_dir is the directory your surface mesh from 'surface_prep.py' is stored
-sss is a float indicating the Superior Saggital Sinus flow rate at peak systole in mL/s
-ss is a float indicating the Straight Sinus flow rate at peak systole in mL/s
-lab is a float indicating the Labbe flow rate at peak systole mL/s
-fen indicates True or False if there is a fenestration (currently only set up for one)
-syl is a float indicating the Sylvian vein flow rate at peak systole mL/s
-trol is a float indicating the Trolard vein flow rate at peak systole mL/s
-emissary is a float indicating the Emissary vein outlet ratio
-condylar is a float indicating the Condylar vein outlet ratio

defaults to one flow rate for the whole geometry, which is 6.816019219 for peak systolic Superior Sinus inflow

This will produce a number of useful attributes, including the boundary layer width to get y+<1
NOTE: if you delete planes, you will have to delete the mapped surface and run this a second time :)

"""
import numpy as np
import pyvista as pv
from geometry_tools.meshing import Mesher
import geometry_tools.vmtk_wrapper as vmtk
import geometry_tools.common as cc
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import interp1d
import matplotlib as plt
from matplotlib import pyplot
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

size = 10
plt.rc('font', size=size) #controls default text size
plt.rc('axes', titlesize=size) #fontsize of the title
plt.rc('axes', labelsize=size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=size) #fontsize of the y tick labels
plt.rc('legend', fontsize=size) #fontsize of the legend

def define_fr(obj_pt, flowrate=5.578888889):
    cell_ids = obj_pt.surf.faces.reshape(-1, 4)[obj_pt.surf.cell_data[obj_pt.name]==1][:,1:]
    ids = cell_ids.flatten()
    obj_pt.surf.point_data[obj_pt.name][ids]=flowrate
    return obj_pt.surf

def mapped_info(prep_dir, sss, ss, lab, fen, syl, trol, emissary, condylar, plot_pressure):
    out_dir = prep_dir.parent
    #surf0_file = sorted(prep_dir.glob('*_noext.vtp'))[0]
    surf_file = sorted(prep_dir.glob('*_cl.vtp'))[0]
    remeshed_file = out_dir/(surf_file.stem  +'_remeshed.vtp')
    dec = str(int(round(float(sss) - int(float(sss)), 1)*10))
    #cent_graph_file = out_dir/(surf_file.stem  +'_centerline_graph_' + str(int(float(sss))) + 'p' + dec + '.vtp')
    #graphed_cl_file = out_dir/(surf_file.stem +'_centerline_graph.vtp')
    cent_graph_vmtk = out_dir/(surf_file.stem +'_centerline_graph_vmtk.vtp')
    if not cent_graph_vmtk.exists(): #if not using centerlines_fixed but rather make_centerlines
        cent_graph_vmtk = prep_dir/(surf_file.stem +'_centerline.vtp')
    cent_file = out_dir/(surf_file.stem + '__' + str(int(float(sss))) + 'p' + dec + 'centerline_mapped.vtp')
    mapped_file = out_dir/(surf_file.stem + '_' + str(int(float(sss))) + 'p' + dec + '_mappedsys.vtp')
    planes_files = out_dir/(surf_file.stem + '_planes_' + str(int(float(sss))) + 'p' + dec + '.vtm')
    if not mapped_file.exists():
        #surf = pv.read(surf0_file) #use unprepped surface for the centerline map
        if not remeshed_file.exists(): #use remeshed surface
            surf = vmtk.surface_remeshing(pv.read(surf_file), edgelength=0.5, iterations=5)
            surf.save(remeshed_file)
        else:
            surf=pv.read(remeshed_file)

        m = Mesher(
                surf,
                include_aneurysms=False
                )
        #m.clip_boundaries()
        #print(cent_file)
        if not cent_file.exists():
        
            m.centerlines = pv.read(cent_graph_vmtk)
            '''
            #first, generate a centerline
            cent, graph = vmtk.network_extractor(m.surf)#vmtk.centerline_geometry(m.centerlines)
            graph.save(cent_graph_file)
            cent.save(graphed_cl_file)
            #use vmtk for each segment
            centerline = pv.PolyData()
            for idx in range(1, len(graph.points), 2):
                inlet_id = [idx-1]
                m.inlet_points = graph.points[inlet_id]
                outlet_id = [idx]
                m.outlet_points = graph.points[outlet_id]
                m.generate_centerlines(include_aneurysms=False)
                if idx == 1:
                    centerline = m.centerlines
                else:
                    centerline += m.centerlines
            m.centerlines = centerline
            '''
            m.centerlines = vmtk.resample_cl(m.centerlines, length=1.5) #so we don't have as many planes and points are equispaced
            #Use the centerline points to create planes
            planes = pv.MultiBlock()
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
                CSsurf.point_data['centerline_id']=ndx
                planes.append(CSsurf)
            m.centerlines.save(cent_file)
            planes.save(planes_files)
        else:
            m.centerlines = pv.read(cent_file)
            planes = pv.read(planes_files)
        #Create mapping to surface
        #Label segments
        #How many main segments do we have? Assuming only unilateral wiht current options
        main_segs=1
        branches = []
        if ss != 'False':
            branches.append('Straight_Sinus')
            main_segs += 1
        if lab != 'False':
            branches.append('Labbe')
            main_segs += 1
        if syl != 'False':
            branches.append('Sylvian_Vein')
            main_segs += 1
        if trol != 'False':
            branches.append('Trolard_Vein')
            main_segs += 1
        if emissary != 'False':
            branches.append('Emissary_Vein')
            main_segs += 1
        if condylar != 'False':
            branches.append('Condylar_Vein')
            main_segs += 1 

        #Some parts of the centerline shouldn't be used to calculate parameters, so take these bits out
        #While we are at it, identify the other branches for later 
        labelled_centerlines = cc.Label_CLs(m.surf, m.centerlines, main_segs, branches, fen)
        m.centerlines.point_data['unusable']=labelled_centerlines.centerline.point_data['branch_centerlines']
        for b in branches: #get all labelled branch centerline points
            m.centerlines.point_data[b]=labelled_centerlines.centerline.point_data[b]
        if fen == 'True':
            m.centerlines.point_data['fen1']=labelled_centerlines.centerline.point_data['fen1']
            m.centerlines.point_data['fen2']=labelled_centerlines.centerline.point_data['fen2']
        m.centerlines.point_data['main_branch']=labelled_centerlines.centerline.point_data['main_branch']

        #don't need to do this next line anymore since we remeshed the surface already
        #m.surf = pv.read(surf_file) #replace surface with flow extension surface to avoid the flow extension issues
        usable_ids0 = np.asarray(np.where(m.centerlines.point_data['unusable']==0))[0]
        #print(usable_ids, m.centerlines.points.shape, m.centerlines.points[usable_ids].shape)
        usable_CL=m.centerlines.points[usable_ids0]
        usable_CL_CSA=m.centerlines.point_data['CSA'][usable_ids0]
        usable_CL_perimeter=m.centerlines.point_data['perimeter'][usable_ids0]
        usable_planes = pv.MultiBlock()
        usable_ids = []
        #print(usable_ids)
        for i in usable_ids0:
            if ("Block-0{}".format(i) in planes.keys()) or ("Block-{}".format(i) in planes.keys()):
                usable_planes.append(planes[i])
                usable_ids.append(i)
        #usable_planes.save(out_dir/(surf_file.stem + '_usable_planes.vtm'))
        merged_usable_planes=usable_planes.combine()   
        planes_points = merged_usable_planes.points
        tree = KDTree(planes_points) #only include usable planes
        _, idx_p = tree.query(m.surf.points) #get plane points closest to surf points

        #get the centerline id of the usable plane and assign the CSA at that centerline point to the surface
        cl_ids = merged_usable_planes.point_data['centerline_id']
        cntr_ids = cl_ids[idx_p].astype(int) #centerline ids corresponding to the plane at the surface point
        m.surf.point_data['CSA']=m.centerlines.point_data['CSA'][cntr_ids]
        m.surf.point_data['perimeter']=m.centerlines.point_data['perimeter'][cntr_ids]

        #if wonky planes were deleted, we need to remove the data at those centerline points and replace with weighted average data between the two neighbouring points
        cntr_ids_avg = np.ones(len(m.centerlines.points), dtype=bool)
        cntr_ids_avg[usable_ids]=False #np.asarray([x for x in range(len(m.centerlines.points)) if x not in usable_ids])#cntr_ids.tolist()])
        if len(cntr_ids_avg) != 0:
            tree_avg = KDTree(m.centerlines.points[usable_ids])
            dist_avg, cind = tree_avg.query(m.centerlines.points[cntr_ids_avg], k=5) #get five closest points on centerlines
            #inverse distance average merged_usable_planes.point_data['centerline_id']
            m.centerlines.point_data['CSA'][cntr_ids_avg]=np.sum((1/(dist_avg+0.0000001))*m.centerlines.point_data['CSA'][cl_ids[cind].astype(int)], axis=1)/np.sum((1/(dist_avg+0.0000001)), axis=1)
            m.centerlines.point_data['perimeter'][cntr_ids_avg]=np.sum((1/(dist_avg+0.0000001))*m.centerlines.point_data['CSA'][cl_ids[cind].astype(int)], axis=1)/np.sum((1/(dist_avg+0.0000001)), axis=1)
        
        #print(np.isnan(np.sum(m.surf.point_data['CSA'])), np.isnan(np.sum(m.surf.point_data['perimeter'])))
        m.surf, neighbour_pts = cc.smooth_mesh_data_local_alt(m.surf, array='CSA', iterations = 10)
        m.surf, _ = cc.smooth_mesh_data_local_alt(m.surf, array='perimeter', neighbour_pt_ids = neighbour_pts, iterations = 10)
        #print(np.isnan(np.sum(m.surf.point_data['CSA'])), np.isnan(np.sum(m.surf.point_data['perimeter'])))
        
        m.surf.save(mapped_file)
        m.centerlines.save(cent_file)

    surf=pv.read(mapped_file) #for some reason have to read in again. boolean array dimension error. Fix this.
    
    m = Mesher(
            surf,
            include_aneurysms=False,
            )
    m.centerlines=pv.read(cent_file)
    planes = pv.read(planes_files)
    #Select flowrate regions assuming unilateral
    flowrate = float(sss) #Superior Saggital Sinus at Peak systolic
    m.surf.point_data['flowrate'] = np.zeros(m.surf.n_points)
    m.centerlines.point_data['flowrate']=np.zeros(m.centerlines.n_points)
    #Superior Saggital Sinus branch:
    #Use the first main branch segment to define SSS flowrate
    main_branch_seg=1
    m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if ss != 'False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Straight_Sinus']==1]=float(ss)
        flowrate += float(ss) #this is the current flowrate of the main branch
        main_branch_seg +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if lab !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Labbe']==1]=float(lab)
        flowrate += float(lab) #this is the current flowrate of the main branch
        main_branch_seg +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if syl !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Sylvian_Vein']==1]=float(syl)
        flowrate += float(syl) #this is the current flowrate of the main branch
        main_branch_seg +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if trol !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Trolard_Vein']==1]=float(trol)
        flowrate += float(trol) #this is the current flowrate of the main branch
        main_branch_seg +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if emissary !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Emissary_Vein']==1]=flowrate*float(emissary)
        flowrate -=flowrate*float(emissary) #this is the current flowrate of the main branch
        main_branch_seg +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if condylar !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Condylar_Vein']==1]=flowrate*float(condylar)
        flowrate -=flowrate*float(condylar) #this is the current flowrate of the main branch
        main_branch_seg +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if fen != 'False':
        #get average CSA ratio between branches:
        fen1_CSA = np.min(m.centerlines.point_data['CSA'][m.centerlines.point_data['fen1']==1])
        fen2_CSA = np.min(m.centerlines.point_data['CSA'][m.centerlines.point_data['fen2']==1])
        ratio1 = fen1_CSA/(fen1_CSA+fen2_CSA)
        ratio2 = fen2_CSA/(fen1_CSA+fen2_CSA)
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen1']==1]=ratio1*m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen1']==1]
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen2']==1]=ratio2*m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen2']==1]
    
    #get the dP value at every point on the entire centerline
    U_c = m.centerlines.point_data['flowrate']/m.centerlines.point_data['CSA']
    rho = 1057
    m.centerlines.point_data['dP']=0.5*rho*(3/2*U_c)**2/133.322 #dP based on max centerline velocity in mmHg calculated in every branch
    U_c_cycleaverage = (m.centerlines.point_data['flowrate']/1.221752101)/m.centerlines.point_data['CSA']
    m.centerlines.point_data['dP_cycle_average']= 0.5*rho*(3/2*U_c_cycleaverage)**2/133.322
    '''
    usable_CL=m.centerlines.points[m.centerlines.point_data['unusable']==0]
    usable_CL_flowrate=m.centerlines.point_data['flowrate'][m.centerlines.point_data['unusable']==0]
    tree3 = KDTree(usable_CL) #only include the usable centerlines
    _, idx_c3 = tree3.query(m.surf.points)
    m.surf.point_data['flowrate']=usable_CL_flowrate[idx_c3]
    '''
    if plot_pressure == True:
        def line_cm(pts,axis=0):
            seg_lens=np.zeros(len(pts))
            seg_lens[1:-1]=np.sqrt(np.sum(np.square(pts[0:-2,:]-pts[1:-1, :]),1)) #length of each segment
            disp=np.array([np.sum(seg_lens[0:ii]) for ii in range(len(seg_lens))]) #displacement in mm
            return disp/10 #displacement in cm

        #Order the points along the main branch - this will work for the unilateral case
        unordered_points = m.centerlines.points[m.centerlines.point_data['main_branch'] > 0]  #for bilateral case, use [m.centerlines.   ['main_branch_l'] != 0] or [m.centerlines.point_data['main_branch_r'] != 0] as the index instead
        map_indices = m.centerlines.point_data['main_branch'] > 0
        #map_indices = m.centerlines.point_data['fen2']!= 1

        # Set a seed point to start the ordering
        # Find the index of the point with the highest Z value, which should be the SSS inlet, but check this!! Can also choose a different seed point (eg. the outlet
        highest_z_index = np.argmax(unordered_points[:, 2])
        seed_point = unordered_points[highest_z_index]
        seed_index = highest_z_index

        # Remove the seed point from the list of unordered points
        remaining_points = np.delete(unordered_points, seed_index, axis=0)
        ordered_points = [seed_point]
        ordered_indices = [seed_index]

        kdtree_orig = KDTree(unordered_points)

        while remaining_points.shape[0] > 0:
            kdtree = KDTree(remaining_points)
            dist, index = kdtree.query(ordered_points[-1])
            _, ndx = kdtree_orig.query(remaining_points[index])
            nearest_point = remaining_points[index]
            ordered_points.append(nearest_point)
            ordered_indices.append(ndx)
            remaining_points = np.delete(remaining_points, index, axis=0)
            

        ordered_points = np.array(ordered_points)
        ordered_indices = np.array(ordered_indices)
        x = np.flip(line_cm(ordered_points))
        #Peak Systolic Pressure Drop
        pyplot.figure(1,figsize=(7, 4))
        dP_main = m.centerlines.point_data['dP'][map_indices][ordered_indices] #only take the pressure drop calculated on the main branch
        dP_main = np.flip(dP_main[0] - dP_main)
        dP_cycleaverage = m.centerlines.point_data['dP_cycle_average'][map_indices][ordered_indices]
        dP_cycleaverage = np.flip(dP_cycleaverage[0]- dP_cycleaverage)
        pyplot.plot(x, dP_main,color='b', label='$P_{peak}$', linewidth=0.5)
        pyplot.plot(x, dP_cycleaverage,color='r', label='$P_{avg}$', linewidth=0.5)
        pyplot.xlabel('Axial Position (cm)', labelpad=-1)
        pyplot.ylabel('Pressure Drop (mmHg)', labelpad=-4)
        pyplot.legend()
        pyplot.title('Pressure Peak Systole')
        pyplot.savefig(out_dir/(surf_file.stem + '_' + str(int(float(sss))) + 'p' + dec + 'Pressuredrop_1D.png')) #save figure
        np.savez(out_dir/(surf_file.stem + '_' + str(int(float(sss))) + 'p' + dec + 'Pressuredrop_1D.npz'), x=x, dP_main=dP_main, dP_average = dP_cycleaverage) #save the data for later

    usable_ids0 = np.asarray(np.where(m.centerlines.point_data['unusable']==0))[0]
    #usable_planes = pv.MultiBlock()
    usable_ids = []
    #print(usable_ids)
    for i in usable_ids0:
        if ("Block-0{}".format(i) in planes.keys()) or ("Block-{}".format(i) in planes.keys()):
            usable_planes.append(planes[i])
            usable_ids.append(i)
    merged_usable_planes=usable_planes.combine()      
    planes_points = merged_usable_planes.points
    usable_CL_flowrate=m.centerlines.point_data['flowrate'][m.centerlines.point_data['unusable']==0]
    usable_planes.save(out_dir/(surf_file.stem + '_usable_planes.vtm'))

    tree3 = KDTree(planes_points) #only include usable planes
    _, idx_p3 = tree3.query(m.surf.points)
    #np.array(usable_ids)
    cl_ids2 = merged_usable_planes.point_data['centerline_id']
    ctr_ids2 = cl_ids2[idx_p3].astype(int)
    m.surf.point_data['flowrate']=m.centerlines.point_data['flowrate'][ctr_ids2]
    #smooth data
    m.surf, _ = cc.smooth_mesh_data_local_alt(m.surf, array='flowrate', iterations = 5)

    nu = (0.0037/1057) #viscosity
    L = 4*m.surf.point_data['CSA']/m.surf.point_data['perimeter']*0.001#Hydraulic diameter (m)
    Deff=2*np.sqrt(m.surf.point_data['CSA']/(np.pi))*0.001 #Effective diameter (m)
    U = (m.surf.point_data['flowrate']/m.surf.point_data['CSA']) #peak systolic velocity in m/s
    U_cycleaverage = ((m.surf.point_data['flowrate']/1.221752101)/m.surf.point_data['CSA'])
    Re = U*L/nu
    Cf = 0.026/(Re**(1/7))
    Tw_rho=Cf*(U**2)/2
    Uf=np.sqrt(Tw_rho)
    m.surf.point_data['Dh']=L
    m.surf.point_data['Deff']=Deff
    m.surf.point_data['mean_velocity']=U
    m.surf.point_data['kolmog_len']=((nu**3)*L/(U**3))**(1/4)
    m.surf.point_data['taylor_len']=np.sqrt(10)*(Re**(1/4))*(((nu**3)*L/(U**3))**(1/4))
    m.surf.point_data['ds_max(y+=1)']=nu/Uf #  
    m.surf.point_data['dP']=0.5*rho*(3/2*U)**2/133.322 #dP based on max centerline velocity in mmHg calculated in every branch
    m.surf.point_data['dP_cycle_average']= 0.5*rho*(3/2*U_cycleaverage)**2/133.322
    m.centerlines.save(cent_file) 
    m.surf.save(mapped_file)

if __name__ == "__main__":
    prep_dir = Path(sys.argv[1]) 
    if len(sys.argv)>3:
        sss = sys.argv[2]
        ss=sys.argv[3]
        lab=sys.argv[4]
        fen=sys.argv[5]
        syl= sys.argv[6]
        trol= sys.argv[7]
        emissary = sys.argv[8]
        condylar = sys.argv[9]
        if len(sys.argv)>10:
            plot_pressure = sys.argv[10]
        else:
            plot_pressure = True #default is true
    else:
        sss = sys.argv[2]#6.816019219
        ss='False'
        lab='False'
        fen='False'
        syl = 'False'
        trol='False'
        emissary='False'
        condylar = 'False'
        if len(sys.argv)>3:
            plot_pressure = sys.argv[3]
        else:
            plot_pressure = True #default is true

    mapped_info(prep_dir=prep_dir, sss = sss, ss = ss, lab = lab, fen = fen, syl = syl, trol=trol, emissary=emissary, condylar=condylar, plot_pressure = plot_pressure)


'''
    Old crappy way. Leaving this here for reference, not for use!
    sss_pt = cc.RefinementSelection(m.surf, name = 'flowrate', title='Choose SupSagSinus Flowrate Zone')
    sss_pt.select() #Selects the region of interest
    flowrate = 6.816019219 #Peak systolic
    m.surf = define_fr(sss_pt, flowrate=flowrate) #Adds data attribute to point array called 'SSS'
    if ss != 'False':
        ss_pt = cc.RefinementSelection(m.surf, name = 'flowrate', title='Choose StrSinus Flowrate Zone')
        ss_pt.select() #Selects the region of interest
        m.surf = define_fr(ss_pt,flowrate=ss)
        comb1_pt = cc.RefinementSelection(m.surf, name = 'flowrate', title='Choose StrSinus+SupSagSinus Flowrate Zone')
        comb1_pt.select() #Selects the region of interest
        flowrate+=float(ss)
        m.surf = define_fr(comb1_pt, flowrate=flowrate) #Adds data attribute to point array called 'Comb1'
    
    #nondom stuff
    if nondom !='False':
        nd_pt = cc.RefinementSelection(m.surf, name = 'flowrate', title='Choose LessDom Flowrate Zone')
        nd_pt.select() #Selects the region of interest
        m.surf = define_fr(nd_pt, flowrate=flowrate*float(nondom))
        comb0_pt = cc.RefinementSelection(m.surf, name = 'flowrate', title='Choose Dom Flowrate Zone')
        comb0_pt.select() #Selects the region of interest
        flowrate -= float(nondom)*flowrate
        flowrate_nondom=float(nondom)*flowrate
        m.surf = define_fr(comb0_pt, flowrate=flowrate)
    if outflow2 !='False': #non-dom side outflow
        of0_pt = cc.RefinementSelection(m.surf, name = 'flowrate', title='Choose Out2 (LessDom) Flowrate Zone')
        of0_pt.select() #Selects the region of interest
        m.surf = define_fr(of0_pt, flowrate=flowrate_nondom*float(outflow2))
        comb4_pt = cc.RefinementSelection(m.surf, name = 'flowrate', title='Choose LessDom-Out0 Flowrate Zone')
        comb4_pt.select() #Selects the region of interest
        flowrate -= float(outflow2)*flowrate_nondom
        m.surf = define_fr(comb4_pt, flowrate=flowrate_nondom)

    #dom stuff
    if lab !='False':
        lab_pt = cc.RefinementSelection(m.surf, name = 'flowrate', title='Choose Labbe/Tant Flowrate Zone')
        lab_pt.select() #Selects the region of interest
        m.surf = define_fr(lab_pt, flowrate=lab)
        comb2_pt = cc.RefinementSelection(m.surf, name = 'flowrate', title='Choose StrSinus+SupSagSinus+Labbe/Tant Flowrate Zone')
        comb2_pt.select() #Selects the region of interest
        flowrate += float(lab)
        m.surf = define_fr(comb2_pt, flowrate=flowrate) #Adds data attribute to point array called 'SS
    if outflow1 !='False':
        of1_pt = cc.RefinementSelection(m.surf, name = 'flowrate', title='Choose Out1 Flowrate Zone')
        of1_pt.select() #Selects the region of interest
        m.surf = define_fr(of1_pt, flowrate=flowrate*float(outflow1))
        comb3_pt = cc.RefinementSelection(m.surf, name = 'flowrate', title='Choose StrSinus+SupSagSinus+Labbe-Out1 Flowrate Zone')
        comb3_pt.select() #Selects the region of interest
        flowrate -= float(outflow1)*flowrate
        m.surf = define_fr(comb3_pt, flowrate=flowrate) #Adds data attribute to point array called 'SS
    m.surf.point_data['flowrate'][m.surf.point_data['flowrate']==0]=flowrate #make sure flowrate is nonzero everywhere
    if fen !='False':
        fen_pt1 = cc.RefinementSelection(m.surf, name = 'FEN1', title='Choose 1st Fenest Flowrate Zone')
        fen_pt1.select() #Selects the region of interest
        fen_pt1.define_surface() 
        m.surf = fen_pt1.surf
        fen_pt2 = cc.RefinementSelection(m.surf, name = 'FEN2', title='Choose 2nd Fenest Flowrate Zone')
        fen_pt2.select() #Selects the region of interest
        fen_pt2.define_surface() #Adds boolean data attribute to point array called 'AG'
        m.surf = fen_pt2.surf
        #get average CSA ratio between branches:
        centerlines=pv.read(cent_file)
        tree3 = KDTree(centerlines.points)
        _, idx_a = tree3.query(m.surf.points[m.surf.point_data['FEN1']==1])
        _, idx_b = tree3.query(m.surf.points[m.surf.point_data['FEN2']==1])
        fen1 = np.mean(centerlines.point_data['CSA'][idx_a])
        fen2 = np.mean(centerlines.point_data['CSA'][idx_b])
        ratio1 = fen1/(fen1+fen2)
        ratio2 = fen2/(fen1+fen2)
        m.surf.point_data['flowrate'][m.surf.point_data['FEN1']==1]=m.surf.point_data['flowrate'][m.surf.point_data['FEN1']==1]*ratio1
        m.surf.point_data['flowrate'][m.surf.point_data['FEN2']==1]=m.surf.point_data['flowrate'][m.surf.point_data['FEN2']==1]*ratio2
    ref_pt = cc.RefinementSelection(m.surf, name = 'ref')
    ref_pt.select() #Selects the region of interest
    ref_pt.define_surface() #Adds boolean data attribute to point array called 'ref'
    m.surf = ref_pt.surf
    '''
