import numpy as np 
import h5py 
import pyvista as pv
import math
from pathlib import Path
import sys
from scipy.spatial import cKDTree as KDTree
import os

def get_bls(mesh):
    cellsize=mesh.compute_cell_sizes().cell_data['Volume']
    neg=np.flatnonzero(cellsize<0)
    print(len(neg))
    ptids=[]
    for i in range(len(neg)):
        ptids += mesh.get_cell(neg[i]).point_ids
    return np.unique(np.array(ptids))

def check_cells(mesh):
    cellsize=mesh.compute_cell_sizes().cell_data['Volume']
    neg=np.flatnonzero(cellsize<0)
    #now fix it
    n3=5*neg+2
    n4=5*neg+3
    cells = mesh.cells.copy()
    cells_n3=cells[n3].copy()
    cells_n4=cells[n4].copy()
    cells[n3]=cells_n4
    cells[n4]=cells_n3
    mesh.cells=cells

def slice(cent_file, heat, mesh_file, domain_files='domains.vtm', case = 'A'):
    mesh = pv.read(mesh_file)
    heat = pv.read(heat)
    cent = pv.read(cent_file)
    #tree = KDTree(mesh.points)

    if not Path('case_{}'.format(case)).exists():
        Path('case_{}'.format(case)).mkdir(parents=True, exist_ok=True)
    case_domains = 'case_{}/case_{}_domains.vtm'.format(case,case)
    if not Path(domain_files).exists():
        domains=pv.MultiBlock()
        #if not Path(case_domains).exists():
        #ds = pv.MultiBlock()
        #all_pts=[]
        sampled = cent.interpolate(heat, radius=1)
        heat.cell_data['original_cells']=np.array(range(heat.n_cells))
        mesh.point_data['original_pts']=np.array(range(mesh.n_points))
        heat.point_data['original_pts']=mesh.point_data['original_pts']
        heat.cell_data['taken']=np.zeros((heat.n_cells,))
        #go by each cell in the dataset:
        for cid in range(cent.n_cells):
            cell = cent.get_cell(cid)
            #points = cell.points
            pt_ids = np.concatenate((cell.point_ids[::2],np.array(cell.point_ids[-1]).reshape(1,))) #point ids of the centerline cell we want to take
            pts = cent.points[pt_ids] #take every 3rd point plus the end point
            #all_pts.append(pts)  
            T_values = sampled['T'][pt_ids]
            print(T_values)
            n_slices = len(pt_ids)
            for ndx in range(n_slices-1):
                T_max=max(T_values[ndx],T_values[ndx+1])
                T_min=min(T_values[ndx],T_values[ndx+1])
                if T_min==T_max:
                    continue
                if abs(T_max-T_min)>20:
                    continue
                newheat = heat.threshold(scalars='taken', invert=True, value=0.5)
                domain = newheat.threshold(scalars='T', invert=False, value=[T_min,T_max]) #values above T_min
                ctr_pt = pts[ndx]+(pts[ndx+1]-pts[ndx])/2
                #make sure every domain has more than one cell...
                if domain.n_cells<=1:
                    continue
                #make sure we are always getting the correct slice
                split = domain.split_bodies()
                if len(split)>1:
                    cm = np.zeros((len(split),3))
                    for i in range(len(split)):
                        cm[i, :] = split[i].center #center of the bounding box
                    tree2 = KDTree(cm)
                    _, j = tree2.query(ctr_pt) #closest center to the center point
                    d=split[j]
                else:
                    d=domain
                heat.cell_data['taken'][d.cell_data['original_cells']]=1
                domains.append(d)
        '''
            ds.save(case_domains)
        else:
            ds=pv.read(case_domains)

        for d in ds:
            d_surf=d.extract_surface()
            select=mesh.select_enclosed_points(d_surf)
            inside = select.threshold(0.5,scalars='SelectedPoints')
            domains.append(inside)
        '''
        #new_cl = pv.PolyData(points)
        #new_cl.save('case_{}/domain_centerline_{}.vtp'.format(case, case))
        domains.save(domain_files)
    else:
        domains=pv.read(domain_files)
    print('Obtained domains!')
    
    #get boundary layers marked by cells that have negative volume
    bln=get_bls(mesh)
    mesh.point_data['bl']=np.zeros(len(mesh.points))
    mesh.point_data['bl'][bln]=1
    for i, dom in enumerate(domains):
        dom.point_data['bl']=mesh.point_data['bl'][dom.point_data['original_pts']]
        #print('{}/{} points are bl'.format(int(np.sum(dom.point_data['bl'])),len(dom.points)))

        check_cells(dom)
        integ_d = dom.integrate_data()
        dom.point_data['vol']=integ_d['Volume'][0]

    print('Found boundary layer points! Saving...')
    domains.save(domain_files)


if __name__=="__main__":

    project = os.environ["PROJECT"]

    if sys.argv[1]=='A':
        #heat0 = 'case_A/case_028_ultraultralow_heat000000.vtu'
        #heat1 = 'case_A/case_028_ultralow_heat000000.vtu'
        #heat2 = 'case_A/case_028_low_heat000000.vtu'
        #heat3 = 'case_A/case_028_med_heat000000.vtu'
        #heat4 = 'case_A/case_028_high_heat000000.vtu'
        #slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat0, mesh_file=project+'/mesh_rez/data/cases/case_A/case_028_ultraultralow/mesh/PTSeg028_ultraultralow.vtu', domain_files='case_A/case_028_ultraultralow_domains.vtm', case = 'A')
        #slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat1, mesh_file=project+'/mesh_rez/data/cases/case_A/case_028_ultralow/mesh/PTSeg028_ultralow.vtu', domain_files='case_A/case_028_ultralow_domains.vtm', case = 'A')
        #slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat2, mesh_file=project+'/mesh_rez/data/cases/case_A/case_028_low/mesh/PTSeg028_low.vtu', domain_files='case_A/case_028_low_domains.vtm', case = 'A')
        #slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat3, mesh_file=project+'/mesh_rez/data/cases/case_A/case_028_med/mesh/PTSeg028_med.vtu', domain_files='case_A/case_028_med_domains.vtm', case = 'A')
        #slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat4, mesh_file=project+'/mesh_rez/data/cases/case_A/case_028_high/mesh/PTSeg028_high.vtu', domain_files='case_A/case_028_high_domains.vtm', case = 'A')
        if len(sys.argv)>2:
            if sys.argv[2]=='uul':
                #heat0 = 'case_A/PTSeg028_uul_0p8_heat000000.vtu'
                #heat1 = 'case_A/PTSeg028_uul_0p64_heat000000.vtu'
                #heat2 = 'case_A/PTSeg028_uul_0p512_heat000000.vtu'
                heat3 = 'case_A/PTSeg028_uul_0p4096_heat000000.vtu'
                #slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat0, mesh_file='case_A/PTSeg028_uul_0p8/mesh/PTSeg028_uul_0p8.vtu', domain_files='case_A/PTSeg028_uul_0p8_domains.vtm', case = 'A')
                #slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat1, mesh_file='case_A/PTSeg028_uul_0p64/mesh/PTSeg028_uul_0p64.vtu', domain_files='case_A/PTSeg028_uul_0p64_domains.vtm', case = 'A')
                #slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat2, mesh_file='case_A/PTSeg028_uul_0p512/mesh/PTSeg028_uul_0p512.vtu', domain_files='case_A/PTSeg028_uul_0p512_domains.vtm', case = 'A')
                slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat3, mesh_file='case_A/PTSeg028_uul_0p4096/mesh/PTSeg028_uul_0p4096.vtu', domain_files='case_A/PTSeg028_uul_0p4096_domains.vtm', case = 'A')
            elif sys.argv[2]=='base':                    
                heat0 = 'case_A/PTSeg028_base_0p8_heat000000.vtu'
                heat1 = 'case_A/PTSeg028_base_0p64_heat000000.vtu'
                #heat2 = 'case_A/PTSeg028_base_0p512_heat000000.vtu'
                heat3 = 'case_A/PTSeg028_base_0p4096_heat000000.vtu'
                slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat0, mesh_file='case_A/PTSeg028_base_0p8/mesh/PTSeg028_base_0p8.vtu', domain_files='case_A/PTSeg028_base_0p8_domains.vtm', case = 'A')
                slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat1, mesh_file='case_A/PTSeg028_base_0p64/mesh/PTSeg028_base_0p64.vtu', domain_files='case_A/PTSeg028_base_0p64_domains.vtm', case = 'A')
                #slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat2, mesh_file='case_A/PTSeg028_base_0p512/mesh/PTSeg028_base_0p512.vtu', domain_files='case_A/PTSeg028_base_0p512_domains.vtm', case = 'A')
                slice(cent_file='case_A/PTSeg028_centerline_re.vtp', heat=heat3, mesh_file='case_A/PTSeg028_base_0p4096/mesh/PTSeg028_base_0p4096.vtu', domain_files='case_A/PTSeg028_base_0p4096_domains.vtm', case = 'A')

    if sys.argv[1]=='C':
        #heat0 = 'case_C/case_106_ultraultralow_heat000000.vtu'
        #heat1 = 'case_C/case_106_ultralow_heat000000.vtu'
        #heat2 = 'case_C/case_106_low_heat000000.vtu'
        #heat3 = 'case_C/case_106_med_heat000000.vtu'
        #heat4 = 'case_C/case_106_high_heat000000.vtu'
        #slice(cent_file='case_C/PTSeg106_centerline_re.vtp',heat=heat0, mesh_file=project+'/mesh_rez/data/cases/case_C/case_106_ultraultralow/mesh/PTSeg106_ultraultralow.vtu', domain_files='case_C/case_106_ultraultralow_domains.vtm', case = 'C')
        #slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat1, mesh_file=project+'/mesh_rez/data/cases/case_C/case_106_ultralow/mesh/PTSeg106_ultralow.vtu', domain_files='case_C/case_106_ultralow_domains.vtm', case = 'C')
        #slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat2, mesh_file=project+'/mesh_rez/data/cases/case_C/case_106_low/mesh/PTSeg106_low.vtu', domain_files='case_C/case_106_low_domains.vtm', case = 'C')
        #slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat3, mesh_file=project+'/mesh_rez/data/cases/case_C/case_106_med/mesh/PTSeg106_med.vtu', domain_files='case_C/case_106_med_domains.vtm', case = 'C')    
        #slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat4, mesh_file=project+'/mesh_rez/data/cases/case_C/case_106_high/mesh/PTSeg106_high.vtu', domain_files='case_C/case_106_high_domains.vtm', case = 'C')
        if len(sys.argv)>2:
            if sys.argv[2]=='uul':
                #heat0 = 'case_C/PTSeg106_uul_0p8_heat000000.vtu'
                #heat1 = 'case_C/PTSeg106_uul_0p64_heat000000.vtu'
                #heat2 = 'case_C/PTSeg106_uul_0p512_heat000000.vtu'
                heat3 = 'case_C/PTSeg106_uul_0p4096_heat000000.vtu'
                #slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat0, mesh_file='case_C/PTSeg106_uul_0p8/mesh/PTSeg106_uul_0p8.vtu', domain_files='case_C/PTSeg106_uul_0p8_domains.vtm', case = 'C')
                #slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat1, mesh_file='case_C/PTSeg106_uul_0p64/mesh/PTSeg106_uul_0p64.vtu', domain_files='case_C/PTSeg106_uul_0p64_domains.vtm', case = 'C')
                #slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat2, mesh_file='case_C/PTSeg106_uul_0p512/mesh/PTSeg106_uul_0p512.vtu', domain_files='case_C/PTSeg106_uul_0p512_domains.vtm', case = 'C')
                slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat3, mesh_file='case_C/PTSeg106_uul_0p4096/mesh/PTSeg106_uul_0p4096.vtu', domain_files='case_C/PTSeg106_uul_0p4096_domains.vtm', case = 'C')
            elif sys.argv[2]=='base':    
                #heat0 = 'case_C/PTSeg106_base_0p8_heat000000.vtu'
                #heat1 = 'case_C/PTSeg106_base_0p64_heat000000.vtu'
                #heat2 = 'case_C/PTSeg106_base_0p512_heat000000.vtu'
                heat3 = 'case_C/PTSeg106_base_0p4096_heat000000.vtu'
                #slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat0, mesh_file='case_C/PTSeg106_base_0p8/mesh/PTSeg106_base_0p8.vtu', domain_files='case_C/PTSeg106_base_0p8_domains.vtm', case = 'C')
                #slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat1, mesh_file='case_C/PTSeg106_base_0p64/mesh/PTSeg106_base_0p64.vtu', domain_files='case_C/PTSeg106_base_0p64_domains.vtm', case = 'C')
                #slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat2, mesh_file='case_C/PTSeg106_base_0p512/mesh/PTSeg106_base_0p512.vtu', domain_files='case_C/PTSeg106_base_0p512_domains.vtm', case = 'C')
                slice(cent_file='case_C/PTSeg106_centerline_re.vtp', heat=heat3, mesh_file='case_C/PTSeg106_base_0p4096/mesh/PTSeg106_base_0p4096.vtu', domain_files='case_C/PTSeg106_base_0p4096_domains.vtm', case = 'C')

    if sys.argv[1]=='B':    
        #heat0 = 'case_B/case_043_ultraultralow_heat000000.vtu'
        #heat1 = 'case_B/case_043_ultralow_heat000000.vtu'
        #heat2 = 'case_B/case_043_low_heat000000.vtu'
        #heat3 = 'case_B/case_043_med_heat000000.vtu'
        #heat4 = 'case_B/case_043_high_heat000000.vtu'
        #cent_file='case_B/PTSeg043_centerline_re.vtp'
        #slice(cent_file=cent_file, heat=heat0, mesh_file=project+'/mesh_rez/data/cases/case_B/case_043_ultraultralow/mesh/PTSeg043_ultraultralow.vtu', domain_files='case_B/case_043_ultraultralow_domains.vtm', case = 'B')
        #slice(cent_file=cent_file, heat=heat1, mesh_file=project+'/mesh_rez/data/cases/case_B/case_043_ultralow/mesh/PTSeg043_ultralow.vtu', domain_files='case_B/case_043_ultralow_domains.vtm', case = 'B')
        #slice(cent_file=cent_file, heat=heat2, mesh_file=project+'/mesh_rez/data/cases/case_B/case_043_low/mesh/PTSeg043_low.vtu', domain_files='case_B/case_043_low_domains.vtm', case = 'B')
        #slice(cent_file=cent_file, heat=heat3, mesh_file=project+'/mesh_rez/data/cases/case_B/case_043_med/mesh/PTSeg043_med.vtu', domain_files='case_B/case_043_med_domains.vtm', case = 'B')
        #slice(cent_file=cent_file, heat=heat4, mesh_file=project+'/mesh_rez/data/cases/case_B/case_043_high/mesh/PTSeg043_high.vtu', domain_files='case_B/case_043_high_domains.vtm', case = 'B')
        if len(sys.argv)>2:
            if sys.argv[2]=='uul':
                #heat0 = 'case_B/PTSeg043_uul_0p8_heat000000.vtu'
                #heat1 = 'case_B/PTSeg043_uul_0p64_heat000000.vtu'
                #heat2 = 'case_B/PTSeg043_uul_0p512_heat000000.vtu'
                heat3 = 'case_B/PTSeg043_uul_0p4096_heat000000.vtu'
                #slice(cent_file='case_B/PTSeg043_centerline_re.vtp', heat=heat0, mesh_file='case_B/PTSeg043_uul_0p8/mesh/PTSeg043_uul_0p8.vtu', domain_files='case_B/PTSeg043_uul_0p8_domains.vtm', case = 'B')
                #slice(cent_file='case_B/PTSeg043_centerline_re.vtp', heat=heat1, mesh_file='case_B/PTSeg043_uul_0p64/mesh/PTSeg043_uul_0p64.vtu', domain_files='case_B/PTSeg043_uul_0p64_domains.vtm', case = 'B')
                #slice(cent_file='case_B/PTSeg043_centerline_re.vtp', heat=heat2, mesh_file='case_B/PTSeg043_uul_0p512/mesh/PTSeg043_uul_0p512.vtu', domain_files='case_B/PTSeg043_uul_0p512_domains.vtm', case = 'B')
                slice(cent_file='case_B/PTSeg043_centerline_re.vtp', heat=heat3, mesh_file='case_B/PTSeg043_uul_0p4096/mesh/PTSeg043_uul_0p4096.vtu', domain_files='case_B/PTSeg043_uul_0p4096_domains.vtm', case = 'B')
            elif sys.argv[2]=='base':
                #heat0 = 'case_B/PTSeg043_base_0p8_heat000000.vtu'
                #heat1 = 'case_B/PTSeg043_base_0p64_heat000000.vtu'
                #heat2 = 'case_B/PTSeg043_base_0p512_heat000000.vtu'
                heat3 = 'case_B/PTSeg043_base_0p4096_heat000000.vtu'
                #slice(cent_file='case_B/PTSeg043_centerline_re.vtp', heat=heat0, mesh_file='case_B/PTSeg043_base_0p8/mesh/PTSeg043_base_0p8.vtu', domain_files='case_B/PTSeg043_base_0p8_domains.vtm', case = 'B')
                #slice(cent_file='case_B/PTSeg043_centerline_re.vtp', heat=heat1, mesh_file='case_B/PTSeg043_base_0p64/mesh/PTSeg043_base_0p64.vtu', domain_files='case_B/PTSeg043_base_0p64_domains.vtm', case = 'B')
                #slice(cent_file='case_B/PTSeg043_centerline_re.vtp', heat=heat2, mesh_file='case_B/PTSeg043_base_0p512/mesh/PTSeg043_base_0p512.vtu', domain_files='case_B/PTSeg043_base_0p512_domains.vtm', case = 'B')
                slice(cent_file='case_B/PTSeg043_centerline_re.vtp', heat=heat3, mesh_file='case_B/PTSeg043_base_0p4096/mesh/PTSeg043_base_0p4096.vtu', domain_files='case_B/PTSeg043_base_0p4096_domains.vtm', case = 'B')

    if sys.argv[1]=='Groccia':
         heat = 'case_Groccia/Groccia_heat000000.vtu'
         slice(cent_file=project+'/Swirl/swirl_cases/Groccia/Groccia_cl_centerline.vtp', heat=heat, mesh_file=project+'/Swirl/swirl_cases/Groccia/mesh/Groccia.vtu', domain_files='case_Groccia/Groccia_domains.vtm', case = 'Groccia')