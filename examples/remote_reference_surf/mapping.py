import numpy as np 
import h5py 
import pyvista as pv
import math
import dataset
from pathlib import Path
import sys
import os
from scipy.spatial import cKDTree as KDTree

def map_Ve(domain_files, case, case_name):
    #folder = '/project/s/steinman/ahaleyyy/mesh_rez/data/cases/case_{}'.format(case)
    folder = '/project/s/steinman/ahaleyyy/Swirl/swirl_cases'
    results = folder+'/'+ case_name + '/results/'
    results_folder = Path((results + os.listdir(results)[0])) #results folder eg. results/art_
    dd = dataset.Dataset(results_folder, case_name=case_name)   
    dd = dd.assemble_mesh()
    domains = pv.read(domain_files)
    dd.mesh.point_data['Ve']=np.zeros((len(dd.mesh.points,)))
    dd.mesh.point_data['Ve_count']=np.zeros((len(dd.mesh.points,)))
    dd.mesh.point_data['Ve_averaged']=np.zeros((len(dd.mesh.points,)))
    for dom in domains:
        #sample = dd.mesh.sample(dom) #sample the domain values to the mesh
        #take the maximum Ve value at each point
        dd.mesh.point_data['Ve'][dom.point_data['original_pts']]=np.maximum(dom.point_data['Ve_avg_t'],dd.mesh.point_data['Ve'][dom.point_data['original_pts']])
        #print(np.sum(sample.point_data['vtkValidPointMask'])/len(dom.points))
    mask=dd.mesh.point_data['Ve']>0
    #dd.mesh.point_data['Ve_averaged'][mask]=dd.mesh.point_data['Ve'][mask]/dd.mesh.point_data['Ve_count'][mask] 
    #selected=dd.mesh.threshold(scalars='Ve_count', value=0.5, invert=False)
    points=dd.mesh.points[mask]
    dd.mesh.save('case_{}/{}_ref.vtu'.format(case,case_name))
    tree=KDTree(points)
    _, pids = tree.query(dd.surf.points)
    dd.surf.point_data['Ve'] = dd.mesh.point_data['Ve'][mask][pids]#will have to smooth this data once we get it offline
    dd.surf.point_data['logVe'] = np.log10(dd.surf.point_data['Ve'])
    dd.surf.point_data['lnVe'] = np.log(dd.surf.point_data['Ve'])
    dd.surf.points *=10**3 #get back into mm for convolving with other datasets
    dd.surf.save('case_{}/{}_ref.vtp'.format(case,case_name))

if __name__=="__main__":
    
    #cent_file='case_A/PTSeg028_centerline_re.vtp'
    #map_Ve(domain_files='case_A/case_028_ultraultralow_domains.vtm', case = 'A', case_name='case_028_ultraultralow')
    #map_Ve(domain_files='case_A/case_028_ultralow_domains.vtm', case = 'A', case_name='case_028_ultralow')
    #map_Ve(domain_files='case_A/case_028_low_domains.vtm', case = 'A', case_name='case_028_low')
    #map_Ve(domain_files='case_A/case_028_med_domains.vtm', case = 'A', case_name='case_028_med')
    #map_Ve(domain_files='case_A/case_028_high_domains.vtm', case = 'A', case_name='case_028_high')
    
    #cent_file='case_C/PTSeg106_centerline_re.vtp'
    #map_Ve(domain_files='case_C/case_106_ultraultralow_domains.vtm', case = 'C', case_name='case_106_ultraultralow')
    #map_Ve(domain_files='case_C/case_106_ultralow_domains.vtm', case = 'C', case_name='case_106_ultralow')
    #map_Ve(domain_files='case_C/case_106_low_domains.vtm', case = 'C', case_name='case_106_low')
    #map_Ve(domain_files='case_C/case_106_med_domains.vtm', case = 'C', case_name='case_106_med')    
    #map_Ve(domain_files='case_C/case_106_high_domains.vtm', case = 'C', case_name='case_106_high')
    
    #cent_file='case_B/PTSeg043_centerline_re.vtp'
    #map_Ve(domain_files='case_B/case_043_ultraultralow_domains.vtm', case = 'B', case_name='case_043_ultraultralow')
    #map_Ve(domain_files='case_B/case_043_ultralow_domains.vtm', case = 'B', case_name='case_043_ultralow')
    #map_Ve(domain_files='case_B/case_043_low_domains.vtm', case = 'B', case_name='case_043_low')
    #map_Ve(domain_files='case_B/case_043_med_domains.vtm', case = 'B', case_name='case_043_med')
    #map_Ve(domain_files='case_B/case_043_high_domains.vtm', case = 'B', case_name='case_043_high')
    project = os.environ["PROJECT"]
    cent_file=project+'/Swirl/swirl_cases/Groccia/Groccia_cl_centerline.vtp'
    map_Ve(domain_files='case_Groccia/Groccia_domains.vtm', case = 'Groccia', case_name='Groccia')
    