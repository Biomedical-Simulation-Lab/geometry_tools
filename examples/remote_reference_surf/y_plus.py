import numpy as np
import math
import time
import sys
import os
import dataset
from pathlib import Path
import h5py
import pyvista as pv
import vtk
import h5py
from scipy.spatial import cKDTree as KDTree


if __name__=="__main__":
    if sys.argv[2]=='uul':
        case=sys.argv[1]+'_uul'
        folder = 'case_{}'.format(sys.argv[1])
        if sys.argv[1]=='A':
            case_names = ['PTSeg028_uul_0p8','PTSeg028_uul_0p64','PTSeg028_uul_0p512','PTSeg028_uul_0p4096']
        elif sys.argv[1]=='B':
            case_names = ['PTSeg043_uul_0p8','PTSeg043_uul_0p64','PTSeg043_uul_0p512','PTSeg043_uul_0p4096']
        elif sys.argv[1]=='C':
            case_names = ['PTSeg106_uul_0p8','PTSeg106_uul_0p64','PTSeg106_uul_0p512','PTSeg106_uul_0p4096']
    elif sys.argv[2] =='base':
        case=sys.argv[1]+'_base'
        folder = 'case_{}'.format(sys.argv[1])
        if sys.argv[1]=='A':
            case_names = ['PTSeg028_base_0p4096']#['PTSeg028_base_0p8','PTSeg028_base_0p64','PTSeg028_base_0p512'
        elif sys.argv[1]=='B':
            case_names = ['PTSeg043_base_0p8','PTSeg043_base_0p64','PTSeg043_base_0p512','PTSeg043_base_0p4096']
        elif sys.argv[1]=='C':
            case_names = ['PTSeg106_base_0p8','PTSeg106_base_0p64','PTSeg106_base_0p512','PTSeg106_base_0p4096']
    else:
        case=sys.argv[1]
        project = os.environ["PROJECT"]
        folder = project+'/mesh_rez/data/cases/case_{}'.format(case)
        case_names = [ name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

    outfolder='DPQ_files/case_{}'.format(case)

    if not Path(outfolder).exists():
        Path(outfolder).mkdir(parents=True, exist_ok=True)
    
    for case_name in case_names:
        results = folder+'/'+ case_name + '/results/'
        results_folder = Path((results + os.listdir(results)[0])) #results folder eg. results/art_

        dd = dataset.Dataset(results_folder, case_name=case_name)
        dd = dd.assemble_mesh()

        wssfolder= (results_folder / 'wss_files')
        wss_files= sorted(wssfolder.glob('*_curcyc_*wss.h5'), key=dd._get_ts)[::3] #for the new refinements
        
        #get closest points on internal mesh to surface:
        tree = KDTree(dd.mesh.points)
        dist , ndx = tree.query(dd.surf.points, k=2)
        nu = 0.0035/1057
        ds = np.array(dist[:,1]).flatten() #take only the second-closest
        y_max = np.zeros((len(dd.surf.points),))
        y_avg = np.zeros((len(dd.surf.points),))
        tsteps=len(wss_files)
        for file in wss_files:
            wss = dd(file=file, array='wss')
            u_tau = np.sqrt(np.linalg.norm(wss, axis=1)/1057)
            y_plus = u_tau*ds/nu
            y_max = np.maximum(y_max,y_plus)
            y_avg += y_plus/tsteps

        dd.surf.point_data['y_max']=y_max
        dd.surf.point_data['y_avg']=y_avg
        dd.surf.point_data['dist']=np.array(dist[:,1]).flatten()

        dd.surf.save(outfolder + '/' + case_name + '_yplus.vtp')
        print('Saved {}_yplus.vtp with maximum = '.format(case_name), max(dd.surf.point_data['y_max']))
