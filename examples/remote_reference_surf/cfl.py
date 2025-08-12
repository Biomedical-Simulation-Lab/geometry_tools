import numpy as np
import sys
import os
import dataset
from pathlib import Path
import h5py
import pyvista as pv
import vtk
import h5py
import csv
import math
import multiprocessing as mp

def get_lmnt_dx(dd):
    dx=[]
    for ndx in range(dd.mesh.n_cells):
        element = dd.mesh.get_cell(ndx)
        if element.dimension<3:
            continue
        else:
            dmax = np.array([element.bounds[1],element.bounds[3],element.bounds[5]])
            dmin = np.array([element.bounds[0],element.bounds[2],element.bounds[4]])
            d = dmax-dmin
            dx.append(d) #max-min
    dd.dx = np.asarray(dx) #m

def get_dt(dd, id_list,output):
    dt_min = 100000000
    for i in id_list:
        dd.mesh.point_data['v'] = dd(i,'u') #m/s
        dd.mesh=dd.mesh.point_data_to_cell_data()
        dt_new = np.abs(dd.dx/dd.mesh.cell_data['v'])
        #print(dt_new.shape)
        dt_new_min = np.min(dt_new)
        if dt_new_min<dt_min:
            dt_min=dt_new_min
    output.put(dt_min)

if __name__=="__main__":
    if len(sys.argv)>2:
        if sys.argv[2]=='uul':
            case=sys.argv[1]+'_uul'
            folder = 'case_{}'.format(sys.argv[1])
            if sys.argv[1]=='A':
                case_names = ['PTSeg028_uul_0p8','PTSeg028_uul_0p64','PTSeg028_uul_0p512','PTSeg028_uul_0p4096']
            elif sys.argv[1]=='B':
                case_names = ['PTSeg043_uul_0p8','PTSeg043_uul_0p64','PTSeg043_uul_0p512','PTSeg043_uul_0p4096']
            elif sys.argv[1]=='C':
                case_names = ['PTSeg106_uul_0p8','PTSeg106_uul_0p64','PTSeg106_uul_0p512','PTSeg106_uul_0p4096']
        elif sys.argv[2]=='base':
            case=sys.argv[1]+'_base'
            folder = 'case_{}'.format(sys.argv[1])
            if sys.argv[1]=='A':
                case_names = ['PTSeg028_base_0p8','PTSeg028_base_0p64','PTSeg028_base_0p512','PTSeg028_base_0p4096']
            elif sys.argv[1]=='B':
                case_names = ['PTSeg043_base_0p8','PTSeg043_base_0p64','PTSeg043_base_0p512','PTSeg043_base_0p4096']
            elif sys.argv[1]=='C':
                case_names = ['PTSeg106_base_0p8','PTSeg106_base_0p64','PTSeg106_base_0p512','PTSeg106_base_0p4096']
    else:
        case = sys.argv[1]
        folder = '../mesh_rez/cases/case_{}'.format(sys.argv[1])
        case_names = [ name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name)) ]
    
    outfolder='DPQ_files/case_{}'.format(case)

    if not Path(outfolder).exists():
        Path(outfolder).mkdir(parents=True, exist_ok=True)

    if len(sys.argv)>2:
        file = str(outfolder)+'/case_{}_CFL.csv'.format(case)
    else:
        file = str(outfolder)+'/case_{}_initial_CFL.csv'.format(sys.argv[1])
    
    outfile = open(file, 'w', encoding='UTF8', newline='')
    writer = csv.writer(outfile)
    writer.writerow(['case_name','tsteps', 'req_tsteps'])

    for case_name in case_names:
        results = folder+'/'+ case_name + '/results/'
        results_folder = Path((results + os.listdir(results)[0])) #results folder eg. results/art_
        if len(sys.argv)>2:
            seg_name = case_name
        else:    
            seg_name = 'PTSeg'+case_name.split('_')[1]+'_'+case_name.split('_')[-1]
        print(seg_name)
        dd = dataset.Dataset(results_folder, case_name=case_name)
        dd = dd.assemble_mesh()
        #serial computation:
        get_lmnt_dx(dd)

        #break up the computation:
        nps = 40
        num = math.floor(len(dd.up_files)/(nps-1))
        up_ids = []
        for i in range(nps-1):
            up_ids.append(range(i*num,(i+1)*num-1))
        up_ids.append(range(nps*num,len(dd.up_files))) #the remaining ids
        
        output = mp.Queue()
        processes = [mp.Process(target=get_dt, args=(dd, up_ids[x], output)) for x in range(nps)]
        # Run processes
        for p in processes:
            p.start()

        dt_min=10000000000
        # Exit the completed processes
        for p in processes:
            p.join()
            dt_min = min(output.get(), dt_min)
        req_tsteps=round(0.915/dt_min)
        file_sh= open(folder+'/'+ case_name + '/'+ seg_name+'.sh',"r")
        for line in file_sh.readlines():
            if 'timesteps' in line:
                tsteps = int(line.split('=')[-1])
        #print(case_name,tsteps, dt_min, req_tsteps)
        writer.writerow([case_name,tsteps, dt_min, req_tsteps])
        outfile.flush()
    outfile.close()



