#Script to compute Ve in parallel at each timestep written by ALKH

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
from mpi4py import MPI
import h5py
import get_dP

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mu = 0.0035
cosines = [-0.146846,-0.129927]
sines = [0.079431,0.015974]

def sample(domain, target):
    alg = vtk.vtkResampleWithDataSet()  # Construct the ResampleWithDataSet object
    alg.SetInputData(domain) 
    alg.SetSourceData(target)
    alg.SetPassCellArrays(False)
    alg.SetPassPointArrays(True)
    alg.SetPassFieldArrays(False)

    alg.SetMarkBlankPointsAndCells(True)
    alg.SetCategoricalData(False)
    return alg

'''
def integrate_data(data):
    alg = vtk.vtkIntegrateAttributes()
    alg.SetInputData(data)
    alg.Update()
    return pv.wrap(alg.GetOutput())
'''

def check_cells(domains):
    #n_dom = len(domains)
    for dom in domains:
        cellsize=dom.compute_cell_sizes().cell_data['Volume']
        neg=np.flatnonzero(cellsize<0)
        #print(len(neg), ' cell volumes are negative!')
        #node indices of each node (tets always go first)
        n3=5*neg+2
        n4=5*neg+3
        cells = dom.cells.copy()
        cells_n3=cells[n3].copy()
        cells_n4=cells[n4].copy()
        cells[n3]=cells_n4
        cells[n4]=cells_n3
        dom.cells=cells

        #now check:
        #check=len(np.flatnonzero(dom.compute_cell_sizes().cell_data['Volume']<0))
        #print(check, ' cell volumes are negative now')
    #volume = vtk.vtkTetra.ComputeVolume([0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1])

def derivs(dd, dom, d_name):
    '''
    if "high" in dd.case_name:
        dom_i=dom
    else:
        #t0=time.time()
        #update the sample alg
        alg = dd.samples[d_name] 
        alg.Update()
        #get the output from the sample alg
        dom_i = pv.wrap(alg.GetOutput())
        #t1=time.time()
        #if rank == 0:   
        #   print('Sampled in {} s'.format(t1-t0), flush=True)
    '''
    #sampling takes too long! Just assign original points
    dom.point_data['u']=dd.mesh.point_data['u'][dom.point_data['original_pts']]
    #print(dom1.point_data['u'])
    deriv = dom.compute_derivative(scalars = 'u', gradient=True) #use this for now since it is faster, but l8r use ZZ
    return deriv['gradient']

def Ve(dd, domains, idx):
    dd.mesh.point_data['u'] = dd(idx)
    n_dom = len(domains)
    Ve_dom = np.zeros((n_dom))
    Ve_bl = np.zeros((n_dom))
    #print('Beginning domain loop...')
    #t0=time.time()
    for j, dom in enumerate(domains):
        J = derivs(dd, dom, domains.get_block_name(j)).reshape(-1, 3, 3)
        #t1=time.time()
        #if rank == 0:
        #    print('Obtained derivatives in {} s'.format(t1-t0))
        D = J + np.transpose(J, axes=(0,2,1))
        dom.point_data['Ve'] = np.sum(np.sum(D*D, axis = 2), axis=1)
        #now just do the BL in this domain:
        dom.point_data['Ve_bl']=np.zeros((len(dom.points)))
        dom.point_data['Ve_bl'] = dom.point_data['Ve']*dom.point_data['bl']
        #t2=time.time()
        #need to check cell volumes are all positive:
        #volume_integrated = dom.integrate_data()
        #if volume_integrated['Ve'][0]<0:
        #    print(np.where(dom.compute_cell_sizes().cell_data['Volume']<0), 'tstep ',idx)
        dom=dom.point_data_to_cell_data()
        #max Ve in the slice:
        max_Ve = np.max(dom.cell_data['Ve'])
        max_Ve_bl = np.max(dom.cell_data['Ve_bl'])
        #t0=time.time()
        #if rank == 0:
        #    print('Integrated in {} s'.format(t0-t2))
        Ve_dom[j] = 0.5*mu*max_Ve#volume_integrated['Ve'][0]
        Ve_bl[j] = 0.5*mu*max_Ve_bl#volume_integrated['Ve_bl'][0]

    return Ve_dom, Ve_bl

if __name__=="__main__":
    time0 = time.time()
    project = os.environ["PROJECT"]
    if sys.argv[2]=='uul':
        case=sys.argv[1]+'_uul'
        folder = 'case_{}'.format(sys.argv[1])
        if sys.argv[1]=='A':
            case_names = ['PTSeg028_uul_0p8','PTSeg028_uul_0p64','PTSeg028_uul_0p512','PTSeg028_uul_0p4096']
        elif sys.argv[1]=='B':
            case_names = ['PTSeg043_uul_0p8','PTSeg043_uul_0p64','PTSeg043_uul_0p512','PTSeg043_uul_0p4096']
        elif sys.argv[1]=='C':
            case_names = ['PTSeg106_uul_0p8','PTSeg106_uul_0p64','PTSeg106_uul_0p512','PTSeg106_uul_0p4096']
        case_names = [name for name in case_names if '4096' in name]
    elif sys.argv[2] =='base':
        case=sys.argv[1]+'_base'
        folder = 'case_{}'.format(sys.argv[1])
        if sys.argv[1]=='A':
            case_names = ['PTSeg028_base_0p8','PTSeg028_base_0p64','PTSeg028_base_0p512','PTSeg028_base_0p4096'] #
        elif sys.argv[1]=='B':
            case_names = ['PTSeg043_base_0p8','PTSeg043_base_0p64','PTSeg043_base_0p512','PTSeg043_base_0p4096']
        elif sys.argv[1]=='C':
            case_names = ['PTSeg106_base_0p8','PTSeg106_base_0p64','PTSeg106_base_0p512','PTSeg106_base_0p4096']
        case_names = [name for name in case_names if '4096' in name]
    else:
        if sys.argv[1]=='Groccia':
            case='Groccia'
            folder = project+'/Swirl/swirl_cases'
            case_names = ['Groccia']
            outfolder = 'DPQ_files/case_Groccia'
        else:
            case = sys.argv[1]
            folder = project+'/mesh_rez/data/cases/case_{}'.format(sys.argv[1])
            case_names = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    outfolder='DPQ_files/case_{}'.format(sys.argv[1])
    '''
    
    '''
    if not Path(outfolder).exists():
        Path(outfolder).mkdir(parents=True, exist_ok=True)
    
    for case_name in case_names:
        if rank == 0:
            print(case_name)
        domains=pv.read('case_{}/{}_domains.vtm'.format(sys.argv[1],case_name))
        check_cells(domains)
        for dom in domains: #NOTE: this adjusts back to mm, but some of the domains are already in mm for some reason. Have to correct point data later by 10**3
            dom.points=dom.points*(10**-3)
        results = folder+'/'+ case_name + '/results/'
        results_folder = (results + os.listdir(results)[0]) #results folder eg. results/art_
        print(results_folder)
        dd = dataset.Dataset(results_folder, case_name=case_name)       
        dd = dd.assemble_mesh()
        if rank == 0:
            print('Beginning computations!', flush=True)
        if size>1:
            total_tsteps=len(dd.up_files)
            pieces = math.floor(total_tsteps/(size-1))
            last_piece = total_tsteps-pieces*(size-1) 
            if rank<size-1:
                tsteps = range(rank*pieces,(rank+1)*pieces)
            else:
                tsteps = range(rank*pieces,rank*pieces+last_piece)
        else:
            total_tsteps=len(dd.up_files)
            tsteps=range(total_tsteps)
            pieces=total_tsteps

        Q = np.zeros((len(tsteps),1))
        Ve_ = np.zeros((len(tsteps),len(domains)))
        Ve_bl = np.zeros((len(tsteps),len(domains)))
        max_time=23.5*60*60
        print_f = True
        outfile = Path(outfolder +'/DPQ_{}.h5'.format(case_name))
        if outfile.exists():
            restart_data = h5py.File(outfolder +'/DPQ_{}.h5'.format(case_name), 'r')
            Ve_root= np.array(restart_data['Ve'])
            Ve_bl_root= np.array(restart_data['Ve_bl'])
            Q_root = np.array(restart_data['Q'])
            if size>1:
                if rank<size-1:
                    Q = Q_root[rank*pieces,(rank+1)*pieces]
                    Ve_ = Ve_root[rank*pieces,(rank+1)*pieces, :]
                    Ve_bl = Ve_bl_root[rank*pieces,(rank+1)*pieces, :]
                else:
                    Q = Q_root[rank*pieces,rank*pieces+last_piece]
                    Ve_ = Ve_root[rank*pieces,rank*pieces+last_piece,:]
                    Ve_bl = Ve_bl_root[rank*pieces,rank*pieces+last_piece, :]
            else:
                Q=Q_root
                Ve_=Ve_root
                Ve_bl=Ve_bl_root
        
        if sys.argv[3]!='skip': #adjust this if you don't want to redo the calculation
            for i, idx in enumerate(tsteps):
                runtime = time.time()-time0
                if runtime<max_time:
                    #get flux at inlet at timestep
                    t = dd._get_time(dd.up_files[idx])
                    if Q[i] == 0: #will only be zero if not yet printed
                        Q[i] = (5.5833e-6)*(1+cosines[0]*math.cos(1*2*math.pi*t/0.915)+sines[0]*math.sin(1*2*math.pi*t/0.915)+cosines[1]*math.cos(2*2*math.pi*t/0.915)+sines[1]*math.sin(2*2*math.pi*t/0.915))
                        Ve_[i,:], Ve_bl[i,:] = Ve(dd, domains, idx)
                    if (i%10==0) and (rank == 0):
                        print('{}%  in {} mins'.format(round(100*i/len(tsteps)),(time.time()-time0)/60), flush=True)
                else:
                    #break out of loop and save state
                    print_f=False
                    break
            if rank == 0:
                if print_f==True: #We are done
                    print('Node is finished computations!')
                else:
                    print('Ended computations, out of time!')

            Ve_root = None
            Ve_bl_root = None
            Q_root = None
            if rank == 0:
                Ve_root = np.zeros((total_tsteps, len(domains)))
                Ve_bl_root = np.zeros((total_tsteps, len(domains)))
                Q_root = np.zeros((total_tsteps,1))
                for i in range(size):
                    if i == 0:
                        Ve_root[i*pieces:(i+1)*pieces , :] = Ve_
                        Ve_bl_root[i*pieces:(i+1)*pieces , :] = Ve_bl
                        Q_root[i*pieces:(i+1)*pieces] = Q
                    elif i<size-1:
                        temp_Ve=np.empty((pieces,len(domains)), dtype=np.float64)
                        temp_Ve_bl=np.empty((pieces,len(domains)), dtype=np.float64)
                        temp_Q=np.empty((pieces, 1), dtype=np.float64)
                        comm.Recv(temp_Ve, i, tag=0)
                        comm.Recv(temp_Ve_bl, i, tag=2)
                        comm.Recv(temp_Q, i, tag=1)
                        Ve_root[i*pieces:(i+1)*pieces , :] = temp_Ve
                        Ve_bl_root[i*pieces:(i+1)*pieces , :] = temp_Ve_bl
                        Q_root[i*pieces:(i+1)*pieces] = temp_Q
                    else:
                        temp_Ve=np.empty((last_piece,len(domains)), dtype=np.float64)
                        temp_Ve_bl=np.empty((last_piece,len(domains)), dtype=np.float64)
                        temp_Q=np.empty((last_piece, 1), dtype=np.float64)
                        comm.Recv(temp_Ve, i, tag=0)
                        comm.Recv(temp_Ve_bl, i, tag=2)
                        comm.Recv(temp_Q, i, tag=1)
                        Ve_root[i*pieces:i*pieces+last_piece, :] = temp_Ve
                        Ve_bl_root[i*pieces:i*pieces+last_piece, :] = temp_Ve_bl
                        Q_root[i*pieces:i*pieces+last_piece] = temp_Q
            else:
                comm.Send(Ve_, 0, tag=0)
                comm.Send(Ve_bl, 0, tag=2)
                comm.Send(Q, 0, tag=1)

            if rank ==0:
                print('Saving DPQ file to {}/DPQ_{}.h5'.format(outfolder,case_name))
                with h5py.File(outfolder +'/DPQ_{}.h5'.format(case_name), 'w') as f:
                        f.create_dataset(name='Ve', data=Ve_root)
                        f.create_dataset(name='Ve_bl', data=Ve_bl_root)
                        f.create_dataset(name='Q', data=Q_root)
        if rank==0:    
            if print_f == True: #computations are complete
                #load up the centerline points and assign attributes to them
                points = np.array([[x,0,0] for x in range(len(domains))])
                centerline = pv.PolyData(points) #points are just the ids of the domains
                centerline.point_data['Ve_avg']=np.max(Ve_root, axis=0)#np.sum(Ve_root, axis=0)/total_tsteps
                centerline.point_data['Ve_bl_avg']=np.max(Ve_bl_root, axis=0)#np.sum(Ve_bl_root, axis=0)/total_tsteps
                centerline.point_data['BL_prop_avg'] = np.max(Ve_bl_root, axis=0)/np.max(Ve_root, axis=0)#np.sum(Ve_bl_root, axis=0)/np.sum(Ve_root, axis=0)         
                d1 = int(math.floor(0.28*total_tsteps))
                d2 = int(math.floor(0.53*total_tsteps))
                centerline.point_data['Ve_decel'] = np.max(Ve_root[d1:d2, :], axis=0)#np.sum(Ve_root[d1:d2, :], axis=0)/(d2-d1)
                centerline.point_data['Ve_bl_decel'] = np.max(Ve_bl_root[d1:d2, :], axis=0)#np.sum(Ve_bl_root[d1:d2, :], axis=0)/(d2-d1)
                centerline.point_data['BL_prop_decel'] = np.max(Ve_bl_root[d1:d2, :], axis=0)/np.max(Ve_root[d1:d2, :], axis=0)#np.sum(Ve_bl_root[d1:d2, :], axis=0)/np.sum(Ve_root[d1:d2, :], axis=0)
                centerline.save('case_{}/{}_domain_centerline.vtp'.format(sys.argv[1], case_name))
                for j, dom in enumerate(domains):
                    dom.point_data['Ve_avg_t']=centerline.point_data['Ve_avg'][j]
                    dom.point_data['Ve_bl_avg_t']=centerline.point_data['Ve_bl_avg'][j]
                    dom.point_data['BL_prop_avg']=centerline.point_data['BL_prop_avg'][j]

                    dom.point_data['Ve_decel_t']=centerline.point_data['Ve_decel'][j]
                    dom.point_data['Ve_bl_decel_t']=centerline.point_data['Ve_bl_decel'][j]
                    dom.point_data['BL_prop_decel']=centerline.point_data['BL_prop_decel'][j]

                domains.save('case_{}/{}_domains.vtm'.format(case,case_name))