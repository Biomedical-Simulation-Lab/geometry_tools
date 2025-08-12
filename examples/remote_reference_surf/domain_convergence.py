import numpy as np
import sys
import os
from pathlib import Path
import pyvista as pv
import csv
from scipy.spatial import cKDTree as KDTree
import dataset
import matplotlib.pyplot as plt 
import math

size = 10
plt.rc('font', size=size) #controls default text size
plt.rc('axes', titlesize=size) #fontsize of the title
plt.rc('axes', labelsize=size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=size) #fontsize of the y tick labels
plt.rc('legend', fontsize=size) #fontsize of the legend

if __name__=="__main__":
    for case in ['A','B','C']:
        folder = 'case_{}'.format(case)
        if len(sys.argv)>1:
            if sys.argv[1]=='uul':
                case = case + '_uul'
                if case=='A_uul':
                    case_names = ['PTSeg028_uul_0p8','PTSeg028_uul_0p64','PTSeg028_uul_0p512','PTSeg028_uul_0p4096']
                elif case=='B_uul':
                    case_names = ['PTSeg043_uul_0p8','PTSeg043_uul_0p64','PTSeg043_uul_0p512','PTSeg043_uul_0p4096']
                elif case=='C_uul':
                    case_names = ['PTSeg106_uul_0p8','PTSeg106_uul_0p64','PTSeg106_uul_0p512','PTSeg106_uul_0p4096']
                leg = {'ultraultralow':'ExtraCoarse','0p8':'x0.8','0p64':'x0.64','0p512':'x0.512','0p4096':'x0.4096'}
                colors = {'ExtraCoarse':"red", 'x0.8':"orange", 'x0.64':"green", 'x0.512':"blue", 'x0.4096':"black"}
            elif sys.argv[1]=='base':
                case = case + '_base'
                if case=='A_base':
                    case_names = ['PTSeg028_base_0p8','PTSeg028_base_0p64','PTSeg028_base_0p512','PTSeg028_base_0p4096']
                elif case=='B_base':
                    case_names = ['PTSeg043_base_0p8','PTSeg043_base_0p64','PTSeg043_base_0p512','PTSeg043_base_0p4096']
                elif case=='C_base':
                    case_names = ['PTSeg106_base_0p8','PTSeg106_base_0p64','PTSeg106_base_0p512','PTSeg106_base_0p4096']
                leg = {'low':'Baseline','0p8':'x0.8','0p64':'x0.64','0p512':'x0.512','0p4096':'x0.4096'}
                colors = {'Baseline':"red", 'x0.8':"orange", 'x0.64':"green", 'x0.512':"blue", 'x0.4096':"black"}
            order0=[0,1,2,3]
        else:
            case = case
            project = os.environ["PROJECT"]
            folder = project+'/mesh_rez/data/cases/case_{}'.format(case)
            case_names = [ name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
            order0=[0,0,0,0,0]
            for idx, case_name in enumerate(case_names):
                if '_ultraultralow' in case_name:
                    order0[0]=idx
                elif '_ultralow' in case_name:
                    order0[1]=idx
                elif '_low' in case_name:
                    order0[2]=idx
                elif '_med' in case_name:
                    order0[3]=idx
                elif '_high' in case_name:
                    order0[4]=idx  
                leg = {'ultraultralow':'ExtraCoarse','ultralow':'Coarse','low':'Baseline','med':'Fine','high':'ExtraFine'}
                colors = {'ExtraCoarse':"red", 'Coarse':"orange", 'Baseline':"green", 'Fine':"blue", 'ExtraFine':"black"}
        plt.figure(figsize=(7, 4))
        plt.margins(x=0, y=0)
        #plt.yscale('log')
        
        centerline_file=list(Path('/project/s/steinman/ahaleyyy/mesh_rez/data/cases/case_{}'.format(case.split('_')[0])).glob('*_centerline_single.vtp'))[0]
        centerline = pv.read(centerline_file)
        outfolder='DPQ_files/case_{}'.format(case)

        if not Path(outfolder).exists():
            Path(outfolder).mkdir(parents=True, exist_ok=True)

        file = str(outfolder)+'/case_{}_domain_convergence.csv'.format(case)
        file_p = Path(file)
        
        outfile = open(file, 'w', encoding='UTF8', newline='')
        writer = csv.writer(outfile)

        def line_cm(pts=np.flip(centerline.points,axis=0)): #centerline goes from outlet to inlet
                seg_lens=np.zeros(len(pts))
                seg_lens[1:-1]=np.sqrt(np.sum(np.square(pts[0:-2,:]-pts[1:-1, :]),1)) #length of each segment
                disp=np.array([np.sum(seg_lens[0:ii]) for ii in range(len(seg_lens))]) #displacement in mm
                return disp/10 #displacement in cm
        x = line_cm()

        if len(sys.argv)>1:
            project = os.environ["PROJECT"]
            orig_folder = project+'/mesh_rez/data/cases/case_{}'.format(case.split('_')[0])
            if sys.argv[1]=='uul':
                orig_case = [name for name in os.listdir(orig_folder) if '_ultraultralow' in name]
            elif sys.argv[1]=='base':
                orig_case = [name for name in os.listdir(orig_folder) if '_low' in name]
            results_folder= orig_folder+'/'+orig_case[0]+'/results'
            dd = dataset.Dataset(results_folder+ os.listdir(results_folder)[0], case_name=orig_case[0])   
            dd = dd.assemble_mesh()
            dd.mesh.points=dd.mesh.points*10**3
            writer.writerow(orig_case)
            domains_file = 'case_{}/{}_domains.vtm'.format(case.split('_')[0], orig_case[0])
            domains = pv.read(domains_file)

            Ve_avg = []
            Ve_bl_avg = []
            dd.mesh.point_data['Ve']=np.zeros((dd.mesh.n_points,))
            dd.mesh.point_data['Ve_bl']=np.zeros((dd.mesh.n_points,))
            for dom in domains:
                dd.mesh.point_data['Ve'][dom.point_data['original_pts']]=dom.point_data['Ve_avg_t']

            tree = KDTree(dd.mesh.points)
            _, idx = tree.query(centerline.points)
            Ve_avg=dd.mesh.point_data['Ve'][idx]
            list1 = ['Ve_avg']+Ve_avg.tolist()
            writer.writerow(list1)
            label=leg[orig_case[0].split('_')[-1]]
            plt.plot(x,Ve_avg, color=colors[label],label=label, linewidth=0.5)

        for c in order0:
            print(case_names[c])
            list0 = [case_names[c]]
            results_folder= folder+'/'+case_names[c]+'/results'
            #print(results_folder)
            dd = dataset.Dataset(results_folder+ os.listdir(results_folder)[0], case_name=case_names[c])   
            dd = dd.assemble_mesh()
            dd.mesh.points=dd.mesh.points*10**3
            writer.writerow(list0)
            domains_file = 'case_{}/{}_domains.vtm'.format(case, case_names[c])
            domains = pv.read(domains_file)
            
            Ve_avg = []
            Ve_bl_avg = []
            dd.mesh.point_data['Ve']=np.zeros((dd.mesh.n_points,))
            dd.mesh.point_data['Ve_bl']=np.zeros((dd.mesh.n_points,))
            for dom in domains:
                dd.mesh.point_data['Ve'][dom.point_data['original_pts']]=dom.point_data['Ve_avg_t']

            tree = KDTree(dd.mesh.points)
            _, idx = tree.query(centerline.points)
            Ve_avg=dd.mesh.point_data['Ve'][idx]
            list1 = ['Ve_avg']+Ve_avg.tolist()
            writer.writerow(list1)
            label=leg[case_names[c].split('_')[-1]]
            plt.ylim((0,math.ceil(np.max(Ve_avg))+5e5))
            plt.plot(x,Ve_avg, color=colors[label],label=label, linewidth=0.5)
        #plt.xlabel('Distance along centerline (cm)', labelpad=-1)
        #plt.ylabel('$max(0.5\mu D:D)$', labelpad=-4)
        plt.legend()
        plt.savefig(outfolder+'/case_{}_domainconv_notlog.png'.format(case))
        outfile.close()
