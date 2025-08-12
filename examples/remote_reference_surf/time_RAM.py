import sys
import os
import h5py
import csv
from pathlib import Path

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
        case_names = [ name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

    outfolder='DPQ_files/case_{}'.format(case)

    if not Path(outfolder).exists():
        Path(outfolder).mkdir(parents=True, exist_ok=True)

    file = str(outfolder)+'/case_{}_time_RAM.csv'.format(case)
    
    outfile = open(file, 'w', encoding='UTF8', newline='')
    writer = csv.writer(outfile)
    writer.writerow(['case_name','RAM (MB)', 'walltime (s)'])

    for case_name in case_names:
        logs_folder = Path(folder + '/'+case_name+'/logs')
        logs = logs_folder.glob('*restart*')
        wtime = 0
        ram=0
        for log in logs:
            file_log= open(log,"r")
            for line in file_log.readlines():
                if 'Total computing time' in line:
                    wtime += int(line.split('= ')[-1].split('.')[0])
                if 'Total memory used' in line:
                    ram = max(int(line.split(': ')[-1].split('.')[0]),ram)
            
        writer.writerow([case_name,ram, wtime])
        outfile.flush()
