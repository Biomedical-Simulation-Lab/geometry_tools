#!/bin/bash

########SBATCH -A ctb-steinman
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
######SBATCH --time=5:59:00
#SBATCH --time=1:00:00
#SBATCH -p debug
#SBATCH --job-name yp_viz
#SBATCH --output=yp_viz_%j.txt

export MPLCONFIGDIR=/scratch/s/steinman/ahaleyyy/.config/mpl
export PYVISTA_USERDATA_PATH=/scratch/s/steinman/ahaleyyy/.local/share/pyvista
export XDG_RUNTIME_DIR=/scratch/s/steinman/ahaleyyy/.local/temp
export TEMPDIR=$SCRATCH/.local/temp
export TMPDIR=$SCRATCH/.local/temp
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=true

module load CCEnv StdEnv/2020 gcc/9.3.0 python/3.9.6 #petsc/3.20.0

source $HOME/.virtualenvs/pv_updated/bin/activate 

#~/xvfb-run-safe python make_viz_yplus.py A case_028_ultraultralow
#~/xvfb-run-safe python make_viz_yplus.py A case_028_ultralow
#~/xvfb-run-safe python make_viz_yplus.py A case_028_low
#~/xvfb-run-safe python make_viz_yplus.py A case_028_med 
#~/xvfb-run-safe python make_viz_yplus.py A case_028_high

#~/xvfb-run-safe python make_viz_yplus.py B case_043_ultraultralow 
#~/xvfb-run-safe python make_viz_yplus.py B case_043_ultralow
#~/xvfb-run-safe python make_viz_yplus.py B case_043_low
#~/xvfb-run-safe python make_viz_yplus.py B case_043_med 
#~/xvfb-run-safe python make_viz_yplus.py B case_043_high 

#~/xvfb-run-safe python make_viz_yplus.py C case_106_ultraultralow
#~/xvfb-run-safe python make_viz_yplus.py C case_106_ultralow
#~/xvfb-run-safe python make_viz_yplus.py C case_106_low
#~/xvfb-run-safe python make_viz_yplus.py C case_106_med 
#~/xvfb-run-safe python make_viz_yplus.py C case_106_high 

#~/xvfb-run-safe python make_viz_yplus.py A_uul PTSeg028_uul_0p8
#~/xvfb-run-safe python make_viz_yplus.py A_uul PTSeg028_uul_0p64
#~/xvfb-run-safe python make_viz_yplus.py A_uul PTSeg028_uul_0p512
#~/xvfb-run-safe python make_viz_yplus.py A_uul PTSeg028_uul_0p4096 

#~/xvfb-run-safe python make_viz_yplus.py B_uul PTSeg043_uul_0p8 
#~/xvfb-run-safe python make_viz_yplus.py B_uul PTSeg043_uul_0p64
#~/xvfb-run-safe python make_viz_yplus.py B_uul PTSeg043_uul_0p512
#~/xvfb-run-safe python make_viz_yplus.py B_uul PTSeg043_uul_0p4096 

#~/xvfb-run-safe python make_viz_yplus.py C_uul PTSeg106_uul_0p8
#~/xvfb-run-safe python make_viz_yplus.py C_uul PTSeg106_uul_0p64
#~/xvfb-run-safe python make_viz_yplus.py C_uul PTSeg106_uul_0p512
#~/xvfb-run-safe python make_viz_yplus.py C_uul PTSeg106_uul_0p4096 

~/xvfb-run-safe python make_viz_yplus.py A_base PTSeg028_base_0p8
~/xvfb-run-safe python make_viz_yplus.py A_base PTSeg028_base_0p64
~/xvfb-run-safe python make_viz_yplus.py A_base PTSeg028_base_0p512
~/xvfb-run-safe python make_viz_yplus.py A_base PTSeg028_base_0p4096 

~/xvfb-run-safe python make_viz_yplus.py B_base PTSeg043_base_0p8 
~/xvfb-run-safe python make_viz_yplus.py B_base PTSeg043_base_0p64
~/xvfb-run-safe python make_viz_yplus.py B_base PTSeg043_base_0p512
~/xvfb-run-safe python make_viz_yplus.py B_base PTSeg043_base_0p4096 

~/xvfb-run-safe python make_viz_yplus.py C_base PTSeg106_base_0p8
~/xvfb-run-safe python make_viz_yplus.py C_base PTSeg106_base_0p64
~/xvfb-run-safe python make_viz_yplus.py C_base PTSeg106_base_0p512
~/xvfb-run-safe python make_viz_yplus.py C_base PTSeg106_base_0p4096 