#!/bin/bash

########SBATCH -A ctb-steinman
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=23:59:00
########SBATCH --time=1:00:00
##########SBATCH -p debug
#SBATCH --job-name DPQ_case_A
#SBATCH --output=DPQ_%j.txt

export MPLCONFIGDIR=/scratch/s/steinman/ahaleyyy/.config/mpl
export PYVISTA_USERDATA_PATH=/scratch/s/steinman/ahaleyyy/.local/share/pyvista
export XDG_RUNTIME_DIR=/scratch/s/steinman/ahaleyyy/.local/temp
export TEMPDIR=$SCRATCH/.local/temp
export TMPDIR=$SCRATCH/.local/temp
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=true


#module load NiaEnv/.2020a intel/2020u1 intelmpi/2020u1 intelpython3/2020u1 cmake/3.16.3 boost/1.69.0 eigen/3.3.7 hdf5/1.8.21 netcdf/4.6.3 gmp/6.2.0 mpfr/4.0.2 swig/4.0.1 petsc/3.10.5 trilinos/12.12.1 fenics/2019.1.0
#source activate oasis
#mpirun -np 1 python laplace.py A notref
#mpirun -np 1 python laplace.py A uul
#mpirun -np 1 python laplace.py A base

#conda deactivate
#module purge
module load CCEnv StdEnv/2020 gcc/9.3.0 python/3.9.6 #petsc/3.20.0
source $HOME/.virtualenvs/pv_updated/bin/activate 

#python slices_heat.py A uul
#python slices_heat.py A base
#mpirun -np 20 python DPQ.py A notref noskip 
#mpirun -np 20 python DPQ.py A uul noskip
mpirun -np 20 python DPQ.py A base noskip

