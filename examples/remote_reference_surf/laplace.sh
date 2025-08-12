#!/bin/bash

#######SBATCH -A ctb-steinman
###########SBATCH -p debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=1:00:00
#SBATCH --job-name heat
#SBATCH --output=heat.txt

#export OMP_NUM_THREADS=20

cd $SLURM_SUBMIT_DIR

module load NiaEnv/.2020a intel/2020u1 intelmpi/2020u1 intelpython3/2020u1 cmake/3.16.3 boost/1.69.0 eigen/3.3.7 hdf5/1.8.21 netcdf/4.6.3 gmp/6.2.0 mpfr/4.0.2 swig/4.0.1 petsc/3.10.5 trilinos/12.12.1 fenics/2019.1.0

source activate oasis

mpirun -np 1 python laplace.py A
#mpirun -np 1 python laplace.py B
#mpirun -np 1 python laplace.py C

#wait
