#!/bin/bash

########SBATCH -A ctb-steinman
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --time=4:59:00
####SBATCH --time=1:00:00
######SBATCH -p debug
#SBATCH --job-name cfl
#SBATCH --output=cfl_%j.txt

export MPLCONFIGDIR=/scratch/s/steinman/ahaleyyy/.config/mpl
export PYVISTA_USERDATA_PATH=/scratch/s/steinman/ahaleyyy/.local/share/pyvista
export XDG_RUNTIME_DIR=/scratch/s/steinman/ahaleyyy/.local/temp
export TEMPDIR=$SCRATCH/.local/temp
export TMPDIR=$SCRATCH/.local/temp
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=true

module load CCEnv StdEnv/2020 gcc/9.3.0 python/3.9.6 #petsc/3.20.0

source $HOME/.virtualenvs/pv_updated/bin/activate 

#python cfl.py A
#python time_RAM.py A
#python cfl.py B
#python time_RAM.py B
#python cfl.py C
#python time_RAM.py C

#python cfl.py A uul
#python time_RAM.py A uul
#python cfl.py B uul
#python time_RAM.py B uul
#python cfl.py C uul
#python time_RAM.py C uul

python cfl.py A base
python time_RAM.py A base
#python cfl.py B base
#python time_RAM.py B base
#python cfl.py C base
#python time_RAM.py C base