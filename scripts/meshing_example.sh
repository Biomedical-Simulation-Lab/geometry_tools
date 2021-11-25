###############################################################################
# Meshing example using geometry_tools
# The environments are specific to DM's workspace, 
# but the rest should be easily adapted.
###############################################################################

###############################################################################
# Get dir of example data file, make a project directory.
# Could change $surf_file and $proj_dir to be file inputs instead.
###############################################################################

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $script_dir

parent_dir="$(dirname "$script_dir")"

surf_file="$parent_dir/data/Pair10a.stl"
proj_dir="$parent_dir/test_mesher"

if [ ! -d $proj_dir ]; then
  mkdir -p $proj_dir;
fi

surf_resampled="$proj_dir/Pair10a_rs.vtp"

###############################################################################
# Now, we actually use the scripts.
# Because VMTK doesn't play nice with some other tools, 
# we use it in a seperate environment.
################################################################################

# Resample surface
source ~/opt/miniconda3/etc/profile.d/conda.sh
conda activate surge
python resample.py $surf_file $surf_resampled

# Clip and prep surface
conda activate vmtk
python surface_prep.py $surf_resampled $proj_dir

# Process and refine
python surface_process.py $proj_dir

# Volume mesh and generate submission files.
# python surface_process.py $proj_dir