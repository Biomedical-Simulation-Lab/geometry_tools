from setuptools import setup

setup(
    name='geometry_tools',
    version='0.2dev',
    packages=['geometry_tools',],
    scripts=['geometry_tools/scripts/surface_prep.py',
    		'geometry_tools/scripts/surface_process.py',
    		'geometry_tools/scripts/volume_meshing.py',
    		'geometry_tools/scripts/meshquality.py',
    		'geometry_tools/scripts/make_mesh.py',
    		'geometry_tools/scripts/make_MISR_DTCL_meshes.py',
    		'geometry_tools/scripts/map_info.py',
    		'geometry_tools/scripts/map_info_bilateral.py',
    		'geometry_tools/scripts/map_info_direct.py',
    		'geometry_tools/scripts/map_DTCL_MISR.py',
    		'geometry_tools/scripts/make_spectro_points.py',
    		'geometry_tools/scripts/make_centerlines.py',
    		'geometry_tools/scripts/centerlines_fixed.py',
    		'geometry_tools/scripts/centerlines_single.py',
			'geometry_tools/scripts/make_ref_mesh.py',],
)
