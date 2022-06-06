from setuptools import setup

setup(
    name='geometry_tools',
    version='0.1dev',
    packages=['geometry_tools',],
    scripts=['geometry_tools/scripts//surface_prep.py',
    		'geometry_tools/scripts/surface_process.py',
    		'geometry_tools/scripts/volume_meshing.py',],
)
