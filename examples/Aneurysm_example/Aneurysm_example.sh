#!/usr/bin/env bash

#In this example, an aneurysm mesh is produced using distance to centerlines and curvature to create the size array, with edge lengths between 0.2-0.3 mm, no refinement or enlargement patches.

surface_prep.py Aneurysm_example.stl ./Aneurysm_example a
surface_process.py ./Aneurysm_example_low a single no_ref reg 0.2 0.3 none >lowmeshout.txt
volume_meshing.py ./Aneurysm_example_low a >>lowmeshout.txt

meshquality.py Aneurysm_example_low/mesh/Aneurysm_example_cl_pr.vtu
