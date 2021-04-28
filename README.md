# Geometry tools 

VMTK is great, but the interfacing via `vmtkscripts` can be cumbersome. 
The module `vmtk_wrapper` is largely a wrapper for `vmtkscripts`, 
incorporating the ease-of-use of `PyVista`. Moving forward, it would be nice
to re-write this using `vtkvmtk` and `PyVista` directly, but wrapping 
`vmtkscripts` works for now. `common` contains some useful odds and ends.
`tubeclipper` contains an experimental script for clipping a mesh "locally"
while maintaining continuity elsewhere. 