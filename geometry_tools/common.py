import vtk 
import numpy as np 
import pyvista as pv 
from scipy.spatial import cKDTree as KDTree 

def vtk_generate_img_stencil(mesh, spacing=0.05):
    """ Resample surf mesh to image.

    This function has memory problems, something isn't freed at end,
    so running it in a loop causes problems.
    """
    bounds = np.array(mesh.bounds)
    bounds_lengths = np.diff(bounds.reshape(3,2)).T[0]

    # Inflate bounds
    inflate_lens = bounds_lengths * 0.05
    bounds[::2] -= inflate_lens
    bounds[1::2] += inflate_lens
    bounds_lengths = np.diff(bounds.reshape(3,2)).T[0]

    spacing = (spacing, spacing, spacing)
    dimensions = (bounds_lengths / spacing).astype(int)
    dims = (dimensions - 1)
    origin = bounds[::2]

    image = pv.UniformGrid() 
    image.dimensions = dimensions 
    image.origin = origin
    image.spacing = spacing
    image.SetScalarType(vtk.VTK_UNSIGNED_CHAR,image.GetInformation())
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)    

    pol2Stenc = vtk.vtkPolyDataToImageStencil()
    pol2Stenc.SetTolerance(0.5) 
    pol2Stenc.SetInputData(mesh)
    pol2Stenc.SetInformationInput(image)
    pol2Stenc.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image)
    stencil.ReverseStencilOn()
    stencil.SetBackgroundValue(1)
    stencil.SetStencilData(pol2Stenc.GetOutput())
    stencil.Update()

    newImage = pv.wrap(stencil.GetOutput())
    newImage.point_arrays['ImageScalars'] = newImage.point_arrays['ImageScalars'][:,0]
    
    return newImage

def vtk_taubin_smooth(mesh, pass_band=0.1, feature_angle=60.0, iterations=20):
    """ Smooth mesh using Taubin method. """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(mesh) 
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff() 
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return pv.wrap(smoother.GetOutput())


def smooth_mesh_data(data_array, points, radius, func=np.mean, mask=None):
    """ Smooth the data, not the mesh. 

    Radius-based moving average filter.
    Incorporate distance weighting?
    """
    tree = KDTree(points)
    inds = [tree.query_ball_point(pt, radius) for pt in points]
    out_array = np.zeros_like(data_array)

    if mask is None:
        mask = np.ones(len(points))

    for idx in range(len(data_array)):
        value = func(data_array[inds[idx]])
        
        if np.any(mask[inds[idx]] == 1):
            out_array[idx] = value 
        else:
            out_array[idx] = data_array[idx]

    return out_array

def create_edge_size_array(surf, max_size=0.3, min_size=0.18, curvature_percentile=80, name='Size'):
    """ Create "Size" array incorporating distance to centerlines and curvature.

    This will likely be refined moving forward.
    """
    surf.point_arrays['Curvature'] = np.abs(surf.curvature())
    curvature_threshold = np.percentile(surf.point_arrays['Curvature'], curvature_percentile)
    
    surf.point_arrays[name] = np.ones(surf.n_points) 
    
    surf.point_arrays[name] = surf.point_arrays['DistanceToCenterlinesArray'] / 4.0
    surf.point_arrays[name][surf.point_arrays['Curvature'] > curvature_threshold] = min_size

    surf.point_arrays[name][surf.point_arrays[name] < min_size] = min_size
    surf.point_arrays[name][surf.point_arrays[name] > max_size] = max_size

    surf.point_arrays[name] = smooth_mesh_data(surf.point_arrays[name], surf.points, 0.5)
    return surf



class SacSelectTool():
    """ Interactively mark points using a probe.
    """ 
    def __init__(self, surf):
        self.surf = surf
        self.surf.point_arrays['Mask'] = np.zeros(self.surf.n_points)
        self.surf.point_arrays['TempMask'] = np.zeros(self.surf.n_points)

    def mask(self, center, radius):
        sphere = pv.Sphere(radius=radius, center=center)
        self.surf = self.surf.select_enclosed_points(sphere)
        mask_index = self.surf.point_arrays['SelectedPoints']
        ids = [x for x in range(self.surf.n_points) if mask_index[x] == True]
        self.selection = self.surf.extract_points(ids)

        self.selection.point_arrays['vtkOGIds'] = self.selection.point_arrays['vtkOriginalPointIds'].copy()
        self.selection = self.selection.extract_largest()
        mask = self.selection.point_arrays['vtkOGIds']
        self.selection = pv.PolyData(self.selection.points, self.selection.cells)
        self.selection.point_arrays['vtkOGIds'] = mask
        self.selection = self.selection.clean()

        self.surf.point_arrays['TempMask'] = self.surf.point_arrays['Mask'].copy()
        self.surf.point_arrays['TempMask'][self.selection.point_arrays['vtkOGIds']] = 1
            
    def select(self):

        def sphere_cb(xyz, probe):
            # Select enclosed points
            self.sphere = pv.Sphere(radius=probe.GetRadius(), center=xyz)
            # self.probe = probe
            self.p.add_mesh(self.sphere, color='r', opacity=0.4, name='probe')
            self.center = probe.GetCenter()
            self.radius = probe.GetRadius()
            self.mask(self.center, self.radius)
            self.p.add_mesh(self.surf, scalars='TempMask', name='surf')

        def choose_cb():
            self.surf.point_arrays['Mask'] = self.surf.point_arrays['TempMask']
            
        self.p = pv.Plotter()
        self.p.add_mesh(self.surf, scalars='TempMask', opacity=1.0, name='surf')
        self.p.add_sphere_widget(
            callback=sphere_cb, 
            center=np.mean(self.surf.points, axis=0), 
            radius=4.0, 
            pass_widget=True,
            theta_resolution=8,
            phi_resolution=8,
            style='wireframe',
            )
        self.p.add_key_event('space', choose_cb)
        self.p.show()

        if np.all(self.surf.point_arrays['Mask'] == 0):
            self.surf.point_arrays['Mask'] = self.surf.point_arrays['TempMask']

def get_network_endpoints(network):
    """ Get terminal points of network
    """
    # First, get all unique endpoints in the network
    endpoints = []
    cells = []
    for ndx in range(network.n_cells):
        cell_mask = np.zeros(network.n_cells, dtype=bool)
        cell_mask[ndx] = True
        cell = network.extract_cells(cell_mask)
        endpoints.append(cell.points[0])
        endpoints.append(cell.points[-1])
        cells.append(cell)

    endpoints = np.array(endpoints)

    # Look for near-duplicates
    distances = np.zeros((endpoints.shape[0], endpoints.shape[0]))
    for idx in range(len(endpoints)):
        for jdx in range(idx, len(endpoints)):
            distances[idx, jdx] = np.linalg.norm(endpoints[idx] - endpoints[jdx])
            distances[jdx, idx] = distances[idx, jdx]

    tol = 1e-4
    distances = distances > tol
    np.fill_diagonal(distances, True)

    unique = [np.all(x == True) for x in distances]
    network_endpoints = endpoints[unique]
    return network_endpoints

def split_network_into_cells(network):
    cells = []
    for ndx in range(network.n_cells):
        cell_mask = np.zeros(network.n_cells, dtype=bool)
        cell_mask[ndx] = True
        cell = network.extract_cells(cell_mask)
        cells.append(cell)
    return cells 

def get_mean_radii(centerlines_branched, grouplist):
    """ Get mean radii of branches.

    Used for Chnafa flow splitting method.
    """
    mean_radii = {}

    # Get mean radii
    for node in grouplist:
        mask = centerlines_branched.cell_arrays['GroupIds'] == node 

        branch_segments = centerlines_branched.extract_cells(mask)
        branch = branch_segments.split_bodies()[0]
        radius = branch.point_arrays['MaximumInscribedSphereRadius']
        
        # Convert to basic line for faster operations
        branch = lines_from_points(branch.points)
        branch.point_arrays['MaximumInscribedSphereRadius'] = radius
        branch = branch.ptc()

        branch = branch.compute_cell_sizes()
        lengths = branch.cell_arrays['Length'] 
        radius = branch.cell_arrays['MaximumInscribedSphereRadius']

        branch_length = np.sum(lengths)

        branch_resistance = np.sum(lengths / radius**4)

        mean_radius = (branch_length / branch_resistance)**0.25

        mean_radii[node] = mean_radius

    return mean_radii

def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


def polyline_from_points(points):
    """ Convert a list of points to a PolyLine"""
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

def check_mem_usage():
    import psutil
    import os
    p = psutil.Process(os.getpid())
    mem_usage = p.memory_info().rss / 1024 / 1024
    print("{} MB".format(mem_usage))
