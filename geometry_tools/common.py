import vtk 
import numpy as np 
import pyvista as pv 
from scipy.spatial import cKDTree as KDTree 
from scipy.interpolate import interp1d

# from pathlib import Path 
# import h5py 
# import ast 

def fix_vmtk_group_ids(surf):
    """ Fix group IDs.

    Args:
        surf (PolyData) : Surface with array GroupIds (optional: Mask)

    Returns:
        PolyData with fixed GroupIds

    VMTK groups IDs are often buggy, this fixes them
    based on connectivity.
    """
    import pygeodesic.geodesic as geodesic

    g_ids = np.unique(surf.point_arrays['GroupIds'])

    # Break into pieces, find which have broken ids
    masks = [surf.point_arrays['GroupIds'] == g for g in g_ids]
    groups = [surf.extract_points(m) for m in masks]

    # Find which id has most mutual with sac, overwrite
    if 'Mask' in surf.point_arrays:
        check_sac = [g.point_arrays['Mask'].sum()/g.n_points for g in groups]
        surf.point_arrays['GroupIds'][surf.point_arrays['Mask'] == 1] = g_ids[np.argmax(check_sac)]

    # Break into pieces, find which have broken ids
    masks = [surf.point_arrays['GroupIds'] == g for g in g_ids]
    groups = [surf.extract_points(m) for m in masks]
    n_parts = np.array([g.split_bodies().n_blocks for g in groups])

    # Broken groups:
    split_idx = [idx for idx, x in enumerate(n_parts > 1) if x == True]
    split_g_ids = g_ids[split_idx]

    if len(split_idx) > 0:

        tree = KDTree(surf.points)
        surf.point_arrays['GroupError'] = np.zeros(surf.n_points, dtype=int)

        for idx in split_idx:
            parts = list(groups[idx].split_bodies())
            small_parts = parts[1:]
            error_points = np.concatenate([x.points for x in small_parts], axis=0)

            _, ii = tree.query(error_points)
            surf.point_arrays['GroupError'][ii] = 1

        target_indices = [idx for idx, x in enumerate(surf.point_arrays['GroupError'] == 1) if x == True]
        source_indices = [idx for idx, x in enumerate(surf.point_arrays['GroupError'] == 0) if x == True]

        target_indices = np.array(target_indices)
        source_indices = np.array(source_indices)

        geoalg = geodesic.PyGeodesicAlgorithmExact(surf.points, surf.faces.reshape(-1, 4)[:, 1:])
        distances, best_source = geoalg.geodesicDistances(source_indices, target_indices)

        surf.point_arrays['GroupIds'][target_indices] = surf.point_arrays['GroupIds'][source_indices[best_source]]
    return surf 

def vtk_generate_img_stencil(mesh, spacing=0.05, bounds=None):
    """ Resample surf mesh to image.

    This function has memory problems, something isn't freed at end,
    so running it in a loop causes problems.
    """
    if bounds is None:
        bounds = np.array(mesh.bounds)
    else:
        bounds = np.array(bounds)

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
    # image.SetScalarType(vtk.VTK_UNSIGNED_CHAR,image.GetInformation())
    # image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)    # 3
    image.point_arrays['ImageScalars'] = np.zeros(np.prod(image.dimensions), dtype=bool)

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
    newImage.point_arrays['ImageScalars'] = newImage.point_arrays['ImageScalars']#[:,0]
    
    return newImage

def vtk_taubin_smooth(mesh, pass_band=0.1, feature_angle=60.0, iterations=20):
    """ Smooth mesh using Taubin method. 
    
    Note:
    This also exists in bsl.common. 
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(mesh) 
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff() 
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    # smoother.GenerateErrorScalarsOn() s
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

def smooth_mesh_data_local(surf, array='GroupIds', 
        func='median', neighbour_pt_ids=None, iterations=1):
    """ Smooth mesh data based on local connectivity.

    Args:
        surf (polydata): Input surface.
        array (str): Name of array to be smoothed.
    Returns:
        surf (polydata): Surface with smoothed array
        neighbour_pt_ids (list of lists): list of neighbouring point ids,
            index by point id.
    """
    if neighbour_pt_ids == None:
        neighbour_pt_ids = get_neighbour_map(surf) 
    
    neighbour_pt_ids = np.array(neighbour_pt_ids)

    surf = surf.copy()
    new_array = surf.point_arrays[array].copy()

    for idx in range(iterations):
        old_array = new_array.copy()
        for pt_id in list(range(surf.n_points)):
            # Uses 2 connexity by default
            neighbours = neighbour_pt_ids[neighbour_pt_ids[pt_id]]
            neighbours = np.unique([item for sublist in neighbours for item in sublist])
            # neighbours = neighbour_pt_ids[pt_id]

            # Unfortunately, np.median has the undesired "fallback" 
            # that uses the mean when the array is even. This is bad for 
            # data like GroupIds.
            if func == 'median':
                num_neighbours = len(neighbours)
                neighbour_vals = np.sort(old_array[neighbours])
                center_index = int(np.ceil(num_neighbours / 2) - 1)
                new_val = neighbour_vals[center_index]

            else:
                new_val = func(old_array[neighbours])

            new_array[pt_id] = new_val                        

    surf.point_arrays[array] = new_array

    return surf, neighbour_pt_ids

# def get_neighbour_map_broken(surf):#, n_points):
#     """ Get full list of adjacent neighbour pts.

#     Args:
#         cells (array): Cell connectivity, shape (n_cells, 3).
#         n_pts (int): Number of point ids.
    
#     Returns:
#         neighbour_pt_ids (list): List of lists containing neighbour pt ids.
#         * Trying with numpy array.
#     """
#     # neighbour_pt_ids = [[] for _ in range(surf.n_points)]
#     edges = surf.extract_all_edges()

#     # edges.points does not neccesarily == surf.points!!
#     # create a map between them
#     tree = KDTree(surf.points)
#     _, ii = tree.query(edges.points, k=1)

#     ee = edges.lines.reshape(-1, 3)[:,1:]
    
#     ee = ee[np.argsort(ee[:, 0])]
    
#     diff = np.diff(ee[:,0])
    
#     upper = np.argwhere(diff) + 1
#     upper = np.concatenate([upper.flatten(), [len(ee)]], axis=0)
#     lower = np.roll(upper,1)
#     lower[0] = 0

#     max_connect = np.diff(upper).max()
#     index = np.zeros((len(upper), 2), dtype=int)
#     neighbour_pt_ids = np.empty((surf.n_points, max_connect), dtype=np.int)
#     neighbour_pt_ids.fill(np.nan)

#     index[:, 0] = lower.flatten()
#     index[:, 1] = upper.flatten()
#     # index[-1] = [upper[-1], len(ee)]    

#     for e, i in enumerate(index): #(surf.n_points):
#         sub = ee[i[0]:i[1]]
#         unique = np.unique(sub[:,1])
#         # print(unique)
#         neighbour_pt_ids[ii[e]][:len(unique)] = unique
#         # neighbour_pt_ids[ii[unique]] = e
#         for u in unique:
#             neighbour_pt_ids[ii[u]] = e

#     neighbour_pt_ids = [x[~np.isnan(x)] for x in neighbour_pt_ids]
#     neighbour_pt_ids = np.array([np.unique(x) for x in neighbour_pt_ids], dtype='object')

#     return neighbour_pt_ids

def get_neighbour_map(surf):#, n_points):
    """ Get full list of adjacent neighbour pts.

    Args:
        cells (array): Cell connectivity, shape (n_cells, 3).
        n_pts (int): Number of point ids.
    
    Returns:
        neighbour_pt_ids (list): List of lists containing neighbour pt ids.
        * Trying with numpy array.
    """
    neighbour_pt_ids = [[] for _ in range(surf.n_points)]
    edges = surf.extract_all_edges()

    # edges.points does not neccesarily == surf.points!!
    # create a map between them
    tree = KDTree(surf.points)
    _, ii = tree.query(edges.points, k=1)

    ee = edges.lines.reshape(-1, 3)[:,1:]
    
    for e in ee:
        neighbour_pt_ids[ii[e[0]]].append(ii[e[1]])
        neighbour_pt_ids[ii[e[1]]].append(ii[e[0]])

    neighbour_pt_ids = np.array([np.unique(x) for x in neighbour_pt_ids], dtype='object')

    return neighbour_pt_ids


def create_edge_size_array(surf, min_edge_size=0.1, max_edge_size=0.4, sac_size=0.15, misr_min=0.1, misr_max=2.5, name='Size',):
    """ Create "Size" array incorporating distance to centerlines and curvature.

    This will likely be refined moving forward.

    Based on DistanceToCenterlinesArray, interpolate between 2.5 mm rad as max, 0.5 mm rad min
    Based on Curvature, interpolate between 0.3 as min, 0.8 as max
    Based on Mask, set to min value where Mask == 1.
    Then take min of each.

    """
    distance_interp = interp1d([misr_min, misr_max], [min_edge_size, max_edge_size], 
        kind='linear',
        bounds_error=False,
        fill_value=(min_edge_size, max_edge_size),
        )
    curv_interp = interp1d([0.3, 0.8], [max_edge_size, min_edge_size], 
        kind='linear',
        bounds_error=False,
        fill_value=(max_edge_size, min_edge_size),
        )

    # The perfectly straight flow extensions end up having high curvature 
    # unless they are perturbed slightly
    surf = surf.compute_normals()
    surf_perturb = surf.copy()
    perturbed_vec = np.einsum(
        'ij,i->ij', 
        surf_perturb.point_arrays['Normals'], 
        np.random.normal(0, 0.0001, surf_perturb.n_points)
        )
    surf_perturb.points = surf_perturb.points + perturbed_vec
    surf_perturb.point_arrays['Curvature'] = np.abs(surf_perturb.curvature('Minimum'))
    
    surf.point_arrays['Curvature'] = surf_perturb.point_arrays['Curvature'] #np.abs(surf.curvature('Minimum'))
    
    surf.point_arrays['SizeDistanceToCenterlinesArray'] = distance_interp(surf.point_arrays['DistanceToCenterlinesArray']) #np.ones(surf.n_points) 
    surf.point_arrays['SizeCurvature'] = curv_interp(surf.point_arrays['Curvature']) #np.ones(surf.n_points)

    surf.point_arrays[name] = np.minimum(surf.point_arrays['SizeDistanceToCenterlinesArray'], surf.point_arrays['SizeCurvature'])
    surf, n_ids = smooth_mesh_data_local(surf, name, np.mean, iterations=1)

    if 'Mask' in surf.point_arrays:
        # First dilate mask to include nearby regions
        surf.point_arrays['MaskDilate'] = surf.point_arrays['Mask'].copy()
        surf, _ = smooth_mesh_data_local(surf, 'MaskDilate', np.max, iterations=6, neighbour_pt_ids=n_ids)

        sac_mask = surf.point_arrays['MaskDilate'] == 1
        # sac_size_array = sac_size * np.ones(len(sac_mask))
        current_size_array = surf.point_arrays[name][sac_mask]

        surf.point_arrays[name][sac_mask] = np.minimum(sac_size, current_size_array)

    else:
        print('No mask in create_edge_size_array.')

    surf, n_ids = smooth_mesh_data_local(surf, name, np.mean, iterations=2)

    return surf


class SacSelectTool():
    """ Interactively mark points using a probe.

    I think this is obsolete? 
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

    

class SelectGeodesic():
    def __init__(self, mesh, scalars='Mask'):
        mesh = mesh.clean()
        mesh = mesh.compute_normals(auto_orient_normals=True)
        self.mesh = mesh
        self.scalars = scalars
        if self.scalars not in mesh.point_arrays:
            self.mesh.point_arrays[self.scalars] = np.zeros(self.mesh.n_points)

        self.current_mask = np.zeros_like(self.mesh.point_arrays[self.scalars])
        self.current_pts = self.mesh.points 
        
        self.stored_points = []
        self.picked_points = []
        self.picked_ids = []
        self.lines = []
        self.interactive = False
        self.tree = KDTree(self.mesh.points)
        
    def interact(self, title='Isolate aneurysms.'):
        self.interactive = True
        self.p = pv.Plotter() 
        self.mesh = self.mesh.compute_normals()
        self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='Reds', show_edges=True)
        self.p.enable_point_picking(
            show_point=True,
            show_message=False,
            callback=self._cb,
            color='red',
            font_size=12,
            point_size=15,
            tolerance=0.005
            )

        self.p.add_text(title, position='upper_left', font_size=18)
        msg = 'Keys:'
        self.p.add_text(msg, position=(0.05, 175), font_size=12)   
        msg = 'f: select points'
        self.p.add_text(msg, position=(0.05, 150), font_size=12)
        msg = 'u: undo'
        self.p.add_text(msg, position=(0.05, 125), font_size=12)
        msg = 'space: complete loop'
        self.p.add_text(msg, position=(0.05, 100), font_size=12)
        msg = 'a: append mask'
        self.p.add_text(msg, position=(0.05, 75), font_size=12)
        msg = 'x: smooth section'
        self.p.add_text(msg, position=(0.05, 50), font_size=12)
        msg = 'q: quit'
        self.p.add_text(msg, position=(0.05, 25), font_size=12)

        self.p.add_key_event('u', self._undo)
        self.p.add_key_event('space', self._finish)
        self.p.add_key_event('a', self.append)
        self.p.add_key_event('x', self.refill_section)
        self.p.add_key_event('c', self._clear)
        self.p.show()
        self.stored_points.append(np.array(self.picked_points))

        return self._get_points()

    def _cb(self, pt):
        """ CB for picking points. """
        self.picked_points.append(pt)
        self.update_points()
        self.update_geodesic()
        self.display()

    def _undo(self):
        """ Remove last picked point and update display. """
        if len(self.picked_points) > 0:
            self.picked_points.pop()
            self.picked_ids.pop() 
        self.update_points()
        self.update_geodesic()
        self.display()

    def _clear(self):
        """ Clear all current picked points and update display. """
        self.picked_points = []
        self.picked_ids = []
        self.lines = []
        self.update_points()
        self.update_geodesic()
        self.display()

    def _finish(self):
        """ Close current loop and update display."""
        self.picked_points.append(self.picked_points[0])
        self.picked_ids.append(self.picked_ids[0])
        self.update_geodesic()
        self.display()
        self.update_mesh()

    def append(self):
        """ Store current geodesic, start a new geodesic. """
        # Stored existing mask array
        # When calling update_mesh, logical or with existing
        self.current_mask = self.mesh.point_arrays[self.scalars].copy() 
        self.current_pts = self.mesh.points 

        self.stored_points.append(self.picked_points)

        self.picked_points = []
        self.picked_ids = []
        self.lines = []
        if self.interactive:
            self.display()

    def update_points(self):
        """ Gets ids of picked points. """
        if len(self.picked_points) > 0:
            _, self.picked_ids = self.tree.query(self.picked_points, k=1)
            self.picked_ids = list(self.picked_ids)
        else:
            self.picked_ids = []
        # Clean duplicates 
        _, idx = np.unique(self.picked_ids, return_index=True)
        # self.picked_ids = list(np.array(self.picked_ids)[idx])
        # self.picked_points = list(np.array(self.picked_points)[idx])

    def update_geodesic(self):
        """ Updates geodesic using current picked points. """
        if len(self.picked_ids) > 1:
            pairwise = zip(self.picked_ids, self.picked_ids[1:])
            self.lines = [self.mesh.geodesic(a, b) for a, b in pairwise]
        else:
            self.lines = []
       
        if len(self.lines) > 0:
            lines = pv.PolyData() 
            self.merged = lines.merge(self.lines)
        

    def update_mesh(self):
        """ Split the mesh based on current geodesic. """
        # Split the mesh
        self.mesh = self.mesh.triangulate()
        tree = KDTree(self.mesh.points)
        _, ii = tree.query(self.merged.points, k=1)
        split, rdx = self.mesh.remove_points(ii)
        split.point_arrays['vtkOGIds'] = rdx
        
        split = split.connectivity()
        region_ids = split.point_arrays['RegionId']
        regions = np.unique(region_ids)
        r_masks = [region_ids == r_id for r_id in regions]
        split = [split.extract_points(r_m, adjacent_cells=False) for r_m in r_masks]
        split = sorted(split, key=lambda x: x.n_points, reverse=True)

        split_pd = [pv.PolyData(s.points, s.cells) for s in split]
        for s, s_pd in zip(split, split_pd):
            for arr in self.mesh.point_arrays:
                s_pd.point_arrays[arr] = s.point_arrays[arr]
            for arr in self.mesh.cell_arrays:
                s_pd.cell_arrays[arr] = s.cell_arrays[arr]

        # Smaller one mark 1, bigger 
        mask = np.ones(self.mesh.n_points, dtype=bool)
        mask[split[0].point_arrays['vtkOGIds']] = 0
        temp_mask = self.mesh.point_arrays[self.scalars]
        temp_mask[mask] = 1
        temp_mask[~mask] = 0

        # DM 11 11 21
        # Commented out the logical or, just used temp_mask
        # Interp old mask onto new
        tree = KDTree(self.current_pts)
        _, ii = tree.query(self.mesh.points,k=1)
        self.mesh.point_arrays[self.scalars] = self.current_mask[ii]
        new_mask = np.logical_or(temp_mask, self.mesh.point_arrays[self.scalars])
        self.mesh.point_arrays[self.scalars] = new_mask

        # self.mesh.point_arrays[self.scalars] = temp_mask

        if self.interactive:
            # self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='coolwarm')
            self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='Reds', show_edges=True)

    def display(self):
        """ Update display based on current state. """
        
        if len(self.picked_ids) > 1:
            tube = polyline_from_points(self.merged.points).tube(0.01)
            # line = pv.PolyData(self.merged.points, self.merged.cells)
            self.p.add_mesh(tube, name='lines', color='b')
        else:
            self.p.add_mesh(pv.Sphere(center=self.mesh.center), name="lines", opacity=0.0)
    
        if len(self.picked_points) > 0:
            points = pv.wrap(np.array(self.picked_points))

            self.p.add_mesh(points, 
                render_points_as_spheres=True, 
                color='r',
                name='points',
                )
        else:
            self.p.add_mesh(pv.Sphere(center=self.mesh.center), name="points", opacity=0.0)

    def smooth_section(self):
        """ Smoothes section with Laplacian filtering. """
        mask = self.mesh.point_arrays[self.scalars] == 1
        submesh = self.mesh.extract_points(mask, adjacent_cells=False)
        submesh = pv.PolyData(submesh.points, submesh.cells)
        submesh = submesh.smooth(n_iter=100, boundary_smoothing=False)
        self.mesh.points[mask] = submesh.points

        if self.interactive:
            # self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='coolwarm')
            self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='Reds', show_edges=True)

    def refill_section(self):
        """ Cut a hole and fill it. 
        
        """
        mask = self.mesh.point_arrays[self.scalars] == 0
        mask_sub = self.mesh.point_arrays[self.scalars] == 1

        self.mesh = self.mesh.clean()
        self.mesh = self.mesh.fill_holes(15.0)
        self.mesh = self.mesh.clean()

        mesh = self.mesh.extract_points(mask, adjacent_cells=False)
        submesh = self.mesh.extract_points(mask_sub)
        submesh = pv.PolyData(submesh.points, submesh.cells)
        edges = submesh.extract_feature_edges(boundary_edges=True, 
            non_manifold_edges=False, feature_edges=False, manifold_edges=False)
        submesh = pv.wrap(edges.points).delaunay_2d()
        # submesh = submesh.decimate(0.7)

        mesh = mesh.merge(submesh)
        mesh = pv.PolyData(mesh.points, mesh.cells)
        # mesh = mesh.boolean_union(submesh)

        # mesh = pv.PolyData(mesh.points, mesh.cells)
        self.mesh = mesh
        self.mesh.point_arrays[self.scalars] = np.zeros(self.mesh.n_points)

        self.mesh = self.mesh.clean()
        self.mesh = self.mesh.fill_holes(20.0)
        
        self.tree = KDTree(self.mesh.points)

        if self.interactive:
            # self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='coolwarm')
            self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='Reds', show_edges=True)

    def _get_points(self):
        """ Put the points in a geometry.
        """
        print("TEST", self.stored_points)
        self.stored_points = [x for x in self.stored_points if len(x) > 1]
        points = pv.MultiBlock()
        if len(self.stored_points) > 0:
            for pts in self.stored_points:
                points.append(pv.wrap(np.array(pts)))
            return points
        else:
            return []


    def save_stored_points(self, outfile=None):
        """ Save stored points for later use. """
        # neck_ids = [np.zeros(len(ll), dtype=int) + idx for idx, ll in enumerate(self.stored_points)]
        # neck_ids = [item for sublist in neck_ids for item in sublist]

        # points = pv.wrap(np.concatenate(self.stored_points, axis=0))
        # points.point_arrays['NeckIds'] = neck_ids

        points = pv.MultiBlock()
        for pts in self.stored_points:
            points.append(pv.wrap(np.array(pts)))

        if outfile is not None:
            points.save(outfile)       
        return points

    def use_stored_points(self, points):
        for pts in points:
            self.picked_points = pts.points
            self.update_points()
            self.update_geodesic()
            self.update_mesh()
            self.append()


class ClickDragDelete:
    """ Click and drag to select, space to delete.

    Clip meshes based on cell picking routines.
    Input surface must be vtkPolyData.
    
    Instructions:
    - Press "r" to toggle between selection/interaction.
    - Press "c" to clear selection.
    - Press "space" to delete cells.
    
    Based on example here:
    https://github.com/pyvista/pyvista/pull/281
    """
    def __init__(self, mesh, title='Clip mesh'):
        self.plotter = pv.Plotter()
        self.plotter.add_text(title, position='upper_left', font_size=18)
        msg = 'r: toggle selection mode'
        self.plotter.add_text(msg, position=(0.05, 75), font_size=12)
        msg = 'c: clear selection'
        self.plotter.add_text(msg, position=(0.05, 50), font_size=12)
        msg = 'space: clip '
        self.plotter.add_text(msg, position=(0.05, 25), font_size=12)
        msg = 'k: flag mesh'
        self.plotter.add_text(msg, position=(0.05, 0.05), font_size=12)

        self.mesh = mesh
        self.clear()

        self.plotter.enable_cell_picking(callback=self, show=False, show_message=False)
        self.plotter.add_key_event('c', callback=self.clear)
        self.plotter.add_key_event('space', callback=self.clip)
        self.plotter.add_key_event('k', callback=self.flag)

        self.flag_inspect = False

        self.plotter.show()

    def display(self):
        if self.mesh.n_points > 0:
            self.plotter.add_mesh(self.mesh,
                scalars='DelMask',
                name='mesh',
                show_scalar_bar=False,
                cmap='Reds')
        else:
            self.plotter.remove_actor('mesh')
        
    def __call__(self, picked_cells):
        self.picked.merge(picked_cells, inplace=True)
        if self.picked.n_cells > 0:
            self.mesh['DelMask'][self.picked.cell_arrays['orig_extract_id']] = 1
        self.display()

        return
    
    def clear(self):
        self.picked = pv.UnstructuredGrid()
        self.mesh.cell_arrays['DelMask'] = np.zeros(self.mesh.n_cells, dtype=bool)

        self.plotter.add_mesh(self.mesh, 
            color='w', 
            scalars='DelMask', 
            name='mesh',
            show_scalar_bar=False)
        self.display()

    def clip(self):
        if self.picked.n_points > 0:
            cells = np.invert(self.mesh.cell_arrays['DelMask'])
            self.mesh = self.mesh.extract_cells(cells)
            self.mesh.cell_arrays['DelMask'] = np.zeros(self.mesh.n_cells, dtype=bool)
            self.plotter.enable_cell_picking(self.mesh, callback=self, show=False, show_message=False)
            self.clear()
            self.display()
        
    def flag(self):
        print('Meshed flagged for further inspection.')
        self.flag_inspect = True

class ClickToDelete():
    """ Click a point, cells that contain it will be deleted.
    
    I don't think this works?
    """
    def __init__(self, mesh):
        self.p = pv.Plotter()
        msg = 'Press f to select point'
        self.p.add_text(msg, position=(0.05, 50), font_size=12)
        msg = 'Press x to delete cells'
        self.p.add_text(msg, position=(0.05, 25), font_size=12)
        msg = 'Press u to udno'
        self.p.add_text(msg, position=(0.05, 0), font_size=12)
        
        self.mesh = mesh
        self.prev_mesh = self.mesh.copy()

        self.p.add_mesh(self.mesh, color='w', name='mesh')
        self.p.enable_point_picking(callback=self, show_message=False)
        self.p.enable_cell_picking()
        self.p.add_key_event('x', callback=self)
        self.p.add_key_event('u', callback=self.undo)

        self.p.show()

    def __call__(self): # picked_cells
        pt = self.p.picked_point_id
        self.prev_mesh = self.mesh.copy()
        self.mesh, _ = self.mesh.remove_points([pt], mode='any')
        self.display()

    def display(self):
        if self.mesh.n_points > 0:
            self.p.add_mesh(self.mesh,
                color='w',
                name='mesh',
            )
        else:
            self.p.remove_actor('mesh')

    def undo(self):
        self.mesh = self.prev_mesh
        self.display()

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
        # print(node)
        mask = centerlines_branched.cell_arrays['GroupIds'] == int(node)

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


def get_sac_surface_mask(mesh, sac):
    """ Get ids of surface points of sac on mesh.
    """
    mesh.point_arrays['vtkOGIds'] = list(range(mesh.n_points))

    sac = sac.fill_holes(20.0)
    sac = sac.compute_normals(auto_orient_normals=True)
    sac_inflate = sac.copy()
    sac_inflate.points = sac.points + 0.1*sac.point_arrays['Normals']

    mesh = mesh.select_enclosed_points(sac_inflate, check_surface=False)
    mesh['SacMask'] = mesh.point_arrays['SelectedPoints']
    mesh, _ = smooth_mesh_data_local(mesh, array='SacMask')

    surf = mesh.extract_surface()
    mesh_sac = surf.extract_points(surf.point_arrays['SacMask'] == 1)

    mesh_sac_ids = mesh_sac.point_arrays['vtkOGIds'].copy()

    mesh_sac_array = np.zeros(mesh.n_points, dtype=int)
    mesh_sac_array[mesh_sac_ids] = 1

    mesh.point_arrays['SurfaceSacMask'] = mesh_sac_array.astype(bool)

    return mesh

    
def decimate_edge_length(surf, target_edge_length):
    edges = surf.extract_all_edges()
    mean_el = edges.compute_cell_sizes().cell_arrays['Length'].mean()
    target_el = target_edge_length
    target_reduction = 1 - (mean_el / target_el)
    surf_d = surf.decimate(target_reduction, volume_preservation=True)
    surf = copy_arrays(surf, surf_d)
    return surf

def copy_arrays(src, dst):
    tree = KDTree(src.points)
    _, ii = tree.query(dst.points, k=1)
    for arr in src.point_arrays:
        dst.point_arrays[arr] = src.point_arrays[arr][ii]

    centers = src.cell_centers()
    tree = KDTree(centers.points)
    _, ii = tree.query(dst.cell_centers().points, k=1)
    for arr in src.cell_arrays:
        dst.cell_arrays[arr] = src.cell_arrays[arr][ii]
        
    return dst
