""" Wrapper for vmtkscripts.

Most functions correspond to a particular item in vmtkscripts,
but some are just used in similar ways. Moving forward, this 
should eventually start using vtkvmtk directly rather than 
vmtkscripts.

Functions often provide a simplified input; feel free to add 
optional arguments.
"""

from vmtk import vmtkscripts
import pyvista 
from geometry_tools import utils 
import pyvista as pv
import numpy as np 
from scipy.spatial import cKDTree as KDTree 
from scipy.optimize import least_squares
import networkx as nx
import gzip


def clipper(surf):
    """ Interactively clip branches """

    clipper = vmtkscripts.vmtkSurfaceClipper()
    clipper.Surface = surf
    clipper.WidgetType = 'box'
    clipper.CleanOutput = 1
    clipper.Interactive = 1
    clipper.Execute()
    surf = pv.wrap(clipper.Surface)
    return surf

def centerlines(surf, seed_selector='pickpoint', resampling=1, 
                resampling_step_length=0.05, smoothing=True, 
                iterations=100, sm_factor=0.1, src_ids=[], 
                target_ids=[], src_pts=[], target_pts=[]):
    """ Generate centerlines from a surface with open profiles.

    Before generating the centerlines, the surface is slightly 
    perturbed to avoid a bug in vmtkCenterlines that occurs
    with perfectly straight flow extensions.
    """

    surf = surf.compute_normals()
    surf_perturb = surf.copy()
    perturbed_vec = np.einsum(
        'ij,i->ij', 
        surf_perturb.point_arrays['Normals'], 
        np.random.normal(0, 0.01, surf_perturb.n_points)
        )
    surf_perturb.points = surf_perturb.points + perturbed_vec
    centerline_filt = vmtkscripts.vmtkCenterlines()
    centerline_filt.Surface = surf
    centerline_filt.SeedSelectorName = seed_selector
    centerline_filt.Resampling = resampling
    centerline_filt.ResamplingStepLength = resampling_step_length
    centerline_filt.smoothing = smoothing
    centerline_filt.iterations = iterations
    centerline_filt.factor = sm_factor
    centerline_filt.SourceIds = src_ids
    centerline_filt.TargetIds = target_ids
    centerline_filt.SourcePoints = src_pts
    centerline_filt.TargetPoints = target_pts
    centerline_filt.Execute()
    centerlines = centerline_filt.Centerlines
    return pv.wrap(centerlines)

def centerline_branches_ids(centerlines):
    """ Identify centerline branches. """

    brancher = vmtkscripts.vmtkBranchExtractor()
    brancher.Centerlines = centerlines
    brancher.RadiusArrayName = utils.radiusArrayName
    brancher.Execute()
    centerlines_branched = brancher.Centerlines
    return pv.wrap(centerlines_branched)

def centerline_endpoint_extractor(centerlines, num_endpoint_spheres=1, num_gap_sphere=1):
    """ Wrapper for vmtkEndpointExtractor """

    extractor = vmtkscripts.vmtkEndpointExtractor()
    extractor.Centerlines = centerlines
    extractor.RadiusArrayName = utils.radiusArrayName
    extractor.GroupIdsArrayName = utils.groupIDsArrayName
    extractor.BlankingArrayName = utils.blankingArrayName
    extractor.NumberOfEndPointSpheres = num_endpoint_spheres
    extractor.NumberOfGapSpheres = num_gap_sphere
    extractor.Execute()
    clipped_centerlines = extractor.Centerlines

    return pv.wrap(clipped_centerlines)

def centerline_endpoint_masking_old(centerlines):
    """ Identify centerline endpoints.
    Don't use, will be deleted.
    """ 
    print('Warning: using OLD centerline endpoint masking')
    groups = np.unique(centerlines.cell_arrays['CenterlineIds'])
    ind = [np.where(centerlines.cell_arrays['CenterlineIds']==g) for g in groups]
    clines = [centerlines.extract_cells(i) for i in ind]
    end_arrays = [np.zeros(c.n_cells) for c in clines]
    for e in end_arrays:
        e[0] = 1
        e[-1] = 1
    centerlines.cell_arrays['EndCells'] = np.zeros(centerlines.n_cells) 
    for i, e in zip(ind, end_arrays):
        centerlines.cell_arrays['EndCells'][i] = e

    return centerlines 

def centerline_endpoint_masking(centerlines, central_group_id=2):
    """ Identify centerline endpoints using GroupId.
    
    Marks centerline endpoints via a masking array 'EndCells'.
    """
    centerlines.cell_arrays['EndCells'] = np.zeros_like(centerlines.cell_arrays['GroupIds'])
    centerlines.cell_arrays['EndCells'] = [1 if x !=central_group_id else 0 for x in centerlines.cell_arrays['GroupIds']]
    return centerlines 


def centerline_branch_clipper_checker(surf, centerlines):
    """ Check endpoints match with open profiles.

    In some cases, VMTK may mistakenly mark a region as an 
    endpoint if it's nearby. This double checks and fixes 
    the masking if present.
    """
    groups = np.unique(surf.point_arrays['GroupIds'])

    # Break surf into components based on GroupIds
    surf_section_idx = [np.where(surf.point_arrays['GroupIds'] == g)[0] for g in groups]
    line_section_idx = [np.where(centerlines.cell_arrays['GroupIds'] == g)[0] for g in groups]
    
    surf_sections = [surf.extract_points(i) for i in surf_section_idx]
    line_sections = [centerlines.extract_cells(i) for i in line_section_idx]
    line_end_bool = np.array([np.any(l.cell_arrays['EndCells'] == 1) for l in line_sections])

    end_sections = [x for b, x in zip(line_end_bool, surf_sections) if b == True]
    end_lines = [x for b, x in zip(line_end_bool, line_sections) if b == True]

    central_sections = [x for b, x in zip(line_end_bool, surf_sections) if b == False]
    central_lines = [x for b, x in zip(line_end_bool, line_sections) if b == False]

    central_group_id = np.median([x.point_arrays['GroupIds'][0] for x in central_sections]).astype(int)

    # Iterate through components, break into disconnected regions
    for s, l in zip(end_sections, end_lines):
        s = s.connectivity()
        s.point_arrays['vtkOGIds'] = s.point_arrays['vtkOriginalPointIds'].copy()

        component_ids = np.unique(s.cell_arrays['RegionId'])
        if len(component_ids) > 1:
            ind = [np.where(s.cell_arrays['RegionId'] == c)[0] for c in component_ids]
            comps = [s.extract_cells(i) for i in ind]

            # Find which region is closest to centerline
            dists = [np.linalg.norm(c.points.mean(axis=0) - l.points.mean(axis=0)) for c in comps]
            nearest_comp = np.argmin(dists)

            # Use remaining comps to un-mask the group id of that section
            remaining_comps = [x for x in list(range(len(comps))) if x != nearest_comp]

            # Find out what they will join to 
            for r in remaining_comps:
                surf.point_arrays['GroupIds'][comps[r].point_arrays['vtkOGIds']] = central_group_id

    # Unfinished -- iter through central sections
    # If they contain subcomps, assign to end that connects

    surf.point_arrays['EndCells'] = np.zeros(surf.n_points)
    surf.point_arrays['EndCells'][surf.point_arrays['GroupIds'] != 2] = 1
    return surf, centerlines

def get_centerline_endpoints(centerlines):
    """ Get terminal points of centerlines. """

    groups = np.unique(centerlines.cell_arrays['CenterlineIds'])
    ind = [np.where(centerlines.cell_arrays['CenterlineIds']==g) for g in groups]
    clines = [centerlines.extract_cells(i) for i in ind]
    clines = [c.ctp() for c in clines]

    origins = [] #np.zeros((2*len(clines), 3))

    for idx in range(len(clines)):
        if idx == 0:
            # REMOVE DUPLICATE ORIGINS
            cut_start_origin = clines[idx].points[0]
            origins.append(cut_start_origin)

        cut_end_origin = clines[idx].points[-1]
        origins.append(cut_end_origin)

    return origins

def get_centerline_endpoints_clip(centerlines):
    """ Get origin and approx normals of endpoints. """

    groups = np.unique(centerlines.cell_arrays['CenterlineIds'])
    ind = [np.where(centerlines.cell_arrays['CenterlineIds']==g) for g in groups]
    clines = [centerlines.extract_cells(i) for i in ind]
    clines = [c.ctp() for c in clines]

    origins = [] #np.zeros((2*len(clines), 3))
    normals = [] #np.zeros((2*len(clines), 3))

    for idx in range(len(clines)):
        clines[idx].point_arrays['EndPoints'] = np.round(clines[idx].point_arrays['EndCells']).astype(int)
        cut_mask = np.abs(np.diff(clines[idx].point_arrays['EndPoints']))
        cut_index = np.where(cut_mask)[0]
        # Get the point before/after
        cut_start_index = [cut_index[0], cut_index[0]-5]
        cut_end_index = [cut_index[1], cut_index[1]+5]

        cut_start_normal = np.diff(clines[idx].points[cut_start_index], axis=0)
        cut_end_normal = np.diff(clines[idx].points[cut_end_index], axis=0)

        cut_start_origin = clines[idx].points[cut_start_index[0]]
        cut_end_origin = clines[idx].points[cut_end_index[0]]

        origins.append(cut_start_origin)
        origins.append(cut_end_origin)
        normals.append(cut_start_normal[0])
        normals.append(cut_end_normal[0])

    # inlet = clines[0].points[0]
    # outlets = [c.points[-1] for c in clines]
    return origins, normals #, outlets

def centerline_branch_clipper(surf, centerlines):
    """ Clip branches using centerlines with VMTK. """
    clipper = vmtkscripts.vmtkBranchClipper()
    clipper.Surface = surf
    clipper.Centerlines = centerlines
    clipper.ClipValue = 0.0
    clipper.RadiusArrayName = utils.radiusArrayName
    clipper.GroupIdsArrayName = utils.groupIDsArrayName
    clipper.BlankingArrayName = utils.blankingArrayName
    clipper.UseRadiusInformation = 1
    clipper.Execute()
    surf = pv.wrap(clipper.Surface)
    clines = pv.wrap(clipper.Centerlines)
    return surf, clines 

def surface_connectivity(surf, group_ids_name=utils.groupIDsArrayName, group_id=2):
    """ Get largest connected region. """

    connector = vmtkscripts.vmtkSurfaceConnectivity()
    connector.Surface = surf
    connector.CleanOutput = 1
    connector.GroupIdsArrayName = group_ids_name
    connector.GroupId = group_id
    connector.Execute()
    surf = pv.wrap(connector.Surface)
    return surf

def flow_extensions(surf, centerlines):
    """ Add flow extensions. """

    extender = vmtkscripts.vmtkFlowExtensions()
    extender.Surface = surf
    extender.Centerlines = centerlines
    extender.AdaptiveExtensionLength = 1
    extender.ExtensionRatio = 4
    extender.CenterlineNormalEstimationDistanceRatio = 1
    extender.Interactive = 0
    extender.ExtensionMode = 'boundarynormal'
    extender.Execute()
    surf = pv.wrap(extender.Surface)
    centerlines = pv.wrap(extender.Centerlines)
    return surf, centerlines

def surface_remeshing(surf, edgelength=0.3, element_size_mode='edgelength', edgearray='Size', iterations=10):
    """ Surface remeshing with VMTK.
    
    If using element_size_mode='edgelengtharray', specify edgearray.
    """
    remesher = vmtkscripts.vmtkSurfaceRemeshing()
    remesher.Surface = surf
    remesher.ElementSizeMode = element_size_mode
    remesher.TargetEdgeLengthArrayName = edgearray
    remesher.Iterations = iterations
    remesher.TargetEdgeLength = edgelength
    remesher.Execute()
    surf = pv.wrap(remesher.Surface)
    return surf 

def assert_all_quads(mesh):
    """ Remove triangles from mesh.

    VMTK's volume mesh will have a surface of triangles.
    This function removes them and returns the quad-only mesh.
    """

    cell_types = np.zeros(mesh.n_cells)
    cells = mesh.cells

    i = 0
    idx = 0
    while i < len(mesh.cells):
        cell_types[idx] = cells[i]
        plus = cells[i]
        i = i + plus  + 1
        idx += 1
    
    quad_mask = np.array(cell_types) == 4
    mesh_quad = mesh.extract_cells(quad_mask)
    return mesh_quad

def volume_meshing(surf):
    """ VMTK volume meshing.
    
    More input options will be added.
    """

    meshgen = vmtkscripts.vmtkMeshGenerator()
    meshgen.Surface = surf
    meshgen.ElementSizeMode = "edgelengtharray"
    meshgen.TargetEdgeLengthArrayName = "Size"
    meshgen.BoundaryLayer = 1
    meshgen.NumberOfSubLayers = 4
    meshgen.BoundaryLayerOnCaps = 0
    meshgen.BoundaryLayerThicknessFactor = 0.85
    meshgen.SubLayerRatio = 0.75
    meshgen.Tetrahedralize = 1
    meshgen.VolumeElementScaleFactor = 0.8
    meshgen.EndcapsEdgeLengthFactor = 1.0
    meshgen.Execute()
    mesh = pv.wrap(meshgen.Mesh)
    return mesh

def bifurcation_ref_systems(centerlines):
    """ Get bifurcation reference systems. """ 

    ref = vmtkscripts.vmtkBifurcationReferenceSystems()
    ref.Centerlines = centerlines
    ref.RadiusArrayName = utils.radiusArrayName 
    ref.Execute()
    return pv.wrap(ref.ReferenceSystems)

def get_bifurcation_vectors(centerlines, ref_systems):
    """ Get bifurcation vectors. """

    bifvec = vmtkscripts.vmtkBifurcationVectors()
    bifvec.Centerlines = centerlines
    bifvec.ReferenceSystems = ref_systems
    bifvec.Execute()
    bif_vectors = pv.wrap(bifvec.BifurcationVectors)
    bif_group_ids = np.unique(bif_vectors.point_arrays['BifurcationGroupIds'])
    bif_groups = []
    for bb in bif_group_ids:
        bif_ind = np.where(bif_vectors.point_arrays['BifurcationGroupIds']==bb)
        bif_groups.append(bif_vectors.extract_points(bif_ind))

    return pv.MultiBlock(bif_groups)

def distance_to_centerlines(surf, centerlines, use_radius=1, project_point_arrays=0):
    """ Get distance between surf and centerlines. """

    dist = vmtkscripts.vmtkDistanceToCenterlines()
    dist.Surface = surf
    dist.Centerlines = centerlines 
    dist.UseRadiusInformation = use_radius
    dist.EvaluateTubeFunction = 0
    dist.EvaluateCenterlineRadius = 0 
    dist.UseCombinedDistance = 0
    dist.ProjectPointArrays = project_point_arrays
    dist.DistanceToCenterlinesArrayName = utils.distanceToCenterlinesArrayName
    dist.RadiusArrayName = utils.radiusArrayName
    dist.Execute()
    return pv.wrap(dist.Surface), pv.wrap(dist.Centerlines)

def surface_centerline_projection(surf, centerlines, pass_arrays=None):
    """ Project centerline arrays onto surface """

    c2 = centerlines.ctp()
    if pass_arrays is not None:
        pass_arrays.append('MaximumInscribedSphereRadius')
        arrs = [c2.point_arrays[x] for x in pass_arrays]
        pts = c2.points 
        cells = c2.lines 
        c2 = pv.PolyData(pts, lines=cells)
        
        for a, x in zip(arrs, pass_arrays):
            c2.point_arrays[x] = a

    proj = vmtkscripts.vmtkSurfaceCenterlineProjection()
    proj.Surface = surf
    proj.Centerlines = c2
    proj.UseRadiusInformation = 1
    proj.RadiusArrayName = utils.radiusArrayName
    proj.Execute()
    return pv.wrap(proj.Surface)

def kite_removal(surf, factor=0.1):
    kite = vmtkscripts.vmtkSurfaceKiteRemoval()
    kite.Surface = surf 
    kite.SizeFactor = factor 
    kite.Execute()
    return pv.wrap(kite.Surface)

def network_extractor(surf):
    """ Extract a basic network and graph layout of a surfaces. """

    ext = vmtkscripts.vmtkNetworkExtraction()
    ext.Surface = surf 
    ext.AdvancementRatio = 1.05
    ext.RadiusArrayName = 'Radius'
    ext.TopologyArrayName = 'TopologyArrayName'
    ext.MarksArrayName = 'MarksArrayName'
    ext.Execute()

    return pv.wrap(ext.Network), pv.wrap(ext.GraphLayout)

def surface_capper(surf):
    """ Add caps to surface. """

    capper = vmtkscripts.vmtkSurfaceCapper()
    capper.Surface = surf
    capper.Method ='centerpoint'
    capper.Interactive = 0
    capper.Execute()
    return pv.wrap(capper.Surface)

def renumber_entity_ids(mesh, inlet_id):
    """ Renumber entity ids
    -------- NOT YET FUNCTIONAL ---------

    Default is the following:
    0 is internal cells
    1 is wall
    2 is inlet
    3, ... is outlets
    """
    entity_ids = mesh.cell_arrays['CellEntityIds'].copy()
    valid_entity_ids = sorted(set(np.unique(entity_ids)) - set([0,1]))

    entity_ids_renumbered = np.zeros_like(entity_ids)
    
    pairs = [[0,0],[1,1],[4,2],[2,4],[3,3]]
    for p in pairs:
        entity_ids_renumbered[entity_ids == p[0]] = p[1]
    mesh.cell_arrays['CellEntityIds'] = entity_ids_renumbered
    return mesh 

def branch_center_normal_rad_area(mesh):
    """ Get center, normal, rad, and area of in/outlets.
    """

    surf = mesh.extract_surface()
    surf = surf.compute_normals(cell_normals=False, point_normals=True,)

    entity_ids = np.unique(surf.cell_arrays['CellEntityIds'])
    
    # entity_id = 0, 1 is internal, wall
    entity_ids = sorted(set(entity_ids) - set([0, 1]))

    points = {}

    def _f(estimate, *real_points):
        real_points = np.array(real_points).reshape(-1,3)
        distance = np.linalg.norm(real_points - estimate, axis=1)
        return distance 

    for e in entity_ids:
        # print('e', e)
        mask = surf.cell_arrays['CellEntityIds'] == e
        cap = surf.extract_cells(mask)
        edges = cap.extract_feature_edges(
            boundary_edges=True, 
            feature_edges=False, 
            manifold_edges=False
            )
        cap = cap.compute_cell_sizes()
        cap_mask = [True if cap.points[idx] not in edges.points else False for idx in range(cap.n_points)]

        cap_pts = cap.extract_points(cap_mask, adjacent_cells=False)        
        # cap_pts = cap_pts.compute_cell_sizes()

        # cap_pts_ALT = cap.extract_points(cap_mask, adjacent_cells=True)        
        # cap_pts_ALT = cap_pts_ALT.compute_cell_sizes()

        center = edges.points.mean(axis=0)

        center_robust_ = least_squares(_f, center, args=edges.points)
        center_robust = center_robust_.x
        radius_robust = center_robust_.fun.mean()


        normal = cap_pts.point_arrays['Normals'].mean(axis=0).reshape(1,-1)
        normal = normal / np.linalg.norm(normal)
        area = cap.cell_arrays['Area'].sum()

        # rad = np.linalg.norm(edges.points - center, axis=1).mean()
        # area = np.pi*rad**2
        # rad2 = (area / np.pi)**0.5
        # print('rad', rad.mean())
        # print('rad2', rad2)
        # print('rad_robust', radius_robust)

        c = pv.wrap(center)
        c.point_arrays['Normal'] = normal
        c.point_arrays['CellEntityIds'] = e
        c.point_arrays['Area'] = area 
        c.point_arrays['Radius'] = radius_robust

        points[e] = c
        
    return points



def sac_mask_tool(surf, centerlines):
    """ ID sac based on centerline branches.

    Sac must be last branch.
    Not robust; only allows 1 sac.
    """
    surf = surface_centerline_projection(surf, centerlines)

    branch_ids = np.unique(surf.point_arrays['GroupIds'])
    sac_idx = branch_ids[-1]
    mask = [0 if sac_idx != x else 1 for x in surf.point_arrays['GroupIds']]
    surf.point_arrays['SacMask'] = mask
    return surf, centerlines 

def extract_group_adjacency(centerlines):
    """ Get adj list of group IDs.

    Note: centerlines must not be branched, must 
    have 1 cell per endpoint (the default format for 
    fresh centerlines).
    """
    centerlines_og = centerlines.copy()
    centerlines = centerline_branches_ids(centerlines)

    # Convert cell data to point data
    centerlines = centerlines.ctp()

    # Create search tree for centerlines with GroupIds
    tree = KDTree(centerlines.points)

    # Prepare points from original structure to query 
    q_points = centerlines_og.points
    dd, ii = tree.query(q_points, k=1)

    # Project onto centerlines_og
    centerlines_og.point_arrays['GroupIds'] = np.round(centerlines.point_arrays['GroupIds'][ii]).astype(int)
    centerlines_og.point_arrays['Blanking'] = np.round(centerlines.point_arrays['Blanking'][ii]).astype(int)

    mask = np.invert(centerlines_og.point_arrays['Blanking'].astype(bool))
    valid_ids = np.unique(centerlines_og.point_arrays['GroupIds'][mask])

    # Split by endpoint
    centerlines_split = centerlines_og.split_bodies()

    # Get adj list
    edges = []
    G = nx.DiGraph()
    for cline in centerlines_split:
        groups = np.unique([x for x in cline.point_arrays['GroupIds'] if x in valid_ids])
        pairs = [(groups[i-1], groups[i]) for i in range(1, len(groups))]
        for p in pairs:
            edges.append(p)
            G.add_edge(p[0], p[1])
    
    edges = sorted(set(edges))

    return edges, G

def write_mesh(mesh, outfile):
    """ Write mesh using VMTK.

    Used for writing dolfin-format XML with
    cellEntityIds array.
    """
    print('New way')

    writer = vmtkscripts.vmtkMeshWriter()
    writer.Mesh = mesh 
    writer.Format = 'dolfin'
    writer.Compressed = 0
    writer.Mode = 'binary'
    writer.CellEntityIdsArrayName = 'CellEntityIds'
    writer.OutputFileName = str(outfile)
    writer.Execute()

    # Then open the file and gzip it
    gz_outfile = str(outfile) + '.gz'

    with open(outfile, 'rb') as orig_file:
        with gzip.open(gz_outfile, 'wb') as zipped_file:
            zipped_file.writelines(orig_file)

def surface_array_smoothing(surf, array_name='Size', connexity=1, relaxation=1.0, iterations=1):
    sm = vmtkscripts.vmtkSurfaceArraySmoothing()
    sm.Surface = surf 
    sm.SurfaceArrayName = array_name
    sm.Connexity = connexity 
    sm.Relaxation = relaxation 
    sm.Iterations = iterations
    sm.Execute()
    return pv.wrap(sm.Surface)
