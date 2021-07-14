import pyvista as pv 
import numpy as np
from geometry_tools import vmtk_wrapper as vmtk
from geometry_tools.surface import Surfer

from pathlib import Path 
from geometry_tools import common as cc 
from scipy.spatial import cKDTree as KDTree 
import h5py 

class Mesher(Surfer):
    """ Basic meshing tools based on VMTK. 

    Attributes:
        surf : surface PolyData or None 
        mesh : volumetric UnstructuredGrid or None
        inlet_points : array of inlet points
        outlet_points : array of outlet points
        centerlines : PolyData centerlines of surf
        centerlines_branched : PolyData centerlines of surf with branch IDs
    """

    def __init__(self, surf=None, mesh=None, inlet_points=None, outlet_points=None, aneurysm_points=None):
        """ Init the mesher instance.

        Either surf or mesh must be given. If mesh, surf will be extracted 
        based on CellEntityIds.
        """
        self.mesh = mesh
        super().__init__(surf=surf, 
            inlet_points=inlet_points, 
            outlet_points=outlet_points, 
            aneurysm_points=aneurysm_points)

        if (self.mesh is not None) and (self.surf is None):
            # Extract surface using entity ids
            surf_mask = self.mesh.cell_arrays['CellEntityIds'] == 1
            surf = self.mesh.extract_cells(surf_mask)
            surf_pt_ids = surf.point_arrays['vtkOriginalPointIds']
            surf = pv.PolyData(surf.points, surf.cells)
            surf.point_arrays['vtkOriginalPointIds'] = surf_pt_ids
            self.surf = surf.compute_normals()

    def set_inlets_outlets(self):
        """ Interactively choose inlet point.

        All other open boundaries will be considered outlets. To update these points 
        (for example, after adding flow extensions), see self.update_inlet_outlets. 
        To save these inlet-outlet points, see self.save_inlet_outlet_points()
        """
        centers_m, inlet_ids, outlet_ids = Surfer.set_inlets_outlets(self)
        if self.mesh is not None:
            self._set_inlet_outlet_entity_ids(centers_m, inlet_ids, outlet_ids)

    def update_inlets_outlets(self):
        """ Update inlet-outlet points based on distance metric.
        """
        centers_m, inlet_ids, outlet_ids = Surfer.update_inlets_outlets(self)
        if self.mesh is not None:
            self._set_inlet_outlet_entity_ids(centers_m, inlet_ids, outlet_ids)

    def _set_inlet_outlet_entity_ids(self, centers_m, inlet_ids, outlet_ids):    
        """ Internal method for keeping track of CellEntityIds.

        Args:
            centers_m : PolyData containing points centered on open profiles
            inlet_ids : list index of inlet ids
            outlet_ids : list of index of outlet ids
        
        Sets:
            inlet_entity_ids : list of inlet entity ids
            outlet_entity_ids : list of outlet entity ids

        """
        entity_ids = np.unique(self.mesh.cell_arrays['CellEntityIds'])
        valid_entity_ids = sorted(set(entity_ids) - set([0, 1]))
        entity_masks = [self.mesh.cell_arrays['CellEntityIds'] == i for i in valid_entity_ids]
        caps = [self.mesh.extract_cells(em) for em in entity_masks]
        cap_centers = [c.points.mean(axis=0) for c in caps]

        tree = KDTree(cap_centers)
        inlet_ids = [tree.query(i)[1] for i in self.inlet_points]
        outlet_ids = list(set(range(centers_m.n_points)) - set(inlet_ids))
        
        self.inlet_entity_ids = [valid_entity_ids[x] for x in inlet_ids]
        self.outlet_entity_ids = [valid_entity_ids[x] for x in outlet_ids]

        if hasattr(self, 'centerlines_branched'): 
            # Match inlet/outlet points with a group id
            c_branch = self.centerlines_branched.copy()
            c_branch = c_branch.ctp()

            tree = KDTree(c_branch.points)
            ii = [tree.query(pt, k=1)[1] for pt in self.inlet_points]
            ei = [tree.query(pt, k=1)[1] for pt in self.outlet_points]
            group_ids_in = [c_branch.point_arrays['GroupIds'][i] for i in ii]
            group_ids_out = [c_branch.point_arrays['GroupIds'][i] for i in ei]
        
            self.inlet_group_ids = group_ids_in
            self.outlet_group_ids = group_ids_out

    def surface_preparation(self,):
        """ Refine surface, add flow extensions. 

        Remeshes surface, clips endpoints normal to centerlines,
        adds flow extensions, creates size array based on local radius
        and curvature, then does a final remshing and cleaning.
        """
        # VMTK uses a weird dtype sometimes, but recreating the surface 
        # from the points and cells fixes this issue. Need to copy 
        # array "Mask" though.
        surf_og = pv.PolyData()
        surf_og.copy_structure(self.surf)
        surf_og.point_arrays['Mask'] = self.surf.point_arrays['Mask']
        surf_og.point_arrays['GroupIds'] = self.surf.point_arrays['GroupIds']

        surf = pv.PolyData(self.surf.points, self.surf.faces)
        surf.point_arrays['Mask'] = surf_og.point_arrays['Mask'].copy()
        surf = surf.clean()

        # Use centerlines without aneurysm
        centerlines = self.centerlines

        # centerlines = vmtk.centerline_endpoint_extractor(
        #     centerlines, num_endpoint_spheres=0, num_gap_sphere=0)
        # centerlines = vmtk.centerline_endpoint_masking(centerlines)

        surf, centerlines = vmtk.distance_to_centerlines(surf, centerlines)
        surf = cc.create_edge_size_array(surf, max_size=0.4, min_size=0.14,)
        surf = vmtk.surface_remeshing(surf, element_size_mode='edgelengtharray', edgearray='Size')

        # surf = vmtk.surface_centerline_projection(
        #     surf, centerlines, pass_arrays=['EndCells', 'GroupIds'])
        
        # surf, centerlines = vmtk.centerline_branch_clipper_checker(surf, centerlines)
        # surf = vmtk.surface_connectivity(surf, group_ids_name='EndCells', group_id=0)

        surf, centerlines = vmtk.flow_extensions(surf, centerlines)
        surf = surf.clean()
        
        self.update_inlets_outlets()
        self.generate_centerlines(include_aneurysms=False)
        centerlines = self.centerlines

        surf = surf.interpolate(surf_og, radius=0.5)
        surf, centerlines = vmtk.distance_to_centerlines(surf, centerlines)
        surf = cc.create_edge_size_array(surf, max_size=0.4, min_size=0.14,)

        # surf, n_ids = cc.smooth_mesh_data_local(surf, 'Size', np.min, iterations=2)
        surf, n_ids = cc.smooth_mesh_data_local(surf, 'Size', np.mean, 
            iterations=1,
            )
        
        surf_rm = vmtk.surface_remeshing(surf, element_size_mode='edgelengtharray', edgearray='Size')
        surf_rm = surf_rm.clean()

        surf_rm = surf_rm.interpolate(surf_og, radius=0.5)
        surf_rm = surf_rm.interpolate(surf, radius=0.5)

        self.surf = surf_rm 
        self.centerlines = centerlines

    def select_refinement_regions(self):
        """ Interactively choose refinement region.

        See docs of geometry_tools.common.SacSelectTool for more info.
        """
        select = cc.SacSelectTool(self.surf)
        select.select()
        self.surf = select.surf 

    def mark_refinement_regions(self):
        """ Automatically mark refinement region based on GroupIds.
        To mark manually, use select_refinement_regions.
        """
        self.surf.point_arrays['Mask'] = np.zeros(self.surf.n_points)
        for an_id in self.aneurysm_group_ids:
            mask = self.surf.point_arrays['GroupIds'] == an_id
            self.surf.point_arrays['Mask'][mask] = 1

    def refine_picked_regions(self):
        """ Update point array "Size" of surf using chosen refinement regions.
        """
        surf = self.surf 
        centerlines = self.centerlines_aneurysm

        surf, centerlines = vmtk.distance_to_centerlines(surf, centerlines)
        surf = cc.create_edge_size_array(surf, max_size=0.4, min_size=0.18,)
        surf.point_arrays['Size'][surf.point_arrays['Mask'] > 0.99] = 0.18
        # surf.point_arrays['Size'] = cc.smooth_mesh_data(surf.point_arrays['Size'], surf.points, 1.0, func=np.mean)
        surf = vmtk.surface_array_smoothing(surf, array_name='Size', iterations=5)
        surf.set_active_scalars('Mask')
        
        self.surf = surf
        self.centerlines_aneurysm = centerlines

    def generate_flow_rates_legacy(self):
        """ Generate flow rates using Christophe's old code.
        """
        from geometry_tools.legacy import networks, network_boundary_conditions
        
        network = networks.Network()
        centerlinesBranches = networks.SetNetworkStructure(self.centerlines, network, print,
            isConnectivityNeeded=True)

        flowSplitting = network_boundary_conditions.FlowSplitting()
        flowSplitting.ComputeAlphas(network, lambda x: None)
        flowSplitting.ComputeBetas(network, lambda x: None)
        flowSplitting.CheckTotalFlowRate(network, lambda x: None)

        betas = {}
        inlet = {}

        for element in network.elements:
            if element.IsAnOutlet():
                networkPoint = element.GetOutPointsx1()[0]
                betas[element.vtkGroupIdList[0]] = element.GetBeta()

            if  element.IsAnInlet():
                inlet[element.vtkGroupIdList[0]] = element.GetInletRadius()

        print('betas \n', betas)

        new_centerlines = pv.wrap(centerlinesBranches)

        for b in betas.keys():
            mask = new_centerlines.cell_arrays['GroupIds'] == b 
            branch_segments = new_centerlines.extract_cells(mask)
            branch = branch_segments.split_bodies()[0]
            radius = branch.point_arrays['MaximumInscribedSphereRadius']

    def generate_flow_rates(self):
        """ Get flow rates based on Christophe method.

        Gives equivalent values as legacy call, just rewritten.
        The centerlines should be passed before calling centerline branch ids.

        WARING! outlet flow divisions is not indexed by CellEntityIds. Fix!
        Need to match G.nodes to a CellEntityId value.
        """ 
        self.get_group_adjacency()

        nodes = [x for x in self.G_no_aneurysm.nodes] 

        centerlines_branched = vmtk.centerline_branches_ids(self.centerlines)
        self.centerlines_branched = centerlines_branched
        mean_radii = cc.get_mean_radii(centerlines_branched, nodes)
        beta_values = {}
        beta_values[0] = 1.

        outlet_flow_divisions = {}

        for node in nodes:
            children = [x for x in self.G_no_aneurysm.successors(node)]
            if len(children) > 0:
                node_radii = mean_radii[node]
                children_radii = np.array([mean_radii[x] for x in children])
                children_sum = np.sum(children_radii**2)
                children_ratio = children_radii**2 / children_sum

                for cdx, ch in enumerate(children):
                    beta_values[ch] = children_ratio[cdx]*beta_values[node]
        
            else:
                outlet_flow_divisions[node] = beta_values[node]

        print('\n' + 'Outlet_flow_divisions', outlet_flow_divisions)
        self.outlet_flow_divisions = outlet_flow_divisions

    def generate_volume_mesh(self):
        mesh = vmtk.volume_meshing(self.surf)
        self.mesh = mesh


    def generate_h5_file(self, outfile):
        """ Generate lab-specific h5 mesh file for our solver.

        File layout:
            - Mesh
                - coordinates
                - topology
                - ID_N
                    - cellIds, coordinates, pointIds, topology
                - Wall
                    - cellIds, coordinates, normal, pointIds, topology
        """
        case_name = outfile.stem

        # Quads only
        mesh_quad = vmtk.assert_all_quads(self.mesh)

        # Get surf
        surf_mask = self.mesh.cell_arrays['CellEntityIds'] == 1
        surf = self.mesh.extract_cells(surf_mask)
        surf_pt_ids = surf.point_arrays['vtkOriginalPointIds']
        surf = pv.PolyData(surf.points, surf.cells)
        surf.point_arrays['vtkOriginalPointIds'] = surf_pt_ids
        surf = surf.compute_normals()

        # Wall points
        wall_coordinates = surf.points 
        wall_pointIds = surf.point_arrays['vtkOriginalPointIds']
        wall_topology = surf.faces.reshape(-1,4)[:,1:]
        wall_cellIds = wall_pointIds[wall_topology]
        wall_normals = surf.point_arrays['Normals']
        
        # Extract caps        
        entity_ids = np.unique(self.mesh.cell_arrays['CellEntityIds'])
        valid_entity_ids = sorted(set(entity_ids) - set([0, 1]))
        entity_masks = [self.mesh.cell_arrays['CellEntityIds'] == i for i in valid_entity_ids]
        caps = [self.mesh.extract_cells(em) for em in entity_masks]

        # Make sure points are duplicate
        tree = KDTree(mesh_quad.points)
        inds = [tree.query(c.points, k=1)[1] == 0 for c in caps]
        dists = np.all([np.all(tree.query(c.points, k=1)[0] == 0) for c in caps])
        assert dists, 'Cap points are not in quad mesh'

        # Get relevant cap quantities
        caps_coordinates = [c.points for c in caps]
        caps_pointIds = [c.point_arrays['vtkOriginalPointIds'] for c in caps]
        caps_topology = [c.cells.reshape(-1,4)[:,1:] for c in caps]
        caps_cellIds = [c.point_arrays['vtkOriginalPointIds'][c.cells.reshape(-1,4)[:,1:]] for c in caps]

        # Now create dataset
        f = h5py.File(outfile, "w")
        
        f.create_dataset('Mesh/coordinates', 
            data=mesh_quad.points, 
            compression="gzip", 
            compression_opts=9
            )
        f.create_dataset('Mesh/topology', 
            data=mesh_quad.cells.reshape(-1,5)[:,1:], 
            compression="gzip", 
            compression_opts=9
            )
        
        f.create_dataset('Mesh/Wall/coordinates', 
            data=wall_coordinates, 
            compression="gzip", 
            compression_opts=9
            )
        f.create_dataset('Mesh/Wall/pointIds', 
            data=wall_pointIds, 
            compression="gzip", 
            compression_opts=9
            )
        f.create_dataset('Mesh/Wall/topology', 
            data=wall_topology, 
            compression="gzip", 
            compression_opts=9
            )
        f.create_dataset('Mesh/Wall/cellIds', 
            data=wall_cellIds, 
            compression="gzip", 
            compression_opts=9
            )
        f.create_dataset('Mesh/Wall/normal', 
            data=wall_normals, 
            compression="gzip", 
            compression_opts=9
            )
        
        for cdx in range(len(caps)):
            f.create_dataset('Mesh/ID_{}/coordinates'.format(cdx + 1), 
                data=caps_coordinates[cdx], 
                compression="gzip", 
                compression_opts=9
                )
            f.create_dataset('Mesh/ID_{}/pointIds'.format(cdx + 1), 
                data=caps_pointIds[cdx], 
                compression="gzip", 
                compression_opts=9
                )
            f.create_dataset('Mesh/ID_{}/topology'.format(cdx + 1), 
                data=caps_topology[cdx], 
                compression="gzip", 
                compression_opts=9
                )
            f.create_dataset('Mesh/ID_{}/cellIds'.format(cdx + 1), 
                data=caps_cellIds[cdx], 
                compression="gzip", 
                compression_opts=9
                )

        f.close()

    def generate_info_file(self, outfile, inlet_vel=0.27, waveform='FC_MCA_10'):
        """ Generate lab-specific info file.

        Formats a bunch of values into strings, including 
        area, radius, flow divisions, etc of inlet/outlet
        """
        # Get branch center, normal, rad, area
        center_points = vmtk.branch_center_normal_rad_area(self.mesh)
     
        # Match GroupIds from self.centerlines_branched to CellEntityIds
        # Or match GroupIds with inlet points, inlets points is matched with entity ids

        # Extract caps   
        entity_ids = np.unique(self.mesh.cell_arrays['CellEntityIds'])
        valid_entity_ids = sorted(set(entity_ids) - set([0, 1]))
        entity_masks = [self.mesh.cell_arrays['CellEntityIds'] == i for i in valid_entity_ids]
        caps = [self.mesh.extract_cells(em) for em in entity_masks]

        info_data = {}

        entity_id_group_id_matcher = dict(zip(self.outlet_entity_ids, self.outlet_group_ids))

        for cdx in range(len(caps)):
            entity_id = valid_entity_ids[cdx]
            center = center_points[entity_id]

            c = center.points[0]
            normal = center.point_arrays['Normal'][0]
            rad = center.point_arrays['Radius'][0]
            area_ = center.point_arrays['Area'][0]

            c = (','.join("{:.12f}".format(x) for x in c))
            c = '(' + c + ')'
            normal = (','.join("{:.12f}".format(x) for x in normal))
            normal = '(' + normal + ')'
            rad = "{:.12f}".format(rad)
            area = "{:.12f}".format(area_)

            dataline = [str(cdx + 1), str(None), c, normal, rad, area]

            if entity_id in self.inlet_entity_ids:
                inflowrate = inlet_vel * area_
                dataline[1] = waveform
                dataline.append("{:.12f}".format(inflowrate))

            elif entity_id in self.outlet_entity_ids:
                # FIX! Need to match GroupIds with CellEntityIds
                flow_division = self.outlet_flow_divisions[entity_id_group_id_matcher[entity_id]]
                dataline.append("{:.12f}".format(flow_division))

            else:
                print('Check inlets outlets')

            info_data[entity_id] = dataline

        infofile = open(outfile, "w")
        header = "# id, wave, center, normal, radius, area, FR(inlet)/AR(outlet)"
        infofile.write(header + 2*"\n")
        infofile.write(outfile.stem + "\n")
        infofile.write("\n")
        infofile.write("<INLETS>" + "\n")

        for e_id in self.inlet_entity_ids:
            dataline = info_data[e_id]    
            dataline = '   '.join(dataline)
            infofile.write(dataline + "\n")

        infofile.write("\n")
        infofile.write("<OUTLETS>" + "\n")
        for e_id in self.outlet_entity_ids:
            dataline = info_data[e_id]    
            dataline = '   '.join(dataline)
            infofile.write(dataline + "\n")

        infofile.close()

    def generate_xml_gz_file(self, xml_gz_file):
        """ Generate xml file using VMTK.
        """
        xml_file = xml_gz_file.parents[0] / xml_gz_file.stem

        # mesh = vmtk.assert_all_quads(self.mesh)

        vmtk.write_mesh(self.mesh, xml_file)
      
if __name__ == "__main__":
    print('See example scripts directory')
