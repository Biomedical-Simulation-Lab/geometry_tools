""" For clipping tubes while maintaining other continuity
"""

import pyvista as pv 
import numpy as np 
import sys
from scipy.spatial import cKDTree as KDTree
import itertools 

class TubeClipper():
	def __init__(self, mesh):
		self.mesh = mesh
		self.mesh_surf = self.mesh.extract_surface()

		# mesh_clip is the initial clip
		self.mesh_clip = None

		# mesh_cc is the clip component connected to the origin
		# often, this is the sac
		self.mesh_cc = None

		# mesh_cc_ragged is the points within mesh_cc
		self.mesh_cc_ragged = None

		# mesh_cc_ragged_inv is the inverse of mesh_cc_ragged
		self.mesh_cc_ragged_inv = None

	def clip(self, normal, origin):
		# Determine the sign mask for each point relative 
		# to the clip origin
		vect = self.mesh.points - origin
		dot = np.dot(vect, normal)
		mask = np.zeros_like(dot)
		mask[dot>=0] = 1
		mask[dot<0] = 0

		# Assign the sign mask to the mesh
		# and convert to cell data
		self.mesh.point_arrays['mask'] = mask
		self.mesh = self.mesh.ptc()
		self.mesh.cell_arrays['mask'] = np.round(
			self.mesh.cell_arrays['mask'], 0).astype(int)

		# Split the mesh based on the sign mask
		# The *_vol meshes will be used to construct the output mesh
		# The other two are for visualization, but follow the same
		# logic.
		mesh_cc_ragged_vol = self.mesh.threshold(value=0.5, 
			scalars='mask', invert=False).split_bodies(label=False)
		mesh_cc_ragged = self.mesh.threshold(value=0.5, 
			scalars='mask', invert=False).extract_surface()

		mesh_cc_ragged_inv_vol = self.mesh.threshold(value=0.5, 
			scalars='mask', invert=True).split_bodies(label=False)
		mesh_cc_ragged_inv = self.mesh.threshold(value=0.5, 
			scalars='mask', invert=True).extract_surface()

		mesh_cc_ragged = mesh_cc_ragged.split_bodies(label=False)
		mesh_cc_ragged_inv = mesh_cc_ragged_inv.split_bodies(label=False)

		for m in mesh_cc_ragged_inv:
			mesh_cc_ragged.append(m)
	
		for m in mesh_cc_ragged_inv_vol:
			mesh_cc_ragged_vol.append(m)

		# Now get adjacency info
		# Check if they're adjacent by merging them, then
		# determinging the number of disconnected bodies
		# Probably a faster way to do this.
		adj = np.zeros((len(mesh_cc_ragged), len(mesh_cc_ragged)), dtype=int)

		combinations = list(itertools.combinations(range(len(mesh_cc_ragged)), r=2))
		connectivity = np.zeros(len(combinations))

		for idx, ids in enumerate(combinations):
			i1, i2 = ids
			c1 = mesh_cc_ragged.get([i1,i2]).combine(merge_points=True)
			c1_num_regions = np.unique(c1.connectivity()['RegionId'])

			if len(c1_num_regions) == 1:
				connectivity[idx] = 1
				adj[i1, i2] = 1
				adj[i2, i1] = 1
				

		# (c1_connectivity)

		# Determine which two blocks are closest to origin
		distances = []
		sidedness = []

		for idx, m in enumerate(mesh_cc_ragged):
			search_points = m.points
			tree = KDTree(search_points, leafsize=2)
			dist, ind = tree.query(origin, k=10) 
			dist = dist.sum() 
			# ind = np.unique(sorted(ind.flatten()))
			distances.append(dist)
			sidedness.append(np.median(m.cell_arrays['mask']))

		distances = np.array(distances).reshape(-1,1)
		sidedness = np.array(sidedness).reshape(-1,1)
		indicies = np.array(range(len(distances))).reshape(-1,1)
		keep = np.zeros_like(distances)

		# Sort the conditions by distances
		conditions = np.hstack([indicies, distances, sidedness, keep])
		conditions = conditions[conditions[:,1].argsort()]

		# Distance == 1 means closest to origin
		distances_mask = np.zeros_like(distances.flatten())
		distances_mask[:2] = 1

		conditions[:,1] = distances_mask

		# Where side == 1, where min distance --> sac, mark
		sac_mask = (conditions[:,1] == 1) & (conditions[:,2] == 1)
		anti_sac_mask = (conditions[:,1] == 1) & (conditions[:,2] == 0)

		# boolean to index
		sac_mask = np.array(list(range(len(sac_mask))))[sac_mask]
		sac_idx = conditions[:,0][sac_mask].astype(int)
		anti_sac_idx = conditions[:,0][anti_sac_mask].astype(int)

		# Disconnect in adj
		adj[sac_idx, anti_sac_idx] = 0
		adj[anti_sac_idx, sac_idx] = 0

		# Get adj list
		adj_list = []

		keep_idx = [sac_idx]
		other_idx = [anti_sac_idx]

		for ii in range(len(adj)):
			row = adj[ii, ii:]
			adj_list.append([idx + ii for idx, x in enumerate(row) if x !=0])
			# print(adj_list)

			if (sac_idx == ii):
				keep_idx.append(adj_list[ii])
			elif (sac_idx in adj_list[ii]):
				keep_idx.append([ii])

			if (anti_sac_idx == ii):
				other_idx.append(adj_list[ii])
			elif (anti_sac_idx in adj_list[ii]):
				other_idx.append([ii])



		keep_idx = list(itertools.chain(*keep_idx))
		keep_idx = np.array(keep_idx)
		other_idx = list(itertools.chain(*other_idx))
		other_idx = np.array(other_idx)

		keep_idx = [x for x in keep_idx if x != anti_sac_idx]
		keep_idx = np.unique(keep_idx)

		other_idx = [x for x in other_idx if x != sac_idx]
		other_idx = np.unique(other_idx)


		self.mesh_cc_ragged = mesh_cc_ragged_vol.get(keep_idx).combine(merge_points=True)

		self.mesh_cc_ragged_inv = mesh_cc_ragged_vol.get(other_idx).combine(merge_points=True)

		self.mesh_cc_ragged.cell_arrays['mask'] = 1
		self.mesh_cc_ragged_inv.cell_arrays['mask'] = 0
		self.mesh_cc_ragged.cell_arrays['mask'] = self.mesh_cc_ragged.cell_arrays['mask'].astype(int)
		self.mesh_cc_ragged_inv.cell_arrays['mask'] = self.mesh_cc_ragged_inv.cell_arrays['mask'].astype(int)

		blocks = pv.MultiBlock()
		blocks.append(self.mesh_cc_ragged)
		blocks.append(self.mesh_cc_ragged_inv)
		self.new_mesh = blocks.combine(merge_points=True)
		self.mesh = self.new_mesh

	def plane_clipping_cb(self, normal, origin):
		self.normal = normal
		self.origin = origin
		
		mesh_clip = self.mesh_surf.clip(normal, origin, invert=False)
		mesh_clip_inv = self.mesh_surf.clip(normal, origin, invert=True)

		if mesh_clip.n_points > 1:
			self.p.remove_actor('mesh')
		else:
			self.p.add_mesh(self.mesh_surf, name='mesh', color='white', opacity=0.1)

		if mesh_clip.n_points < 1:
			self.p.remove_actor('clip')
		else:
			self.p.add_mesh(mesh_clip, name='clip', color='green') 

		if mesh_clip_inv.n_points < 1:
			self.p.remove_actor('inv')
		else:
			self.p.add_mesh(mesh_clip_inv, name='inv', color='red') 


	def update(self):
		# print(self.normal, self.origin)
		self.clip(self.normal, self.origin)

		if self.mesh_cc_ragged.n_points > 1:
			self.p.remove_actor('mesh')
		else:
			self.p.add_mesh(self.mesh_surf, name='mesh', color='white', opacity=0.1)

		if self.mesh_cc_ragged.n_points < 1:
			self.p.remove_actor('clip')
		else:
			self.p.add_mesh(self.mesh_cc_ragged, name='clip', color='green') 

		if self.mesh_cc_ragged_inv.n_points < 1:
			self.p.remove_actor('inv')
		else:
			self.p.add_mesh(self.mesh_cc_ragged_inv, name='inv', color='red') 


	def triplane_cb(self, point, i, widget):
		dist, ind = self.sphere_tree.query(point, k=1)
		point = self.mesh_surf.points[ind]
		widget.SetCenter(point)
		self.triplane.points[i] = point
		a, b, c = self.triplane.points
		normal = np.cross((b-a), (c-a))
		normal = normal / np.linalg.norm(normal)
		origin = self.triplane.points.mean(axis=0)
		self.plane_clipping_cb(normal, origin)


	def interact(self):
		self.p = pv.Plotter()
		self.p.add_mesh(self.mesh_surf, name='mesh', color='white')
		self.p.add_plane_widget(self.plane_clipping_cb)
		self.p.add_key_event(
			key='space',
			callback=self.update,
			)
		self.p.show()


	def interact_2(self):
		self.p = pv.Plotter()
		self.p.add_mesh(self.mesh_surf, name='mesh', color='white')

		bounds = np.array(self.mesh_surf.bounds).reshape(3,2).T

		# Create the clipping surface
		self.triplane = pv.PolyData()
		init_points = np.array([
			bounds[0], 
			np.array(bounds.mean(axis=0)) + np.array((0,10,0)), 
			bounds[1]])

		self.sphere_tree = KDTree(self.mesh_surf.points, leafsize=2)
		dist, ind = self.sphere_tree.query(init_points, k=1) 

		self.triplane.points = self.mesh_surf.points[ind]

		self.triplane.faces = np.array([3, 0, 1, 2])

		# print('self.triplane.points?', self.triplane.points)
		self.p.add_sphere_widget(self.triplane_cb, center=self.triplane.points,pass_widget=True)


		# p.add_sphere_widget(callback, center=surf.points)
		self.p.add_mesh(self.triplane, color=True)

		# self.p.add_plane_widget(self.plane_clipping_cb)
		self.p.add_key_event(
			key='space',
			callback=self.update,
			)
		self.p.show()



if __name__ == "__main__":
	meshfile = '/Users/danmacdonald/Data/data/c0099_MCA_T.vtu'
	mesh = pv.read(meshfile)

	t = TubeClipper(mesh)
	t.interact_2()
	new_mesh = t.new_mesh
