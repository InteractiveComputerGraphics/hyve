#%% Import
from sklearn.neighbors import NearestNeighbors
import meshio
import fileseq
import numpy as np
import os
import trimesh

#%% Define paths

# Path to the ScanNet ground truth surfaces
scannet_path = 'datasets/ScanNet/processed/gt_meshes/'

# Example paths
reconstructed_base_path = {
	# 'gt': 'datasets/ScanNet/processed/',
	# 'gt_points': 'datasets/ScanNet/processed/test_point_cloud',
	# 'max': 'experiments/scannet/HYVE/version_2/
}

reconstructed_path = {key: os.path.join(reconstructed_base_path[key], 'reconstructions') for key in reconstructed_base_path}

# Path to save the extracted surfaces for rendering
processed_path = {key: os.path.join(reconstructed_base_path[key], 'extracted_surfaces') for key in reconstructed_base_path}

#%% Load surface sequences

# Load the ScanNet ground truth surfaces
scannet_files = fileseq.findSequencesOnDisk(scannet_path)[0]

# Load the reconstructed ScanNet surfaces
reconstructed_files = {key: fileseq.findSequencesOnDisk(reconstructed_path[key])[0] for key in reconstructed_path}

#%% Extract surfaces by loading each surface. Then the closest point of all points in the reconstruction on the surface of the gt is found. If the distance is below a threshold, the point is kept.

# Define the threshold for the closest point search
# threshold = 0.025
threshold = 0.2

for key in reconstructed_files:
	# Create the directory to save the extracted surfaces
	os.makedirs(processed_path[key], exist_ok=True)

	# Iterate over all ScanNet ground truth surfaces
	for i, scannet_file in enumerate(scannet_files):
		# Load the ScanNet ground truth surface using meshio
		meshio_scannet_surface = meshio.read(scannet_file)
		scannet_surface_vertices = meshio_scannet_surface.points
		scale = np.array([0.9, 0.9, 1.0])
		bbox_min = np.min(scannet_surface_vertices, axis=0) * scale
		bbox_max = np.max(scannet_surface_vertices, axis=0) * scale - np.array([0.0, 0.0, 0.1])

		# Load the reconstructed ScanNet surface
		reconstructed_file = reconstructed_files[key][i]
		meshio_reconstructed_surface = meshio.read(reconstructed_file)
		reconstructed_surface_vertices = meshio_reconstructed_surface.points

		# Find the closest point on the reconstructed surface for each point on the ScanNet ground truth surface using NearestNeighbors
		nn = NearestNeighbors(n_neighbors=1)
		nn.fit(scannet_surface_vertices)
		distances, indices = nn.kneighbors(reconstructed_surface_vertices)

		# Save the extracted surface
		# Create a mask for vertices to keep
		keep_mask = distances.flatten() < threshold

		# Reindex faces
		if 'triangle' in meshio_reconstructed_surface.cells_dict:
			faces = meshio_reconstructed_surface.cells_dict['triangle']

			# Step 4: Reindex faces
			# Get the mapping from old vertex indices to new ones
			old_to_new_indices = np.cumsum(keep_mask) - 1

			# Update faces, filter out faces that contain removed vertices
			valid_faces = np.all(keep_mask[faces], axis=1)  # Find faces that only use valid vertices
			new_faces = old_to_new_indices[faces[valid_faces]]  # Remap the valid faces
			new_vertices = meshio_reconstructed_surface.points[keep_mask]
			new_vertex_normals = meshio_reconstructed_surface.point_data['normals'][keep_mask] if 'normals' in meshio_reconstructed_surface.point_data else None

			# Copy the extracted mesh into trimesh
			extracted_trimesh_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, vertex_normals=new_vertex_normals)
			extracted_trimesh_mesh = extracted_trimesh_mesh.slice_plane(bbox_min, [1, 0, 0])
			# extracted_trimesh_mesh = extracted_trimesh_mesh.slice_plane(bbox_min, [0, 1, 0])
			extracted_trimesh_mesh = extracted_trimesh_mesh.slice_plane(bbox_max, [-1, 0, 0])
			extracted_trimesh_mesh = extracted_trimesh_mesh.slice_plane(bbox_max, [0, -1, 0])
			extracted_trimesh_mesh = extracted_trimesh_mesh.slice_plane([0,0,0.1], [0, 0, -1])
			
			# Convert into meshio mesh
			extracted_meshio_mesh = meshio.Mesh(extracted_trimesh_mesh.vertices, {'triangle': extracted_trimesh_mesh.faces}, point_data={'normals': extracted_trimesh_mesh.vertex_normals})
			
			# Save the extracted surface
			meshio.write(os.path.join(processed_path[key], f'extracted_{i:04d}.vtu'), extracted_meshio_mesh)
		else:
			new_vertices = meshio_reconstructed_surface.points[keep_mask]
			mask_a = new_vertices[:, 0] > bbox_min[0]
			mask_b = new_vertices[:, 0] < bbox_max[0]
			mask_c = new_vertices[:, 1] < bbox_max[1]
			mask_d = new_vertices[:, 2] < 0.1

			mask = mask_a & mask_b & mask_c & mask_d
			new_vertices = new_vertices[mask]

			# Convert into meshio mesh
			extracted_meshio_mesh = meshio.Mesh(new_vertices, {'vertex': np.arange(len(new_vertices)).reshape(-1, 1)}, point_data={'normals': meshio_reconstructed_surface.point_data['normals'][keep_mask][mask]})

			# Save the extracted surface
			meshio.write(os.path.join(processed_path[key], f'extracted_{i:04d}.vtu'), extracted_meshio_mesh)


		print(f'Extracted surface {i} for {key}.')

# %%
