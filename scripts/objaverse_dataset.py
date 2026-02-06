# %% Import dependencies
import objaverse
import trimesh
import os
import polyscope as ps
from dataclasses import dataclass, field, asdict
from typing import Literal, Tuple, cast
import random
from hyve.preprocess import scale_to_unit_cube, generate_sdf_samples_unit_cube, scale_to_unit_sphere, generate_sdf_samples_unit_sphere, generate_sdf_samples_grid
import torch
from einops import rearrange, pack
from torch_geometric.data import Data
from pathos import multiprocessing as mp
import json
from itertools import starmap

# %% Config
@dataclass(kw_only=True)
class ObjaverseConfig:
	raw_dir: str
	processed_dir: str
	gt_meshes: str
	num_surface_mesh_points: int = 100000
	num_meshes_train: int = 2000
	num_meshes_test: int = 200
	random_seed: int = 2
	scale_to_unit: Literal['cube', 'sphere'] = 'cube'
	sdf_samples_ratio: Tuple[int, int, int] = field(default_factory=lambda: (2, 2, 1))
	num_sdf_samples_train: int = 200000
	num_sdf_samples_test: int = 200000

cfg = ObjaverseConfig(raw_dir="datasets/objaverse/raw/",
					 processed_dir="datasets/objaverse/processed/",
					 gt_meshes="datasets/objaverse/processed/gt_meshes/")

random.seed(cfg.random_seed)

# %% Set download path
objaverse.BASE_PATH = cfg.raw_dir
objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")

# %% Get CC0 object metadata
uids = objaverse.load_uids()
annotations = objaverse.load_annotations(uids)

# %% Filter out the relevant entries and create a list of staff picked objects for validation
cc0_uids_like_count = [(uid, annotation['likeCount'], annotation['staffpickedAt']) for uid, annotation in annotations.items() if annotation["license"] == "cc0" and annotation['archives']['glb']['faceCount'] > 0]
cc0_uids_sorted = sorted(cc0_uids_like_count, key=lambda x: x[1], reverse=True)

cc0_uids_all = [x[0] for x in cc0_uids_sorted]
cc0_uids_staff_picked = [x[0] for x in cc0_uids_sorted if x[2] is not None]
cc0_uids_non_staff_picked = [x[0] for x in cc0_uids_sorted if x[2] is None]

# %% Download all objects
objects = objaverse.load_objects(uids=cc0_uids_all, download_processes=4)

# %% Select objects for training and testing
# Ensure that the test set includes all staff_picked instances
test_set_staff_picked = cc0_uids_staff_picked.copy()

# Randomly select non_staff_picked instances for the test set
num_non_staff_picked_for_test = max(0, cfg.num_meshes_test - len(test_set_staff_picked))
test_set_non_staff_picked = random.sample(cc0_uids_non_staff_picked, num_non_staff_picked_for_test, )

# Combine staff_picked and non_staff_picked instances for the test set
test_set = test_set_staff_picked + test_set_non_staff_picked
test_set_idx = [i for i in range(len(test_set))]
test_set_files = {uuid: f"{id}_test_{cfg.random_seed}_{cfg.num_sdf_samples_test}.pt" for (uuid, id) in zip(test_set, test_set_idx)}

# Training set sampling
train_set = random.sample([*(set(cc0_uids_non_staff_picked) - set(test_set))], cfg.num_meshes_train)
train_set_idx = [i for i in range(len(test_set), len(test_set)+len(train_set))]
train_set_files = {uuid: f"{id}_train_{cfg.random_seed}_{cfg.num_sdf_samples_train}.pt" for (uuid, id) in zip(train_set, train_set_idx)}

# Ensure that the training set is disjoint from the test set
print(f"Train set disjoint from test set: {len(set(train_set) - set(test_set_staff_picked)) == cfg.num_meshes_train}")

# %% Generate train and test split
# Make sure the directory exists
os.makedirs(cfg.processed_dir, exist_ok=True)
os.makedirs(cfg.gt_meshes, exist_ok=True)

scale_to_unit_fn = scale_to_unit_cube
generate_sdf_samples_fn = generate_sdf_samples_unit_cube
if cfg.scale_to_unit == 'cube':
	scale_to_unit_fn = scale_to_unit_cube
	generate_sdf_samples_fn = generate_sdf_samples_unit_cube
elif cfg.scale_to_unit == 'sphere':
	scale_to_unit_fn = scale_to_unit_sphere
	generate_sdf_samples_fn = generate_sdf_samples_unit_sphere

def generate_training_data(id, uuid):
	mesh = cast(trimesh.Trimesh, trimesh.load(objects[uuid], force='mesh'))
	mesh.vertices = scale_to_unit_fn(mesh.vertices)
	pos, sd, normal = generate_sdf_samples_fn(mesh.vertices, mesh.faces, cfg.num_sdf_samples_train, cfg.sdf_samples_ratio)
	y, _ = pack([pos, sd, normal], "n *")
	y = torch.from_numpy(rearrange(y, "(b n) x -> b n x", b=1)).float()

	outfile = os.path.join(cfg.processed_dir, train_set_files[uuid])
	print(f"Input UUID: {uuid}\n\t Output to: {outfile}")

	surf_samples, surf_faces = trimesh.sample.sample_surface(mesh, cfg.num_surface_mesh_points)
	surf_normals = mesh.face_normals[surf_faces]

	torch_data = Data(x=torch.from_numpy(surf_normals).float(),
				   	  y=y,
				      pos=torch.from_numpy(surf_samples).float())
	
	processed_data = {'volume': 0,  # None is not allowed
					'surface': torch_data,
					'id': int(id)}

	torch.save(processed_data, outfile)

	return outfile

def generate_test_data(id, uuid):
	mesh = cast(trimesh.Trimesh, trimesh.load(objects[uuid], force='mesh'))
	mesh.vertices = scale_to_unit_fn(mesh.vertices)
	pos, sd, normal = generate_sdf_samples_grid(mesh.vertices, mesh.faces, cfg.num_sdf_samples_test)
	y, _ = pack([pos, sd, normal], "n *")
	y  = torch.from_numpy(rearrange(y,  "(b n) x -> b n x", b=1)).float()

	outfile = os.path.join(cfg.processed_dir, test_set_files[uuid])
	print(f"Input UUID: {uuid}\n\t Output to: {outfile}")

	surf_samples, surf_faces = trimesh.sample.sample_surface(mesh, cfg.num_surface_mesh_points)
	surf_normals = mesh.face_normals[surf_faces]

	torch_data = Data(x=torch.from_numpy(surf_normals).float(),
				   	  y=y,
				      pos=torch.from_numpy(surf_samples).float())
	
	processed_data = {'volume': 0,  # None is not allowed
					'surface': torch_data,
					'id': int(id)}

	torch.save(processed_data, outfile)

	gt_mesh_file = os.path.join(cfg.gt_meshes, f"surface_test_{id}.ply")
	print(f"\t Ground truth mesh output to: {gt_mesh_file}")
	mesh.export(gt_mesh_file, include_attributes=False)
	return outfile

# %% Save test files
# with mp.Pool(mp.cpu_count()//2) as pool:
test_files = list(starmap(generate_test_data, zip(test_set_idx, test_set)))
train_files = list(starmap(generate_training_data, zip(train_set_idx, train_set)))

# %% Output info
with open(f"train_rel_{cfg.random_seed}_{cfg.num_meshes_train}.json", "w") as f:
	train_relative = [file for file in train_set_files.values()]
	json.dump(train_relative, f, indent=2)

with open(f"test_rel_{cfg.random_seed}_{cfg.num_meshes_test}.json", "w") as f:
	test_relative = [file for file in test_set_files.values()]
	json.dump(test_relative, f, indent=2)

with open(f"config.json", "w") as f:
	json.dump(asdict(cfg), f, indent=2)

# %%
def demo():
	# Force concatenation into a mesh
	mesh = trimesh.load(list(objects.values())[0], force='mesh')
	# mesh.show()

	# Sample the surface of the mesh and retrieve surface normals
	samples, faces = trimesh.sample.sample_surface(mesh, 100000)
	normals = mesh.face_normals[faces]
	# Use polyscope to show the mesh
	ps.init()
	vis_mesh = ps.register_surface_mesh("object", mesh.vertices, mesh.faces)
	vis_pc = ps.register_point_cloud("object_samples", samples)
	vis_pc.add_vector_quantity("normal", normals)
	ps.show()
