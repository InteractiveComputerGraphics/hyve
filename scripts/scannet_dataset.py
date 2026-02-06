#%% Import
from cgi import test
from genericpath import isdir
import random
from einops import pack, rearrange
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import Data
import polyscope as ps
import polyscope.imgui as psim
import fileseq
import meshio
import os
import json
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import List, Literal, Optional, cast
from tqdm import tqdm
from pathos import multiprocessing as mp
import numpy as np

import trimesh

from hyve.preprocess import generate_sdf_samples_grid, generate_sdf_samples_unit_cube, scale_to_unit_cube

@dataclass(kw_only=True)
class ScannetConfig:
	base_dir: str
	processed_dir: str
	gt_mesh_dir: str
	# train_config: str
	# test_config: str

	num_surface_mesh_points: int = 100000
	random_seed: int = 2

	sdf_samples_ratio: tuple[int, int, int] = field(default_factory=lambda: (2, 2, 1))

	num_sdf_samples_train: int = 200000
	num_sdf_samples_test: int = 200000


cfg = ScannetConfig(
	base_dir="datasets/ScanNet/",
	processed_dir="datasets/ScanNet/processed/",
	gt_mesh_dir="datasets/ScanNet/processed/gt_meshes/",
	)

random.seed(cfg.random_seed)
# %% Check the directory and get a list of all files and folders
# Information about the dataset can be found here: https://kaldir.vc.in.tum.de/scannet_benchmark/documentation

# Training split
train_files = os.listdir(os.path.join(cfg.base_dir, "scans"))
raw_train_files = [os.path.join('scans', f, f"{f}_vh_clean_2.ply") for f in train_files]
processed_train_files = [os.path.join('scans', f, f"{f}.pt") for f in train_files]
print(f"Found {len(train_files)} train files in the base directory.")

# Test split
test_files = os.listdir(os.path.join(cfg.base_dir, "scans_test"))
raw_test_files = [os.path.join('scans_test', f, f"{f}_vh_clean_2.ply") for f in test_files]
processed_test_files = [os.path.join('scans_test', f, f"{f}.pt") for f in test_files]
gt_mesh_test_files = [os.path.join(f"surface_{f}.ply") for f in test_files]
print(f"Found {len(test_files)} test files in the base directory.")

#%% Check if raw test and train files exist
raw_train_files = [f for f in raw_train_files if os.path.isfile(os.path.join(cfg.base_dir, f))]
print(f"Found {len(raw_train_files)} train files on disk.")

raw_test_files = [f for f in raw_test_files if os.path.isfile(os.path.join(cfg.base_dir, f))]
print(f"Found {len(raw_test_files)} test files on disk.")

#%% Visualize a train sample using polyscope
train_sample = random.choice(raw_train_files)
train_sample = os.path.join(cfg.base_dir, train_sample)
print(f"Visualizing train sample: {train_sample}")

mesh = trimesh.load(train_sample, force='mesh')
mesh.show()
# ps.init()
# ps_mesh = ps.register_surface_mesh("train_sample", mesh.vertices, mesh.faces)
# ps.show()


#%% Get info about meshes
train_vertices = [len(trimesh.load(os.path.join(cfg.base_dir, file), force='mesh').vertices) for file in tqdm(raw_train_files)]

# Histogram of the number of vertices
plt.hist(train_vertices, bins=100)
plt.show()
# test_vertices = [len(trimesh.load(file, force='mesh').vertices) for _, file in tqdm(raw_test_files.items())]

# %% Generate train and test split
# Make sure the directory exists
os.makedirs(cfg.processed_dir, exist_ok=True)
os.makedirs(cfg.gt_mesh_dir, exist_ok=True)

def get_generation_function(mode: Literal['train', 'test']):

	def generate_data(id):
		if mode == 'train':
			mesh = cast(trimesh.Trimesh, trimesh.load(os.path.join(cfg.base_dir, raw_train_files[id]), force='mesh'))
			mesh.vertices = scale_to_unit_cube(mesh.vertices)
			pos, sd, normal = generate_sdf_samples_unit_cube(mesh.vertices, mesh.faces, cfg.num_sdf_samples_train, cfg.sdf_samples_ratio)
			outfile = os.path.join(cfg.processed_dir, processed_train_files[id])
		elif mode == 'test':
			mesh = cast(trimesh.Trimesh, trimesh.load(os.path.join(cfg.base_dir, raw_test_files[id]), force='mesh'))
			mesh.vertices = scale_to_unit_cube(mesh.vertices)
			pos, sd, normal = generate_sdf_samples_grid(mesh.vertices, mesh.faces, cfg.num_sdf_samples_test)
			outfile = os.path.join(cfg.processed_dir, processed_test_files[id])
		else:
			raise RuntimeError(f"Unrecognized mode: {mode}.")

		y, _ = pack([pos, sd, normal], "n *")
		y = torch.from_numpy(rearrange(y, "(b n) x -> b n x", b=1)).float()

		print(f"Input ID: {id}\n\t Output to: {outfile}")
		os.makedirs(os.path.dirname(outfile), exist_ok=True)

		# Sample only when more points are needed, else sample from existing vertices
		if cfg.num_surface_mesh_points > len(mesh.vertices):
			num_extra_points = cfg.num_surface_mesh_points - len(mesh.vertices)
			print(f"\t Sampling {num_extra_points} extra points from the surface.")
			surf_samples, surf_faces = trimesh.sample.sample_surface(mesh, num_extra_points)
			surf_normals = mesh.face_normals[surf_faces]

			# Add the original vertices
			surf_samples = np.concatenate((mesh.vertices, surf_samples), axis=0)
			surf_normals = np.concatenate((mesh.vertex_normals, surf_normals), axis=0)
			print(f"\t Total number of points: {len(surf_samples)}")

		else:
			# Sample the correct number from existing vertices
			samples = np.random.choice(len(mesh.vertices), cfg.num_surface_mesh_points, replace=False)
			surf_samples = mesh.vertices[samples]
			surf_normals = mesh.vertex_normals[samples]

		# surf_samples, surf_faces = trimesh.sample.sample_surface(mesh, cfg.num_surface_mesh_points)
		# surf_normals = mesh.face_normals[surf_faces]

		torch_data = Data(x=torch.from_numpy(surf_normals).float(),
						y=y,
						pos=torch.from_numpy(surf_samples).float())
		
		processed_data = {'volume': 0,  # None is not allowed
						'surface': torch_data,
						'id': int(id)}

		torch.save(processed_data, outfile)

		if mode == 'test':
			gt_mesh_file = os.path.join(cfg.gt_mesh_dir, f"surface_test_{id}.ply")
			print(f"\t Ground truth mesh output to: {gt_mesh_file}")
			mesh.export(gt_mesh_file, include_attributes=False)
		return outfile

	return generate_data

# %% Generate data
generate_data_fun = get_generation_function('train')
train_ids = np.arange(len(raw_train_files))
generate_data_fun(train_ids[0])
# with mp.Pool(mp.cpu_count()) as pool:
train_files = list(map(generate_data_fun, tqdm(train_ids)))

generate_data_fun = get_generation_function('test')
test_ids = np.arange(len(raw_test_files))
generate_data_fun(test_ids[0])
test_files = list(map(generate_data_fun, tqdm(test_ids)))

# %% Output config file
with open(f"config.json", "w") as f:
	json.dump(asdict(cfg), f, indent=2)

# Output list of train files as json
with open(f"train_files_rel.json", "w") as f:
	json.dump(processed_train_files, f, indent=2)

# Output list of test files as json
with open(f"test_files_rel.json", "w") as f:
	json.dump(processed_test_files, f, indent=2)


# %% 
