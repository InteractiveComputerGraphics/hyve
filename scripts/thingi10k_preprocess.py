#%% Imports
from dataclasses import dataclass, field, asdict
import os
import meshio
import json
import yaml
import pandas as pd
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import torch

import multiprocessing as mp

from itertools import starmap
from tqdm import tqdm
from typing import Literal, Tuple
from hyve.preprocess import scale_to_unit_cube, generate_sdf_samples_unit_cube, scale_to_unit_sphere, generate_sdf_samples_unit_sphere, generate_sdf_samples_grid
from einops import pack, rearrange
from torch_geometric.data import Data, DataLoader  # type: ignore
from torch_geometric.transforms import FaceToEdge  # type: ignore
from torch_geometric.utils import from_trimesh  # type: ignore
# from jsonargparse import CLI


#%% Dataclass and config
@dataclass(kw_only=True)
class Thingi10kConfig:
	surface_mesh_dir: str
	processed_dir: str
	valid_file_ids: str
	num_meshes_train: int = 2000
	num_meshes_test: int = 200
	max_mesh_points: int = 5000
	max_mesh_ar: float = 10
	random_seed: int = 2
	scale_to_unit: Literal['cube', 'sphere'] = 'cube'
	sdf_samples_ratio: Tuple[int, int, int] = field(default_factory=lambda: (2, 2, 1))
	num_sdf_samples_train: int = 100000
	num_sdf_samples_test: int = 200000

cfg = Thingi10kConfig(
	surface_mesh_dir='datasets/thingi10k/raw/10k_surface/', 
	processed_dir='datasets/thingi10k/processed/',
	valid_file_ids='datasets/thingi10k/valid_file_ids.json'
	)

with open(cfg.valid_file_ids, 'r') as f:
	valid_file_ids = json.load(f)

# %% Get mesh information
def get_mesh_info(id):
	sf_filename = os.path.join(cfg.surface_mesh_dir, f"{id}_sf.obj")
	try:
		mesh = meshio.read(sf_filename)
		bbox_l = np.max(mesh.points, axis=0) - np.min(mesh.points, axis=0)
		ar = np.max(bbox_l) / np.min(bbox_l)
		return {"points": len(mesh.points), "cells": len(mesh.cells[0].data), "id": id, "surface": sf_filename, "aspect": ar}
	except ValueError:
		print(f"Invalid Mesh: {sf_filename}")
		return None
	except Exception as e:
		print(f"Missing File or others: {sf_filename}\n{e}")


with mp.Pool(mp.cpu_count()) as pool:
	mesh_info = pool.map(get_mesh_info, tqdm(valid_file_ids['valid_file_ids']))
mesh_info = [info for info in mesh_info if info is not None]

#%% Filter meshes
data = pd.DataFrame(mesh_info)
data_f = data.copy()

iqr_p = data["points"].quantile(0.75) - data["points"].quantile(0.25)
iqr_c = data["cells"].quantile(0.75) - data["cells"].quantile(0.25)

print("Before filtering")
print(f"\tMean number of points: {data['points'].mean()}, Std. of points: {data['points'].std()}")
print(f"\tMean number of cells: {data['cells'].mean()}, Std. of cells: {data['cells'].std()}")
print(f"\tNumber of entries: {len(data)}")

# Only filter out upper outliers
# data_f = data_f[((data["points"] <= data["points"].quantile(0.75) + iqr_p*1.5) & (data_f["cells"] <= data_f["cells"].quantile(0.75) + iqr_c*1.5))]
data_f = data_f[((data["points"] <= cfg.max_mesh_points) & (data["cells"] <= data["cells"].quantile(0.75) + iqr_c * 1.5))]

print("After filtering points")
print(f"\tMean number of points: {data_f['points'].mean()}, Std. of points: {data_f['points'].std()}")
print(f"\tMean number of cells: {data_f['cells'].mean()}, Std. of cells: {data_f['cells'].std()}")
print(f"\tNumber of entries: {len(data_f)}")

data_f = data_f[data_f["aspect"] <= cfg.max_mesh_ar]

print("After filtering aspect ratio")
print(f"\tMean number of points: {data_f['points'].mean()}, Std. of points: {data_f['points'].std()}")
print(f"\tMean number of cells: {data_f['cells'].mean()}, Std. of cells: {data_f['cells'].std()}")
print(f"\tNumber of entries: {len(data_f)}")


# Select a certain number of instances from this set
data_s = data_f.sample(cfg.num_meshes_test+cfg.num_meshes_train, random_state=cfg.random_seed)
print("After selecting")
print(f"\tMean number of points: {data_s['points'].mean()}, Std. of points: {data_s['points'].std()}")
print(f"\tMean number of cells: {data_s['cells'].mean()}, Std. of cells: {data_s['cells'].std()}")
print(f"\tNumber of entries: {len(data_s)}")

selection = data_s.to_dict('records')
# with open(osp.join(processed_data_dir, f"valid_pairs_{args.name}.json"), "w") as f:
#     json.dump(selection, f, indent=2)

#%% Plot filtered, unfiltered and selected data
fig, ax = plt.subplots(2, 3, figsize=(12, 10))
fig.text(0.2, 0.92, "Original Data")
fig.text(0.45, 0.92, "Without Outliers")
fig.text(0.75, 0.92, "Selection")
ax[0][0].boxplot(data["points"], labels=["N Points"])
ax[0][1].boxplot(data_f["points"], labels=["N Points"])
ax[0][2].boxplot(data_s["points"], labels=["N Points"])
ax[1][0].boxplot(data["cells"], labels=["N Cells"])
ax[1][1].boxplot(data_f["cells"], labels=["N Cells"])
ax[1][2].boxplot(data_s["cells"], labels=["N Cells"])
plt.show()

#%% Save to torch function
def save_to_torch(idx, mesh, y, filename):
	surface_mesh_torch = from_trimesh(mesh)
	surface_mesh_torch.update(Data(y=y, x=mesh.vertex_normals))

	# meshio.write_points_cells(os.path.join(os.path.dirname(outfile), "test.ply"), pts[pts_id_map], [("triangle", surface)])
	processed_data = {'volume': 0,  # None is not allowed
					'surface': FaceToEdge(remove_faces=True)(surface_mesh_torch),
					'id': int(idx)}

	outfile = os.path.join(cfg.processed_dir, filename)
	torch.save(processed_data, outfile)

	return filename, outfile

# %% Generate train and test split
# Make sure the directory exists
os.makedirs(cfg.processed_dir, exist_ok=True)

if cfg.scale_to_unit == 'cube':
	scale_to_unit_fn = scale_to_unit_cube
	generate_sdf_samples_fn = generate_sdf_samples_unit_cube
elif cfg.scale_to_unit == 'sphere':
	scale_to_unit_fn = scale_to_unit_sphere
	generate_sdf_samples_fn = generate_sdf_samples_unit_sphere

idx = data_s.values[0][2]
mesh_file = data_s.values[0][3]

def generate_training_data(idx, mesh_file):
	mesh = trimesh.load_mesh(mesh_file)
	mesh.vertices = scale_to_unit_fn(mesh.vertices)
	pos, sd, normal = generate_sdf_samples_fn(mesh.vertices, mesh.faces, cfg.num_sdf_samples_train, cfg.sdf_samples_ratio)
	y, _ = pack([pos, sd, normal], "n *")
	y = torch.from_numpy(rearrange(y, "(b n) x -> b n x", b=1)).float()
	filename = f"{idx}_train_{cfg.random_seed}_{cfg.num_sdf_samples_train}.pt"
	return save_to_torch(idx, mesh, y, filename)

def generate_test_data(idx, mesh_file):
	mesh = trimesh.load_mesh(mesh_file)
	mesh.vertices = scale_to_unit_fn(mesh.vertices)
	pos, sd, normal = generate_sdf_samples_grid(mesh.vertices, mesh.faces, cfg.num_sdf_samples_test)
	y, _ = pack([pos, sd, normal], "n *")
	y  = torch.from_numpy(rearrange(y,  "(b n) x -> b n x", b=1)).float()
	filename = f"{idx}_test_{cfg.random_seed}_{cfg.num_sdf_samples_test}.pt"
	return save_to_torch(idx, mesh, y, filename)

# %% Save test files
with mp.Pool(mp.cpu_count()) as pool:
	train_files = list(pool.starmap(generate_training_data, data_s.values[:cfg.num_meshes_train, 2:4]))
	test_files = list(pool.starmap(generate_test_data, data_s.values[cfg.num_meshes_train:, 2:4]))

# %% Output info
with open(f"train_rel_{cfg.random_seed}_{cfg.num_meshes_train}.json", "w") as f:
	train_relative = [file[0] for file in train_files]
	json.dump(train_relative, f, indent=2)
with open(f"train_abs_{cfg.random_seed}_{cfg.num_meshes_train}.json", "w") as f:
	train_absolute = [file[1] for file in train_files]
	json.dump(train_absolute, f, indent=2)

with open(f"test_rel_{cfg.random_seed}_{cfg.num_meshes_test}.json", "w") as f:
	test_relative = [file[0] for file in test_files]
	json.dump(test_relative, f, indent=2)
with open(f"test_abs_{cfg.random_seed}_{cfg.num_meshes_test}.json", "w") as f:
	test_absolute = [file[1] for file in test_files]
	json.dump(test_absolute, f, indent=2)

with open(f"config.json", "w") as f:
	json.dump(asdict(cfg), f, indent=2)

# %% Copy surface meshes for testing into directory
test_split_file = '/local-hdd/sjeske/Documents/Projects/hyve/data/configs/thingi10k/splits/test_rel_2_200.json'


test_split = None
with open(test_split_file, 'r') as f:
	test_split = json.load(f)

test_mesh_dir = '/local-hdd/sjeske/Data/processed/thingi10k/test_meshes/'
for file in test_split:
	id = file.split('_')[0]
	source_mesh_file = os.path.join(cfg.surface_mesh_dir, f"{id}_sf.obj")
	source_mesh = meshio.read(source_mesh_file)
	source_mesh.points = scale_to_unit_fn(source_mesh.points)

	target_mesh_file = os.path.join(test_mesh_dir, f"{id}_test.ply")
	source_mesh.write(target_mesh_file)

#%% Spacing

















# %% Generate fixed vertex count dataset
cfg = Thingi10kConfig(
	surface_mesh_dir='/local-hdd/sjeske/Data/raw/10k_surface/', 
	processed_dir='/local-hdd/sjeske/Data/processed/thingi10k_fixed_vc/',
	valid_file_ids='/local-hdd/sjeske/Documents/Projects/hyve/data/configs/thingi10k/valid_file_ids.json'
	)
num_surface_mesh_points = 2048

processed_train_files_list = '/local-hdd/sjeske/Documents/Projects/hyve/data/configs/thingi10k/splits/train_rel_2_2000.json'
processed_train_files = []
with open(processed_train_files_list, 'r') as f:
	processed_train_files = json.load(f)

train_indices = [file.split('_')[0] for file in processed_train_files]
raw_train_files = [os.path.join(cfg.surface_mesh_dir, f"{idx}_sf.obj") for idx in train_indices]


processed_test_files_list = '/local-hdd/sjeske/Documents/Projects/hyve/data/configs/thingi10k/splits/test_rel_2_200.json'
processed_test_files = []
with open(processed_test_files_list, 'r') as f:
	processed_test_files = json.load(f)

test_indices = [file.split('_')[0] for file in processed_test_files]
raw_test_files = [os.path.join(cfg.surface_mesh_dir, f"{idx}_sf.obj") for idx in test_indices]

# Test that the files exist
print(f"All train files exist: {all([os.path.isfile(file) for file in raw_train_files])}")
print(f"All test files exist: {all([os.path.isfile(file) for file in raw_test_files])}")

print(f"Found {len(raw_train_files)} train files\nFound {len(raw_test_files)} test files.")

#%% Save to torch function
def save_to_torch(idx, mesh, y, filename):
	# Sample only when more points are needed, else sample from existing vertices
	if num_surface_mesh_points > len(mesh.vertices):
		num_extra_points = num_surface_mesh_points - len(mesh.vertices)
		print(f"\t Sampling {num_extra_points} extra points from the surface.")
		surf_samples, surf_faces = trimesh.sample.sample_surface(mesh, num_extra_points)
		surf_normals = mesh.face_normals[surf_faces]

		# Add the original vertices
		surf_samples = np.concatenate((mesh.vertices, surf_samples), axis=0)
		surf_normals = np.concatenate((mesh.vertex_normals, surf_normals), axis=0)
		print(f"\t Total number of points: {len(surf_samples)}")

	else:
		# Sample the correct number from existing vertices
		samples = np.random.choice(len(mesh.vertices), num_surface_mesh_points, replace=False)
		surf_samples = mesh.vertices[samples]
		surf_normals = mesh.vertex_normals[samples]

	# surf_samples, surf_faces = trimesh.sample.sample_surface(mesh, cfg.num_surface_mesh_points)
	# surf_normals = mesh.face_normals[surf_faces]

	torch_data = Data(x=torch.from_numpy(surf_normals).float(),
					y=y,
					pos=torch.from_numpy(surf_samples).float())
	
	processed_data = {'volume': 0,  # None is not allowed
					'surface': torch_data,
					'id': int(idx)}

	outfile = os.path.join(cfg.processed_dir, filename)
	torch.save(processed_data, outfile)

	return filename, outfile

# %% Generate train and test split
# Make sure the directory exists
os.makedirs(cfg.processed_dir, exist_ok=True)

if cfg.scale_to_unit == 'cube':
	scale_to_unit_fn = scale_to_unit_cube
	generate_sdf_samples_fn = generate_sdf_samples_unit_cube
elif cfg.scale_to_unit == 'sphere':
	scale_to_unit_fn = scale_to_unit_sphere
	generate_sdf_samples_fn = generate_sdf_samples_unit_sphere

def generate_training_data(idx, mesh_file):
	mesh = trimesh.load_mesh(mesh_file)
	mesh.vertices = scale_to_unit_fn(mesh.vertices)
	pos, sd, normal = generate_sdf_samples_fn(mesh.vertices, mesh.faces, cfg.num_sdf_samples_train, cfg.sdf_samples_ratio)
	y, _ = pack([pos, sd, normal], "n *")
	y = torch.from_numpy(rearrange(y, "(b n) x -> b n x", b=1)).float()
	filename = f"{idx}_train_{cfg.random_seed}_{cfg.num_sdf_samples_train}_{num_surface_mesh_points}.pt"
	return save_to_torch(idx, mesh, y, filename)

def generate_test_data(idx, mesh_file):
	mesh = trimesh.load_mesh(mesh_file)
	mesh.vertices = scale_to_unit_fn(mesh.vertices)
	pos, sd, normal = generate_sdf_samples_grid(mesh.vertices, mesh.faces, cfg.num_sdf_samples_test)
	y, _ = pack([pos, sd, normal], "n *")
	y  = torch.from_numpy(rearrange(y,  "(b n) x -> b n x", b=1)).float()
	filename = f"{idx}_test_{cfg.random_seed}_{cfg.num_sdf_samples_test}_{num_surface_mesh_points}.pt"
	return save_to_torch(idx, mesh, y, filename)

# %% Save test files
with mp.Pool(mp.cpu_count()) as pool:
	train_files = list(pool.starmap(generate_training_data, zip(train_indices[:], raw_train_files[:])))
	test_files = list(pool.starmap(generate_test_data, zip(test_indices[:], raw_test_files[:])))

# %% Output info
with open(f"train_rel_{cfg.random_seed}_{cfg.num_meshes_train}_fixed_vc.json", "w") as f:
	train_relative = [file[0] for file in train_files]
	json.dump(train_relative, f, indent=2)

with open(f"test_rel_{cfg.random_seed}_{cfg.num_meshes_test}_fixed_vc.json", "w") as f:
	test_relative = [file[0] for file in test_files]
	json.dump(test_relative, f, indent=2)



# %%
