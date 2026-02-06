#%% Imports
from dataclasses import dataclass, field, asdict
from importlib.metadata import files
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
class ShapenetConfig:
	surface_mesh_dir: str
	processed_dir: str
	file_list: str
	num_meshes: int = 1000
	max_mesh_points: int = 100000
	max_mesh_ar: float = 100
	random_seed: int = 2
	scale_to_unit: Literal['cube', 'sphere'] = 'cube'
	mode: Literal['train', 'test'] = 'train'
	sdf_samples_ratio: Tuple[int, int, int] = field(default_factory=lambda: (2, 2, 1))
	num_sdf_samples: int = 100000
	overwrite_existing: bool = False
	# num_sdf_samples_test: int = 200000

cfg = ShapenetConfig(
	surface_mesh_dir='datasets/ShapeNetCore.V2', 
	processed_dir='datasets/ShapeNetCore.V2_processed',
	file_list='datasets/shapenetv2/planes_train.json'
	)

# cfg.file_list='datasets/shapenetv2/planes_train.json'
# cfg.mode = 'train'

cfg.file_list='datasets/shapenetv2/planes_test.json'
cfg.mode = 'test'
# cfg.num_meshes = 100
# cfg.num_sdf_samples = 100000

cfg.overwrite_existing = True

with open(cfg.file_list, 'r') as f:
	files_dict = json.load(f)

# %% Get mesh information
def get_mesh_info(id):
	sf_filename = os.path.join(cfg.surface_mesh_dir, f"{id}.ply")
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

in_files_list = []
out_files_list = []
ids = []
for class_id in files_dict:
	os.makedirs(os.path.join(cfg.processed_dir, class_id), exist_ok=True)
	for file in files_dict[class_id]:
		infile = os.path.join(cfg.surface_mesh_dir, class_id, f"{file}.ply")
		outfile = os.path.join(cfg.processed_dir, class_id, f"{file}.pt")
		if not os.path.isfile(infile):
			continue
		if os.path.isfile(outfile) and not cfg.overwrite_existing:
			continue

		in_files_list += [infile]
		out_files_list += [outfile]
		ids += [os.path.join(class_id, file)]


with mp.Pool(mp.cpu_count()) as pool:
	mesh_info = pool.map(get_mesh_info, tqdm(ids))
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
data_s = data_f.sample(cfg.num_meshes, random_state=cfg.random_seed)
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
def save_to_torch(id, idx, mesh, y):
	surface_mesh_torch = from_trimesh(mesh)
	surface_mesh_torch.update(Data(y=y, x=mesh.vertex_normals))

	# meshio.write_points_cells(os.path.join(os.path.dirname(outfile), "test.ply"), pts[pts_id_map], [("triangle", surface)])
	processed_data = {'volume': 0,  # None is not allowed
					'surface': FaceToEdge(remove_faces=True)(surface_mesh_torch),
					'id': id}

	outfile = os.path.join(cfg.processed_dir, f"{idx}.pt")
	torch.save(processed_data, outfile)

	return outfile

# %% Generate train and test split
# Make sure the directory exists
os.makedirs(cfg.processed_dir, exist_ok=True)

if cfg.scale_to_unit == 'cube':
	scale_to_unit_fn = scale_to_unit_cube
	generate_sdf_samples_fn = generate_sdf_samples_unit_cube
elif cfg.scale_to_unit == 'sphere':
	scale_to_unit_fn = scale_to_unit_sphere
	generate_sdf_samples_fn = generate_sdf_samples_unit_sphere 

id = data_f.index.to_numpy()
idx = data_f.values[:,2]
mesh_file = data_f.values[:,3]

generate_data_input, _ = pack([id, idx, mesh_file], "n *")

def generate_data(id, idx, mesh_file):
	mesh = trimesh.load_mesh(mesh_file)
	mesh.vertices = scale_to_unit_fn(mesh.vertices)
	pos, sd, normal = generate_sdf_samples_fn(mesh.vertices, mesh.faces, cfg.num_sdf_samples, cfg.sdf_samples_ratio) if cfg.mode == 'train' else generate_sdf_samples_grid(mesh.vertices, mesh.faces, cfg.num_sdf_samples)
	y, _ = pack([pos, sd, normal], "n *")
	y = torch.from_numpy(rearrange(y, "(b n) x -> b n x", b=1)).float()
	return save_to_torch(id, idx, mesh, y)


# %% Save test files
with mp.Pool(mp.cpu_count()) as pool:
	train_files = list(pool.starmap(generate_data, generate_data_input))

# %% Move target meshes into a specified folder
source_dir = "datasets/ShapeNetCore.V2/"
target_dir = "datasets/ShapeNetCore.V2/surface_meshes/02691156/"


for id, file in zip(id, mesh_file):
	if not os.path.isfile(file):
		continue
	source_mesh = meshio.read(file)
	source_mesh.points = scale_to_unit_fn(source_mesh.points)
	target_mesh_file = os.path.join(target_dir, f"{id:04d}_sf.vtu")
	source_mesh.write(target_mesh_file)

# %% Blank













# %% Generate dataset with fixed vertex count
processed_dir_fixed_vc = cfg.processed_dir + "_fixed_vc"
os.makedirs(processed_dir_fixed_vc, exist_ok=True)
class_id = "02691156"
num_surface_mesh_points = 6000

processed_file_ids = [file.split(".")[0] for file in os.listdir(os.path.join(cfg.processed_dir, class_id))]
processed_file_ids = [file for file in processed_file_ids if file in files_dict[class_id]]
indices = [torch.load(os.path.join(cfg.processed_dir, class_id, f"{file}.pt"))['id'] for file in tqdm(processed_file_ids)]

raw_files = [os.path.join(cfg.surface_mesh_dir, class_id, f"{file}.ply") for file in processed_file_ids]
raw_files = [file for file in raw_files if os.path.isfile(file)]

print(f"Number of processed files: {len(processed_file_ids)}")
print(f"Number of corresponding raw files: {len(raw_files)}")


# %% Processing functions
generate_data_input = zip(indices[:], processed_file_ids[:], raw_files[:])

def save_to_torch(id, idx, mesh, y):
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
					'id': id}

	outfile = os.path.join(processed_dir_fixed_vc, f"{idx}.pt")
	torch.save(processed_data, outfile)

	return outfile

# Generate train and test split
if cfg.scale_to_unit == 'cube':
	scale_to_unit_fn = scale_to_unit_cube
	generate_sdf_samples_fn = generate_sdf_samples_unit_cube
elif cfg.scale_to_unit == 'sphere':
	scale_to_unit_fn = scale_to_unit_sphere
	generate_sdf_samples_fn = generate_sdf_samples_unit_sphere 

def generate_data(id, idx, mesh_file):
	mesh = trimesh.load_mesh(mesh_file)
	mesh.vertices = scale_to_unit_fn(mesh.vertices)
	pos, sd, normal = generate_sdf_samples_fn(mesh.vertices, mesh.faces, cfg.num_sdf_samples, cfg.sdf_samples_ratio) if cfg.mode == 'train' else generate_sdf_samples_grid(mesh.vertices, mesh.faces, cfg.num_sdf_samples)
	y, _ = pack([pos, sd, normal], "n *")
	y = torch.from_numpy(rearrange(y, "(b n) x -> b n x", b=1)).float()
	return save_to_torch(id, idx, mesh, y)


# %% Save test files
with mp.Pool(mp.cpu_count()) as pool:
	train_files = list(pool.starmap(generate_data, generate_data_input))
# %%
