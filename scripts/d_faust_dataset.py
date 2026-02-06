#%% Import
from genericpath import isdir
import random
from einops import pack, rearrange
import torch
from torch_geometric.datasets.dynamic_faust import DynamicFAUST
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

import trimesh

from hyve.preprocess import generate_sdf_samples_grid, generate_sdf_samples_unit_cube, scale_to_unit_cube

@dataclass(kw_only=True)
class DFaustConfig:
	base_dir: str
	processed_dir: str
	gt_mesh_dir: str
	train_config: str
	test_config: str
	# SAL and SALD use 500k
	num_surface_mesh_points: int = 100000
	random_seed: int = 2
	sdf_samples_ratio: tuple[int, int, int] = field(default_factory=lambda: (2, 2, 1))
	# SAL and SALD use 500k (250k close and 250k more distant)
	num_sdf_samples_train: int = 200000
	num_sdf_samples_test: int = 200000
	# subjects: List[str] = field(default_factory=lambda: [
		# '50002', '50004', '50007', '50009', '50020', '50021', '50022', '50025',
		# '50026', '50027' 
		# ])
	# categories: List[str] = field(default_factory=lambda: [
		# 'chicken_wings', 'hips', 'jiggle_on_toes', 'jumping_jacks', 'knees',
		# 'light_hopping_loose', 'light_hopping_stiff', 'one_leg_jump',
		# 'one_leg_loose', 'personal_move', 'punching', 'running_on_spot', 
		# 'running_on_spot_bugfix', 'shake_arms', 'shake_hips', 'shake_shoulders' 
		# ])

cfg = DFaustConfig(
	base_dir="datasets/d_faust/scans",
	processed_dir="datasets/d_faust/processed/scans/",
	gt_mesh_dir="datasets/d_faust/processed/gt_meshes/",
	train_config="datasets/d_faust/SAL/train_all_every5.json",
	test_config="datasets/d_faust/SAL/test_all_every5.json",
	)

random.seed(cfg.random_seed)
# %% Check the SALD train/ test split
train_split = cfg.train_config
with open(train_split, "r") as f:
	content = json.load(f)

total_train_files = 0
train_files = []
train_ids = []
current_id = 0
for sid in content['scans']:
	for cat in content['scans'][sid]:
		files = content['scans'][sid][cat]
		total_train_files += len(files)
		train_files += [f"{sid}/{cat}/{file}" for file in files]
		train_ids += [i+current_id for i in range(len(files))]
		current_id += len(files)

test_split = cfg.test_config
with open(test_split, "r") as f:
	content = json.load(f)

total_test_files = 0
test_files = []
test_ids = []
for sid in content['scans']:
	for cat in content['scans'][sid]:
		files = content['scans'][sid][cat]
		total_test_files += len(files)
		test_files += [f"{sid}/{cat}/{file}" for file in files]
		test_ids += [i+current_id for i in range(len(files))]
		current_id += len(files)

print(f"Found a total of: {total_train_files} train files and")
print(f"Found a total of: {total_test_files} train files")

print(f"Intersection of: {len(set(train_files).intersection(set(test_files)))} files")

#%% Check to open the actual files
raw_train_files = [os.path.join(cfg.base_dir, f+".ply") for f in train_files]
raw_train_files = {i: f for i, f in zip(train_ids, raw_train_files) if os.path.isfile(f)}

processed_train_files = {i: os.path.join(cfg.processed_dir, f+".pt") for i, f in zip(train_ids, train_files)}

print(f"Found {len(raw_train_files)} train files on disk.")

raw_test_files = [os.path.join(cfg.base_dir, f+".ply") for f in test_files]
raw_test_files = {i: f for i, f in zip(test_ids, raw_test_files) if os.path.isfile(f)}

processed_test_files = {i: os.path.join(cfg.processed_dir, f+".pt") for i, f in zip(test_ids, test_files)}

print(f"Found {len(raw_test_files)} test files on disk.")

#%% Get info about meshes
train_vertices = [len(trimesh.load(file, force='mesh').vertices) for _, file in tqdm([*raw_train_files.items()][:100])]
# test_vertices = [len(trimesh.load(file, force='mesh').vertices) for _, file in tqdm(raw_test_files.items())]

# %% Generate train and test split
# Make sure the directory exists
os.makedirs(cfg.processed_dir, exist_ok=True)
os.makedirs(cfg.gt_mesh_dir, exist_ok=True)

def get_generation_function(mode: Literal['train', 'test']):

	def generate_data(id):
		if mode == 'train':
			mesh = cast(trimesh.Trimesh, trimesh.load(raw_train_files[id], force='mesh'))
			mesh.vertices = scale_to_unit_cube(mesh.vertices)
			pos, sd, normal = generate_sdf_samples_unit_cube(mesh.vertices, mesh.faces, cfg.num_sdf_samples_train, cfg.sdf_samples_ratio)
			outfile = processed_train_files[id]
		elif mode == 'test':
			mesh = cast(trimesh.Trimesh, trimesh.load(raw_test_files[id], force='mesh'))
			mesh.vertices = scale_to_unit_cube(mesh.vertices)
			pos, sd, normal = generate_sdf_samples_grid(mesh.vertices, mesh.faces, cfg.num_sdf_samples_test)
			outfile = processed_test_files[id]
		else:
			raise RuntimeError(f"Unrecognized mode: {mode}.")

		y, _ = pack([pos, sd, normal], "n *")
		y = torch.from_numpy(rearrange(y, "(b n) x -> b n x", b=1)).float()

		print(f"Input ID: {id}\n\t Output to: {outfile}")
		os.makedirs(os.path.dirname(outfile), exist_ok=True)

		surf_samples, surf_faces = trimesh.sample.sample_surface(mesh, cfg.num_surface_mesh_points)
		surf_normals = mesh.face_normals[surf_faces]

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

		return outfile

	return generate_data

# %% Generate data
generate_data_fun = get_generation_function('train')
generate_data_fun(train_ids[0])
# with mp.Pool(mp.cpu_count()) as pool:
train_files = list(map(generate_data_fun, tqdm(train_ids)))

generate_data_fun = get_generation_function('test')
generate_data_fun(test_ids[0])
test_files = list(map(generate_data_fun, tqdm(test_ids)))

# %% Output config file
with open(f"config.json", "w") as f:
	json.dump(asdict(cfg), f, indent=2)


# %% Debug things
def debug():
	# % Check RAW scan files
	base_dir = "datasets/d_faust/scans/"
	subjects = [
		'50002', '50004', '50007', '50009', '50020', '50021', '50022', '50025',
		'50026', '50027'
	]

	categories = [
		'chicken_wings', 'hips', 'jiggle_on_toes', 'jumping_jacks', 'knees',
		'light_hopping_loose', 'light_hopping_stiff', 'one_leg_jump',
		'one_leg_loose', 'personal_move', 'punching', 'running_on_spot', 
		'running_on_spot_bugfix', 'shake_arms', 'shake_hips', 'shake_shoulders'
	]

	total_files = 0
	for (sid, cat) in product(subjects, categories):
		cat_dir = os.path.join(base_dir, sid, cat)
		if not os.path.isdir(cat_dir):
			# print(f"Missing {sid}/{cat}")
			continue
		files = [file for file in os.listdir(cat_dir)][::5]
		total_files += len(files)

	dataset_files = total_files
	train_files = dataset_files//4 * 3
	test_files = dataset_files//4
	print(f"Found a total of: {total_files} files")
	print(f"\tSplit of 1:4 would result in {dataset_files} files")
	print(f"\tNumber of train files: {train_files}")
	print(f"\tNumber of test files: {test_files}")


	#% Load dataset
	dataset = DynamicFAUST("datasets/d_faust")

	# % Get a dataloader
	data = dataset[0]
	len(dataset)

	# %Define polyscope callback
	frame = 0
	max_frame = len(dataset)
	def callback():
		global frame, data, pc, mesh

		changed, frame = psim.InputInt("frame", frame, step=1, step_fast=10) 
		if changed:
			data = dataset[frame]
			ps.register_point_cloud("faust", data.pos[0])
			ps.register_surface_mesh("faust_sf", data.pos[0], data.face.T.numpy())


	# % Visualize
	ps.init()
	ps.set_user_callback(callback)

	ps.show()

	# % Define Dataset
	dataset = fileseq.findSequencesOnDisk('datasets/d_faust/scans/50002/jumping_jacks/')[0]
	mesh = meshio.read(dataset[0])

	# % Define polyscope callback
	frame = 0
	max_frame = len(dataset)
	def callback():
		global frame, data, pc, mesh

		changed, frame = psim.InputInt("frame", frame, step=1, step_fast=10) 
		if changed:
			data = meshio.read(dataset[frame])
			ps.register_point_cloud("faust", data.points)
			ps.register_surface_mesh("faust_sf", data.points, data.cells_dict['triangle'])


	# % Visualize
	ps.init()
	ps.set_user_callback(callback)

	ps.show()

	# % Debug a single generated file
	data = torch.load("datasets/d_faust/processed/scans/50002/one_leg_jump/one_leg_jump.006628.pt")
	ps.init()

	sf = ps.register_point_cloud("sf", data['surface'].pos)
	sf.add_vector_quantity("normals", data['surface'].x)

	vol = ps.register_point_cloud("vol", data['surface'].y[0,:,:3])
	vol.add_scalar_quantity("sd", data['surface'].y[0,:,3])
	vol.add_vector_quantity("normals", data['surface'].y[0,:,4:7])

	ps.show()