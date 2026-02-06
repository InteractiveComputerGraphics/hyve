#%% Imports
from dataclasses import dataclass, field, asdict
import os
import sys
import meshio
import json
import yaml
import pandas as pd
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import torch
import subprocess

import multiprocessing as mp

from itertools import starmap
from tqdm import tqdm
from typing import Literal, Tuple, List
from hyve.preprocess import scale_to_unit_cube, generate_sdf_samples_unit_cube, scale_to_unit_sphere, generate_sdf_samples_unit_sphere, generate_sdf_samples_grid
from einops import pack, rearrange
from torch_geometric.data import Data, DataLoader  # type: ignore
from torch_geometric.transforms import FaceToEdge  # type: ignore
from torch_geometric.utils import from_trimesh  # type: ignore
from jsonargparse import CLI
# from jsonargparse import CLI


#%% Dataclass and config
@dataclass(kw_only=True)
class ShapenetConfig:
	source_dir: str
	processed_dir: str
	split_file: str
	tetwild_bin: str  # Download / build from here https://github.com/wildmeshing/fTetWild
	tetwild_args: List[str] = field(default_factory=lambda: [
		'--max-threads', '1',
		# '--manifold-surface'
	])
	cleanup: bool = True
	overwrite: bool = False
	num_threads: int = mp.cpu_count()

if not hasattr(sys, 'ps1') or not sys.ps1:
	cfg: ShapenetConfig = CLI(ShapenetConfig)
else:
	cfg = ShapenetConfig(
		source_dir='datasets/ShapeNetCore.V2/', 
		processed_dir='datasets/ShapeNetCore.V2_processed/',
		split_file='',
		tetwild_bin='external/fTetWild/FloatTetwild_bin'
		)

	# cfg.split_file = 'datasets/shapenetv2/planes_train.json'
	cfg.split_file = 'datasets/shapenetv2/planes_test.json'

with open(cfg.split_file, 'r') as f:
	file_list = json.load(f)

if os.path.isfile(cfg.tetwild_bin):
	print("Found tetwild.")

#%% Iterate over all files and make list of infiles and outfiles
in_out_files = []
processed_files = []
for class_id in file_list:
	os.makedirs(os.path.join(cfg.processed_dir, class_id), exist_ok=True)

	for file_id in file_list[class_id]:
		infile = os.path.join(cfg.source_dir, class_id, file_id, "models", "model_normalized.obj")
		outfile = os.path.join(cfg.processed_dir, class_id, file_id)
		if not os.path.isfile(infile):
			continue
		if os.path.isfile(outfile+'.ply'):
			processed_files += [f"{file_id}.ply"]
			if not cfg.overwrite:
				continue
		in_out_files += [(infile, outfile)]

print(f"Found {len(processed_files)} processed files...")
print(f"Need to process {len(in_out_files)} files...")

# %% Define how to run tetwild and run on remaining files
def run_tetwild(infile, outfile):
	subprocess.check_call([cfg.tetwild_bin, '-i', infile, '-o', outfile, *cfg.tetwild_args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	surface_file = outfile + "__tracked_surface.stl"
	mesh = meshio.read(outfile + "__tracked_surface.stl")
	mesh.write(outfile+'.ply')	

	# Remove unneeded files
	if cfg.cleanup:
		os.remove(outfile + "_.msh")
		os.remove(outfile + "_.csv")
		os.remove(outfile + "__sf.obj")
		os.remove(surface_file)

	print(f"Finished {surface_file}")

with mp.Pool(cfg.num_threads) as pool:
	list(pool.starmap(run_tetwild, in_out_files))

# %%
