# from hyve_utils import FemSim, extract_surface_mesh  # type: ignore
import pytorch_lightning as pl
from scipy.spatial.transform import Rotation as R  # type: ignore
from tqdm import tqdm  # type: ignore
import meshio  # type: ignore
import copy
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import FaceToEdge  # type: ignore
from torch_geometric.utils import from_trimesh  # type: ignore
from hyve.preprocess import *
from pathos import multiprocessing as mp  # type: ignore
import trimesh
import numpy as np
import torch
import os
import subprocess
import json
from hyve.dataset.dynamic_dataset import DynamicDataset
from string import Template
from math import ceil
import shutil

from typing import Any, Dict, Optional, Callable, List

ipc_base_config = Template(
    "shapes input 1\n"
    "$meshfile 0 0 0  $rotx $roty $rotz  1 1 1\n"
    "selfFric 0.1\n"
    "ground 0.1 $dropheight\n"
    "tol 1\n"
    "1e-1\n"
    "epsv 1e-1\n"
    "time 1.5 0.025"
)


class IPC(pl.LightningDataModule):
    def __init__(self, 
                 output_dir: str, 
                 batch_size: int = 4, 
                 batch_size_test: int = 4, 
                 num_workers: int = 4, 
                 z_sort_positions: bool = False, 
                 z_sort_cell_width: float = 5e-4,
                 knn_instead_of_mesh: Optional[int] = None, 
                 subsample: Optional[int] = None) -> None:
        super(IPC, self).__init__()

        # Dataset parameters
        self.output_dir = output_dir
        self.z_sort_positions = z_sort_positions
        self.z_sort_cell_width = z_sort_cell_width
        self.knn_instead_of_mesh = knn_instead_of_mesh
        self.subsample = subsample
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Stage either fit or test
        Split up the data depending on which stage should be executed
        Assume that output_dir has the structure
        output_dir/
        |- meta.json
        |- train/  Directory containing binary training data files
        |- val/    Directory containing binary validation data files
        |- test/   Directory containing binary testing data files
        --------------
        """
        print("Loading data from disk")
        # Load the data from file assuming it is called meta.json
        self.meta = {}
        with open(os.path.join(self.output_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)

    def train_dataloader(self):
        file_list = [os.path.join(self.output_dir, f) for f in self.meta["processed"]["train"]]
        return DataLoader(DynamicDataset(file_list, z_sort_positions=self.z_sort_positions, z_sort_cell_width=self.z_sort_cell_width, knn_instead_of_mesh=self.knn_instead_of_mesh, subsample=self.subsample), batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=False, shuffle=True)

    def val_dataloader(self):
        file_list = [os.path.join(self.output_dir, f) for f in self.meta["processed"]["val"]]
        return DataLoader(DynamicDataset(file_list, z_sort_positions=self.z_sort_positions, z_sort_cell_width=self.z_sort_cell_width, knn_instead_of_mesh=self.knn_instead_of_mesh, subsample=self.subsample), batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=False)

    def test_dataloader(self):
        file_list = [os.path.join(self.output_dir, f) for f in self.meta["processed"]["test"]]
        return DataLoader(DynamicDataset(file_list, z_sort_positions=self.z_sort_positions, z_sort_cell_width=self.z_sort_cell_width, knn_instead_of_mesh=self.knn_instead_of_mesh), batch_size=self.batch_size_test, num_workers=self.num_workers,
                          pin_memory=False)