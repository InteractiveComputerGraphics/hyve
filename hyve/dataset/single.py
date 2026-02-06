import os
import json
import torch
from typing import Optional
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import trimesh
from hyve.dataset.dynamic_dataset import DynamicDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pytorch_lightning as pl
from tqdm import tqdm

from hyve.preprocess import replace_mesh_with_knn_graph, scale_to_unit_cube


class Single(pl.LightningDataModule):
    """
    This dataset needs to be used with knn neighbor detection
    """
    def __init__(self, mesh_file: str, scaled_output_file: Optional[str] = None, subsample: Optional[int] = None, knn_instead_of_mesh: Optional[int] = None) -> None:

        super().__init__() 
        self.mesh_file = mesh_file 
        self.scaled_output_file = scaled_output_file
        self.subsample = subsample
        self.knn_instead_of_mesh = knn_instead_of_mesh
        self.data = self.load_gt_mesh(self.mesh_file, self.subsample, self.scaled_output_file)
        if self.knn_instead_of_mesh is not None:
            self.data = replace_mesh_with_knn_graph(self.data, self.knn_instead_of_mesh)
        self.save_hyperparameters()
    
    @staticmethod
    def load_gt_mesh(gt_mesh: str, subsample: Optional[int] = None, scaled_output_file: Optional[str] = None):
        mesh = trimesh.load(gt_mesh, force='mesh')
        mesh.vertices = scale_to_unit_cube(mesh.vertices)
        if scaled_output_file is not None:
            mesh.export(scaled_output_file)
        if subsample is not None:
            surf_samples, surf_faces = trimesh.sample.sample_surface(mesh, subsample)
            surf_normals = mesh.face_normals[surf_faces]
            data = Data(x=torch.from_numpy(surf_normals.copy()).float(), y=torch.rand((1,10,7)).float(), pos=torch.from_numpy(surf_samples.copy()).float())
        else:
            data = Data(x=torch.from_numpy(mesh.vertex_normals.copy()).float(), y=torch.rand((1,10,7)).float(), pos=torch.from_numpy(mesh.vertices.copy()).float())
        idx = 0

        return {'volume': 0, 'surface': data, 'id': int(idx)}
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader([self.data], batch_size=1, pin_memory=True, num_workers=1)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader([self.data], batch_size=1, pin_memory=True, num_workers=1)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader([self.data], batch_size=1, pin_memory=True, num_workers=1)



if __name__ == "__main__":
    import polyscope as ps
    import meshio
    import numpy as np

    data = Single("datasets/honey/honey.ply")

    train_data = data.train_dataloader()
    val_data = data.val_dataloader()
    test_data = data.test_dataloader()

    for data in tqdm(train_data):
        print(data)

    for data in tqdm(val_data):
        print(data)

    for data in tqdm(test_data):
        print(data)