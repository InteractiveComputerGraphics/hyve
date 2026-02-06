import os
import json
import torch
from typing import Optional
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import trimesh
from hyve.dataset.dynamic_dataset import DynamicDataset
from torch_geometric.loader import DataLoader  # type: ignore
from torch_geometric.data import Data
import pytorch_lightning as pl
from tqdm import tqdm


class Objaverse(pl.LightningDataModule):
    """
    This dataset needs to be used with knn neighbor detection
    """
    def __init__(self, 
                 data_dir: str, 
                 train_split: str, 
                 test_split: str, 
                 batch_size: int = 8, 
                 batch_size_test: int = 4, 
                 num_workers: int = 6,
                 knn_instead_of_mesh: int = 8,
                 gt_mesh_dir: Optional[str] = None,
                 subsample: Optional[int] = None,
                 subsample_test: Optional[int] = None,
                 save_input_pointcloud: Optional[str] = None,
                 in_memory: bool = False) -> None:

        super().__init__() 
        self.data_dir = data_dir 
        self.gt_mesh_dir = gt_mesh_dir
        self.train_split = train_split
        self.test_split = test_split

        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers

        self.knn_instead_of_mesh = knn_instead_of_mesh
        self.subsample = subsample
        self.subsample_test = subsample_test

        self.save_input_pointcloud = save_input_pointcloud
        self.in_memory = in_memory

        with open(train_split, 'r') as f:
            train_files_list = [os.path.join(self.data_dir, file) for file in json.load(f)]
            self.train_files_list = [file for file in train_files_list if os.path.isfile(file)]

        with open(test_split, 'r') as f:
            test_files_list = [os.path.join(self.data_dir, file) for file in json.load(f)]
            self.test_files_list = [file for file in test_files_list if os.path.isfile(file)]
        
        self.gt_mesh_files_list = None
        if self.gt_mesh_dir is not None and os.path.isdir(self.gt_mesh_dir) and os.listdir(self.gt_mesh_dir) != 0:
            self.gt_mesh_files_list = [os.path.join(self.gt_mesh_dir, file) for file in os.listdir(self.gt_mesh_dir)]
        
        print(f"Found {len(self.train_files_list)} train files\nFound {len(self.test_files_list)} test files.")

        if self.gt_mesh_files_list is not None:
            print(f"Using {len(self.gt_mesh_files_list)} ground truth mesh files for testing.")

        self.save_hyperparameters()
    
    @staticmethod
    def load_gt_mesh(gt_mesh: str):
        mesh = trimesh.load(gt_mesh, force='mesh')
        data = Data(x=torch.from_numpy(mesh.vertex_normals.copy()).float(), y=torch.rand((1,10,7)).float(), pos=torch.from_numpy(mesh.vertices.copy()).float())
        idx = os.path.splitext(os.path.basename(gt_mesh))[0].split("_")[-1]

        return {'volume': 0, 'surface': data, 'id': int(idx)}
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(DynamicDataset(self.train_files_list, knn_instead_of_mesh=self.knn_instead_of_mesh, subsample=self.subsample, in_memory=self.in_memory, save_input_pointcloud=self.save_input_pointcloud), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(DynamicDataset(self.test_files_list, knn_instead_of_mesh=self.knn_instead_of_mesh, in_memory=self.in_memory, save_input_pointcloud=self.save_input_pointcloud), batch_size=self.batch_size_test, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        if self.gt_mesh_files_list is None:
            return DataLoader(DynamicDataset(self.test_files_list, knn_instead_of_mesh=self.knn_instead_of_mesh, in_memory=self.in_memory, subsample=self.subsample_test, save_input_pointcloud=self.save_input_pointcloud), batch_size=self.batch_size_test, num_workers=self.num_workers, pin_memory=True)
        else:
            return DataLoader(DynamicDataset(self.gt_mesh_files_list, knn_instead_of_mesh=self.knn_instead_of_mesh, in_memory=self.in_memory, subsample=self.subsample_test, load_callback=Objaverse.load_gt_mesh, save_input_pointcloud=self.save_input_pointcloud), batch_size=self.batch_size_test, num_workers=self.num_workers, pin_memory=True)



if __name__ == "__main__":
    import polyscope as ps
    import meshio
    import numpy as np
    # data = Objaverse("/local-hdd/sjeske/Data/processed/objaverse/torch", "data/configs/objaverse/train_rel_2_2000.json", "data/configs/objaverse/test_rel_2_200.json", num_workers=4, gt_mesh_dir="/local-hdd/sjeske/Data/processed/objaverse/gt_meshes/")
    data = Objaverse("/local-hdd/sjeske/Data/processed/objaverse/torch", "data/configs/objaverse/train_rel_2_2000.json", "data/configs/objaverse/test_rel_2_200.json", num_workers=4, batch_size_test=1)

    train_data = data.train_dataloader()
    val_data = data.val_dataloader()
    test_data = data.test_dataloader()

    # Save test data as vtu points
    output_dir = "/local-hdd/sjeske/Data/processed/objaverse/test_point_cloud/"
    for data in tqdm(test_data):
        pos = data['surface'].pos        
        n = data['surface'].x.squeeze()
        meshio.write_points_cells(
            os.path.join(output_dir, f"pointcloud_test_{data['id'].item()}.vtu"),
            pos,
            [('vertex', np.arange(len(pos)).reshape(-1, 1))],
            point_data={'normals': n}
        )

    for data in tqdm(train_data):
        if not torch.all(data['surface'].pos.isfinite() == True):
            print("Found nan in pos!")
        if not torch.all(data['surface'].x.isfinite() == True):
            print("Found nan in x!")
        if not torch.all(data['surface'].edge_index.isfinite() == True):
            print("Found nan in edge_index!")
        if not torch.all(data['surface'].y.isfinite() == True):
            print("Found nan in y!")
        if not torch.allclose(data['surface'].x.norm(dim=-1), torch.ones(len(data['surface'].x))):
            print("Found non normalized normal!")
        if not torch.all(data['surface'].pos > torch.Tensor([-1,-1,-1])) and torch.all(data['surface'].pos < torch.Tensor([1,1,1])):
            print("Not normalized!")

        pass

    for data in tqdm(val_data):
        pass