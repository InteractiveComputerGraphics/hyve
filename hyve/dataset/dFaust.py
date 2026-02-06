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


class DFaust(pl.LightningDataModule):
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
                 subsample_val: Optional[int] = None,
                 subsample_test: Optional[int] = None,
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
        self.subsample_val = subsample_val
        self.subsample_test = subsample_test

        self.in_memory = in_memory

        with open(train_split, 'r') as f:
            files = json.load(f)
            train_files_list = [os.path.join(self.data_dir, f"{sid}/{cat}/{file}.pt") for sid in files['scans'] for cat in files['scans'][sid] for file in files['scans'][sid][cat]]
            self.train_files_list = [file for file in train_files_list if os.path.isfile(file)]

        with open(test_split, 'r') as f:
            files = json.load(f)
            test_files_list = [os.path.join(self.data_dir, f"{sid}/{cat}/{file}.pt") for sid in files['scans'] for cat in files['scans'][sid] for file in files['scans'][sid][cat]]
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
        return DataLoader(DynamicDataset(self.train_files_list, knn_instead_of_mesh=self.knn_instead_of_mesh, subsample=self.subsample, in_memory=self.in_memory), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(DynamicDataset(self.test_files_list, knn_instead_of_mesh=self.knn_instead_of_mesh, subsample=self.subsample_val, in_memory=self.in_memory), batch_size=self.batch_size_test, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        if self.gt_mesh_files_list is None:
            return DataLoader(DynamicDataset(self.test_files_list, knn_instead_of_mesh=self.knn_instead_of_mesh, subsample=self.subsample_test, in_memory=self.in_memory), batch_size=self.batch_size_test, num_workers=self.num_workers, pin_memory=True)
        else:
            return DataLoader(DynamicDataset(self.gt_mesh_files_list, knn_instead_of_mesh=self.knn_instead_of_mesh, subsample=self.subsample_test, in_memory=self.in_memory, load_callback=DFaust.load_gt_mesh), batch_size=self.batch_size_test, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    import polyscope as ps
    import meshio
    import numpy as np

    data = DFaust("/local-hdd/sjeske/Data/processed/d-faust-scans/scans", "data/configs/d_faust/SAL/train_all_every5.json", "data/configs/d_faust/SAL/test_all_every5.json", num_workers=4, batch_size_test=1)

    train_data = data.train_dataloader()
    val_data = data.val_dataloader()
    test_data = data.test_dataloader()

    # Write a custom config to reconstruct the correct meshes for rendering
    processed_dir = "/local-hdd/sjeske/Data/processed/d-faust-scans/"
    filepath = "/local-hdd/sjeske/Documents/Projects/hyve/data/configs/d_faust/render.json"
    ids = [7980,7709,7209,6348,7897,7306,6300,7187]
    found_files = {'scans': {}}
    # Recurse through processed_dir
    for root, dirs, files in os.walk(os.path.join(processed_dir, 'scans')):
        for file in files:
            if file.endswith(".pt"):
                print(f"Testing {file}")
                data = torch.load(os.path.join(root, file))
                if data['id'] not in ids:
                    continue
                print(f"\t Found {file} in dir {root}")

    # Save test data as vtu points
    output_dir = "/local-hdd/sjeske/Data/processed/d-faust-scans/test_point_cloud/"
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
        if not torch.all(data['surface'].y[:,:,:3].isfinite() == True):
            print("Found nan in y pos!")
        if not torch.all(data['surface'].y[:,:,3].isfinite() == True):
            print("Found nan in y sd!")
        if not torch.all(data['surface'].y[:,:,4:7].isfinite() == True):
            print("Found nan in y normals!")
        if not torch.allclose(data['surface'].x.norm(dim=-1), torch.ones(len(data['surface'].x))):
            print("Found non normalized normal!")
        if not torch.all(data['surface'].pos > torch.Tensor([-1,-1,-1])) and torch.all(data['surface'].pos < torch.Tensor([1,1,1])):
            print("Not normalized!")

        pass

    for data in tqdm(val_data):
        if not torch.all(data['surface'].pos.isfinite() == True):
            print("Found nan in pos!")
        if not torch.all(data['surface'].x.isfinite() == True):
            print("Found nan in x!")
        if not torch.all(data['surface'].edge_index.isfinite() == True):
            print("Found nan in edge_index!")
        if not torch.all(data['surface'].y[:,:,:3].isfinite() == True):
            print("Found nan in y pos!")
        if not torch.all(data['surface'].y[:,:,3].isfinite() == True):
            print("Found nan in y sd!")
        if not torch.all(data['surface'].y[:,:,4:7].isfinite() == True):
            print("Found nan in y normals!")
        if not torch.allclose(data['surface'].x.norm(dim=-1), torch.ones(len(data['surface'].x))):
            print("Found non normalized normal!")
        if not torch.all(data['surface'].pos > torch.Tensor([-1,-1,-1])) and torch.all(data['surface'].pos < torch.Tensor([1,1,1])):
            print("Not normalized!")
        pass