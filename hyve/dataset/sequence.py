import os
import json
import re
import meshio
import numpy as np
import torch
from typing import List, Optional, Tuple
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import trimesh
# from hyve_utils import FemSim, extract_surface_mesh  # type: ignore
from hyve.dataset.dynamic_dataset import DynamicDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pytorch_lightning as pl
from tqdm import tqdm

from hyve.preprocess import replace_mesh_with_knn_graph, scale_to_unit_cube, solve_shape_matching

def extract_numbers(filename: str) -> List[int]:
    return [int(num) for num in re.findall(r'\d+', filename)]
def sort_key(filename: str) -> Tuple:
    return tuple(extract_numbers(filename))


class Sequence(pl.LightningDataModule):
    """
    This dataset needs to be used with knn neighbor detection
    """
    def __init__(self, input_dir: str, gt_mesh_file: str, scaled_output_file: Optional[str] = None, subsample: Optional[int] = None, knn_instead_of_mesh: Optional[int] = None) -> None:

        super().__init__() 
        self.input_dir = input_dir
        input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".vtu") or f.endswith(".ply") or f.endswith(".obj")], key=sort_key)
        self.input_files = [os.path.join(input_dir, f) for f in input_files]

        self.gt_mesh_file = gt_mesh_file 
        self.scaled_output_file = scaled_output_file
        self.subsample = subsample
        self.knn_instead_of_mesh = knn_instead_of_mesh
        self.data = [self.load_gt_mesh(file, self.subsample, idx, self.gt_mesh_file) for idx, file in enumerate(self.input_files)]
        self.transformations = [self.extract_transformation(file, self.gt_mesh_file) for file in self.input_files]
        if self.knn_instead_of_mesh is not None:
            self.data = [replace_mesh_with_knn_graph(d, self.knn_instead_of_mesh) for d in self.data]
        self.save_hyperparameters()
    
    @staticmethod
    def load_gt_mesh(gt_mesh: str, subsample: Optional[int] = None, idx: Optional[int] = None, gt_mesh_file: Optional[str] = None):
        mesh = trimesh.load(gt_mesh, force='mesh')
        mesh.vertices = scale_to_unit_cube(mesh.vertices)
        if gt_mesh_file is not None:
            gt_mesh = meshio.read(gt_mesh_file)
            sim = FemSim()
            sim.addMesh(gt_mesh.points, gt_mesh.cells_dict['tetra'])
            _, surface_id_map = sim.getRemappedSurfaceMesh()
            surface_points = gt_mesh.points[surface_id_map]
            R = solve_shape_matching(surface_points, mesh.vertices)
            mesh.vertices = np.matmul(mesh.vertices, R)
        if subsample is not None:
            surf_samples, surf_faces = trimesh.sample.sample_surface(mesh, subsample)
            surf_normals = mesh.face_normals[surf_faces]
            data = Data(x=torch.from_numpy(surf_normals.copy()).float(), y=torch.rand((1,10,7)).float(), pos=torch.from_numpy(surf_samples.copy()).float())
        else:
            data = Data(x=torch.from_numpy(mesh.vertex_normals.copy()).float(), y=torch.rand((1,10,7)).float(), pos=torch.from_numpy(mesh.vertices.copy()).float())
        

        return {'volume': 0, 'surface': data, 'id': 0 if idx is None else int(idx)}
    
    @staticmethod
    def extract_transformation(gt_mesh: str, ref_mesh: str) -> Tuple[np.ndarray, float, np.ndarray]:
        mesh = trimesh.load(gt_mesh, force='mesh')
        mesh.vertices, translation, scale = scale_to_unit_cube(mesh.vertices, True)
        gt_mesh = meshio.read(ref_mesh)
        sim = FemSim()
        sim.addMesh(gt_mesh.points, gt_mesh.cells_dict['tetra'])
        _, surface_id_map = sim.getRemappedSurfaceMesh()
        surface_points = gt_mesh.points[surface_id_map]
        R = solve_shape_matching(surface_points, mesh.vertices)
        mesh.vertices = np.matmul(mesh.vertices, R)

        return translation, scale, R

    def apply_inverse_transformations(self, other_sequence_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        other_input_files = sorted([f for f in os.listdir(other_sequence_dir) if f.endswith(".vtu") or f.endswith(".ply") or f.endswith(".obj")], key=sort_key)
        other_input_files = [os.path.join(other_sequence_dir, f) for f in other_input_files]
        assert len(other_input_files) == len(self.input_files), "The other sequence must have the same number of files as the original sequence."

        for idx, (file, (translation, scale, R)) in enumerate(zip(other_input_files, self.transformations)):
            mesh = meshio.read(file)
            mesh.points = np.matmul(mesh.points, R.T)
            if 'normals' in mesh.point_data:
                mesh.point_data['normals'] = np.matmul(mesh.point_data['normals'], R.T)
            mesh.points = mesh.points * scale
            mesh.points = mesh.points + translation
            output_file = os.path.join(output_dir, os.path.basename(file))
            meshio.write(output_file, mesh)
            print(f"Saved transformed mesh to {output_file}")
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.data, batch_size=1, pin_memory=False, num_workers=15)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.data, batch_size=1, pin_memory=False, num_workers=15)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.data, batch_size=1, pin_memory=False, num_workers=15)



if __name__ == "__main__":
    data = Sequence("/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/ipc_raw/long/", "/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/ipc_raw/mesh.msh", knn_instead_of_mesh=8)

    # data.apply_inverse_transformations("/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/BaselineLatentGridInvertedInterpolated/knn/version_7/reconstructions", "/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/BaselineLatentGridInvertedInterpolated/knn/version_7/reconstructions_transformed")
    # data.apply_inverse_transformations("/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/ONet/best/reconstructions", "/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/ONet/best/reconstructions_transformed")
    # data.apply_inverse_transformations("/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/ConvONet/best/reconstructions", "/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/ConvONet/best/reconstructions_transformed")
    # data.apply_inverse_transformations("/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/IFNet/best/reconstructions", "/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/IFNet/best/reconstructions_transformed")
    # data.apply_inverse_transformations("/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/Shape2VecSet/version_5/reconstructions", "/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/Shape2VecSet/version_5/reconstructions_transformed")
    data.apply_inverse_transformations("/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/POCO/best/reconstructions", "/local-hdd/sjeske/Documents/Projects/hyve/data/experiments/dragon_sequence_test/lightning_logs/POCO/best/reconstructions_transformed")

    train_data = data.train_dataloader()
    val_data = data.val_dataloader()
    test_data = data.test_dataloader()

    for data in tqdm(train_data):
        print(data)

    for data in tqdm(val_data):
        print(data)

    for data in tqdm(test_data):
        print(data)