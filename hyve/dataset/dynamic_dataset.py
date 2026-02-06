import pytorch_lightning as pl
from torch_geometric.data import Data
from hyve.preprocess import *
from pathos import multiprocessing as mp
import torch
from typing import Optional, List, Callable

# Generic dynamic dataset class for loading individual processed data files
class DynamicDataset:
    def __init__(self, file_list: List[str], 
                 z_sort_positions: bool = False, 
                 z_sort_cell_width: float = 5e-4,
                 subsample: Optional[int] = None,
                 knn_instead_of_mesh: Optional[int] = None,
                 noise_scale_normal: Optional[float] = None,
                 noise_scale_uniform: Optional[float] = None,
                 in_memory: bool = False,
                 load_callback: Callable[[str], ProcessedDataItem] = lambda *x: torch.load(*x, weights_only=False)):

        self.file_list = file_list
        self.z_sort_positions = z_sort_positions
        self.z_sort_cell_width = z_sort_cell_width
        self.subsample = subsample
        self.knn_instead_of_mesh = knn_instead_of_mesh
        self.noise_scale_normal = noise_scale_normal
        self.noise_scale_uniform = noise_scale_uniform
        self.in_memory = in_memory
        self.load_callback = load_callback

        self.data = []
        if self.in_memory:
            with mp.Pool(mp.cpu_count()//2) as pool:
                self.data = pool.map(self.load_callback, self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        if self.in_memory:
            data = self.data[item]
        else:
            data = self.load_callback(self.file_list[item])

        if self.z_sort_positions:
            data = z_sort_mesh_positions(data, self.z_sort_cell_width)
        if self.subsample is not None:
            """ Select a specific number of nodes and discard the rest.
            Only do this if there are more points than subsample implies."""
            n_sub = self.subsample
            if len(data['surface'].pos) > self.subsample:
                indices = torch.randint(0, len(data['surface'].pos), (self.subsample,), dtype=torch.long)
                data['surface'].update(Data(pos=data['surface'].pos[indices], x=data['surface'].x[indices] if data['surface'].x is not None else None))
        if self.knn_instead_of_mesh is not None:
            data = replace_mesh_with_knn_graph(data, self.knn_instead_of_mesh)
        # If noise scale is not None, add noise to the positions in the direction of the normals
        if self.noise_scale_normal is not None:
            # Generate offsets from -noise_scale to noise_scale
            noise_offsets = (torch.rand(data['surface'].pos.shape[0]) * 2 - 1) * self.noise_scale_normal
            data['surface'].update(Data(pos=data['surface'].pos + noise_offsets[:, None] * data['surface'].x))
        elif self.noise_scale_uniform is not None:
            # Generate random direction and offsets from -noise_scale to noise_scale
            noise_offsets = torch.nn.functional.normalize(torch.rand(data['surface'].pos.shape[0], 3) * 2 - 1) * (torch.rand(data['surface'].pos.shape[0]) * 2 - 1)[:, None] * self.noise_scale_uniform
            data['surface'].update(Data(pos=data['surface'].pos + noise_offsets))


        return data
