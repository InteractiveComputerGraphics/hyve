import os
import numpy as np
import pydiscregrid as pd  # type: ignore
import torch
import pickle
from torch_geometric.utils import to_undirected  # type: ignore
from torch_geometric.nn.pool import knn_graph
from torch_geometric.transforms import SamplePoints  # type: ignore
from torch_geometric.data import Data  # type: ignore
from typing import Union, Tuple, Type

from hyve.sdf_types import *


def move_to_origin(points: np.ndarray, return_midpoint: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    max = np.max(points, axis=0)
    min = np.min(points, axis=0)
    midpoint = 0.5 * (max + min)
    processed_points = points - midpoint
    if return_midpoint:
        return processed_points, midpoint
    return processed_points


def save_pickle(filename: str, object: Type) -> None:
    with open(filename, "wb") as f:
        pickle.dump(object, f)


def load_pickle(filename: str) -> Type:
    with open(filename, "rb") as f:
        return pickle.load(f)


def dir_is_good(directory: str) -> bool:
    try:
        os.makedirs(directory)
        return True
    except OSError:
            print(f"The directory {directory} already exists")
            return False


def get_scaling_factor(points: np.ndarray) -> float:
    # Computation from deep SDF paper. 1.03 Factor heuristically determined
    return np.max(np.linalg.norm(points, ord=2, axis=1)) * 1.03


def scale_to_unit_sphere(points: np.ndarray) -> np.ndarray:
    processed_points = move_to_origin(points)

    # Computation from deep SDF paper. 1.03 Factor heuristically determined
    radius = get_scaling_factor(processed_points)
    processed_points = processed_points / radius
    return processed_points


def scale_to_unit_cube(points: np.ndarray, return_transformation: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray, float]:
    if return_transformation:
        processed_points, midpoint = move_to_origin(points, True)
    else:
        processed_points = move_to_origin(points)

    # Computation from deep SDF paper. 1.25 Factor so that near surface sample points perturbed with maximum deviation
    # of 0.2 will still be within the unit cube. 1.25 is the same as scaling everything with 0.8
    max_distance_from_origin = 1.25 * np.max(np.abs(processed_points))
    processed_points = processed_points / max_distance_from_origin

    if return_transformation:
        return processed_points, midpoint, max_distance_from_origin

    return processed_points


def tetmesh_to_undirected(tetmesh: Union[torch.Tensor, torch.LongTensor, torch.IntTensor]) -> Union[torch.LongTensor, torch.Tensor]:
    """
    Convert a mesh consisting of tetrahedral elements to a list of edges of an undirected graph
    :param tetmesh: torch tensor (ntets x 4)
    :return: edge_index torch tensor (2 x nedges)
    """
    edges = torch.cat([tetmesh[:, [0, 1]], tetmesh[:, [1, 2]], tetmesh[:, [2, 3]]]).long().transpose(0, 1)
    edges = to_undirected(edges)
    return edges


def generate_sdf_samples_from_sample_points(points: np.ndarray, cells: np.ndarray, sample_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tri_mesh = pd.TriangleMesh(points, cells)

    md = pd.TriangleMeshDistance(tri_mesh)

    sd_normal = md.signed_distance_normal(sample_points)
    sd, normal = sd_normal[:, 0].reshape(-1, 1), sd_normal[:, 1:]

    return sd, normal


def generate_sdf_samples_unit_sphere(points: np.ndarray, cells: np.ndarray, samples: int = 10000, samples_ratio: Tuple[int, int, int] = (1, 1, 1)):
    """
    Generate `samples` number of samples from the surface mesh `mesh` using mesh distance implemented in discregrid
    This function generates the same amount of samples on the mesh with distance 0, close to the mesh within sd 0.1
    and uniformly in the unit sphere with sd up to 1.
    :param points: vertex positions of the surface mesh
    :param cells: surface mesh connectivity
    :param samples: number of samples to generate, this number may sometimes not be achieved as points are sampled
    :param samples_ratio: ratio of surface samples : near surface samples : uniform volume samples
    within the unit cube and discarded if they are outside of the unit sphere
    :return: (samples x 3, samples x 1) tuple consisting of sample points and sd values
    """
    tri_mesh = pd.TriangleMesh(points, cells)
    md = pd.TriangleMeshDistance(tri_mesh)

    # Per category samples
    per_cat_samples = [np.ceil(samples * n/np.sum(samples_ratio)).astype(np.int32) for n in samples_ratio]

    # Set up torch geometric data and sample points
    tri_pos = torch.from_numpy(points).float()
    tri_face = torch.from_numpy(cells).long().transpose(1, 0)
    tg_data = Data(pos=tri_pos, face=tri_face)

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    surface_samples = SamplePoints(max(per_cat_samples[0]+per_cat_samples[1], 1), True, True)(tg_data)  # ADDING 1 SAMPLE HERE SO I DONT HAVE TO CARE ABOUT SOME SPECIAL CASES
    surface_points = surface_samples.pos[:per_cat_samples[0]]
    surface_normal = surface_samples.normal[:per_cat_samples[0]]

    near_surface_points = surface_samples.pos[per_cat_samples[0]:] + torch.nn.functional.normalize(torch.rand(per_cat_samples[1], 3)-0.5, p=2) * torch.rand(per_cat_samples[1], 1) * 0.2

    uniform_points = np.random.uniform(-1, 1, size=[int(2.5 * per_cat_samples[2]), 3])
    filter = np.linalg.norm(uniform_points, ord=2, axis=1) <= 1.
    # Take all remaining points and filter to the correct number lateron
    uniform_points = uniform_points[filter][:]
    
    sd_normal = md.signed_distance_normal(np.concatenate([uniform_points, near_surface_points.numpy()]))
    sd, normal = sd_normal[:, 0].reshape(-1, 1), sd_normal[:, 1:]

    # Filter the points again and cut off at the desired sample amount
    sample_points = np.concatenate([surface_points.numpy(), uniform_points, near_surface_points.numpy()])[:samples]
    normal = np.concatenate([surface_normal.numpy(), normal])[:samples]
    sd = np.concatenate([np.zeros((surface_points.shape[0], 1)), sd])[:samples]

    torch.set_num_threads(prev_threads)
    return sample_points, sd, normal


def generate_sdf_samples_unit_cube(points: np.ndarray, cells: np.ndarray, samples: int = 10000, samples_ratio: Tuple[int, int, int] = (1, 1, 1)):
    """
    Generate `samples` number of samples from the surface mesh `mesh` using mesh distance implemented in discregrid
    This function generates the same amount of samples on the mesh with distance 0, close to the mesh within sd 0.1
    and uniformly in the unit sphere with sd up to 1.
    :param points: vertex positions of the surface mesh
    :param cells: surface mesh connectivity
    :param samples: number of samples to generate, this number may sometimes not be achieved as points are sampled
    :param samples_ratio: ratio of surface samples : near surface samples : uniform volume samples
    within the unit cube and discarded if they are outside of the unit sphere
    :return: (samples x 3, samples x 1) tuple consisting of sample points and sd values
    """
    tri_mesh = pd.TriangleMesh(points, cells)
    md = pd.TriangleMeshDistance(tri_mesh)

    # Per category samples
    per_cat_samples = [np.ceil(samples * n/np.sum(samples_ratio)).astype(np.integer) for n in samples_ratio]

    # Set up torch geometric data and sample points
    tri_pos = torch.from_numpy(points).float()
    tri_face = torch.from_numpy(cells).long().transpose(1, 0)
    tg_data = Data(pos=tri_pos, face=tri_face)

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    surface_samples = SamplePoints(max(per_cat_samples[0]+per_cat_samples[1], 1), True, True)(tg_data)  # ADDING 1 SAMPLE HERE SO I DONT HAVE TO CARE ABOUT SOME SPECIAL CASES
    surface_points = surface_samples.pos[:per_cat_samples[0]]
    surface_normal = surface_samples.normal[:per_cat_samples[0]]

    near_surface_points = surface_samples.pos[per_cat_samples[0]:] + torch.nn.functional.normalize(torch.rand(per_cat_samples[1], 3)-0.5, p=2) * torch.rand(per_cat_samples[1], 1) * 0.2

    uniform_points = np.random.uniform(-1, 1, size=[int(1.5 * per_cat_samples[2]), 3])
    # Take all remaining points and filter to the correct number lateron
    
    sd_normal = md.signed_distance_normal(np.concatenate([uniform_points, near_surface_points.numpy()]))
    sd, normal = sd_normal[:, 0].reshape(-1, 1), sd_normal[:, 1:]

    # Filter the points again and cut off at the desired sample amount
    sample_points = np.concatenate([surface_points.numpy(), uniform_points, near_surface_points.numpy()])[:samples]
    normal = np.concatenate([surface_normal.numpy(), normal])[:samples]
    sd = np.concatenate([np.zeros((surface_points.shape[0], 1)), sd])[:samples]

    torch.set_num_threads(prev_threads)
    return sample_points, sd, normal


def generate_sdf_samples_grid(points: np.ndarray, cells: np.ndarray, samples: int = 10000):
    tri_mesh = pd.TriangleMesh(points, cells)
    md = pd.TriangleMeshDistance(tri_mesh)

    grid_res = np.floor(np.cbrt(samples)).astype(np.int32)
    grid_p = np.linspace(-1, 1, grid_res)
    sample_points = np.array(np.meshgrid(grid_p, grid_p, grid_p, indexing="ij")).transpose(1, 2, 3, 0).reshape(-1, 3)
    sample_points = np.ascontiguousarray(sample_points)
    
    sd_normal = md.signed_distance_normal(sample_points)
    sd, normal = sd_normal[:, 0].reshape(-1, 1), sd_normal[:, 1:]

    return sample_points, sd, normal

def generate_grid(grid_res: int):
    grid_p = np.linspace(-1, 1, grid_res)
    sample_points = np.array(np.meshgrid(grid_p, grid_p, grid_p, indexing="ij")).transpose(1, 2, 3, 0).reshape(-1, 3)
    sample_points = np.ascontiguousarray(sample_points)

    return torch.from_numpy(sample_points)


def solve_shape_matching(rest_pos: np.array, deformed_pos: np.array) -> np.array:
    """
    Solve for the best rotation to match up two point clouds. Assume unit masses all around.
    """
    from hyve_utils import polar_decomposition_stable

    rest_cm = rest_pos.mean(axis=0)
    deformed_cm = deformed_pos.mean(axis=0)

    q = rest_pos - rest_cm
    p = deformed_pos - deformed_cm

    # Rest mat just cov matrix of q
    A = (q.reshape(-1, 3, 1) * q.reshape(-1, 1, 3)).sum(axis=0)

    # B matrix is cov matrix of p and q
    B = (p.reshape(-1, 3, 1) * q.reshape(-1, 1, 3)).sum(axis=0)

    # Inverse rest matrix times new cov matrix
    B = B @ np.linalg.inv(A)

    # Compute rotation from polar decomposition
    rot = polar_decomposition_stable(B)

    return rot

def morton_encode(pos: torch.Tensor) -> torch.Tensor:
    # Convert floating-point coordinates to 32-bit integers while maintaining precision
    # x_int = pos[:,0].float().view(torch.int32)
    # y_int = pos[:,1].float().view(torch.int32)
    # z_int = pos[:,2].float().view(torch.int32)
    x_int = pos[:,0]
    y_int = pos[:,1]
    z_int = pos[:,2]
    
    # Interleave the bits of x, y, and z to create the Morton code
    result = torch.zeros_like(x_int)
    for i in range(32):  # Assuming 32-bit integers
        result |= ((x_int & (1 << i)) << 2*i) | ((y_int & (1 << i)) << (2*i + 1)) | ((z_int & (1 << i)) << (2*i + 2))

    return result

def sort_by_morton(pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Encoding coordinates using the Morton curve
    morton_codes = morton_encode(pos)
    
    # Sorting the coordinates based on Morton codes
    _, sorted_indices = torch.sort(morton_codes)
    
    # Reordering the original coordinates
    sorted_pos = pos[sorted_indices]
    
    return sorted_pos, sorted_indices, morton_codes

def z_sort_mesh_positions(mesh: ProcessedDataItem, hash_cell_width: float = 5e-4) -> ProcessedDataItem:
    if isinstance(mesh["volume"], Data):
        # Do sorting for volume mesh
        hash_cells = torch.floor(mesh["volume"].pos / hash_cell_width).int()
        _, sort_indices, morton_code = sort_by_morton(hash_cells)
        reverse_sort = torch.zeros_like(sort_indices)
        reverse_sort[sort_indices] = torch.arange(0, len(sort_indices))
        mesh["volume"] = Data(edge_index=reverse_sort[mesh["volume"].edge_index], y=mesh['volume'].y, pos=mesh['volume'].pos[sort_indices])

    # meshio.write_points_cells("morton_code_volume.vtu", mesh["volume"].pos, {"line": mesh['volume'].edge_index.T}, point_data={"morton_code": morton_code[sort_indices]})

    if isinstance(mesh["surface"], Data):
        # Do sorting for surface mesh
        hash_cells = torch.floor(mesh["surface"].pos / hash_cell_width).int()
        _, sort_indices, morton_code = sort_by_morton(hash_cells)
        reverse_sort = torch.zeros_like(sort_indices)
        reverse_sort[sort_indices] = torch.arange(0, len(sort_indices))
        mesh["surface"] = Data(edge_index=reverse_sort[mesh["surface"].edge_index], y=mesh['surface'].y, pos=mesh["surface"].pos[sort_indices])

    return mesh

def replace_mesh_with_knn_graph(mesh: ProcessedDataItem, knn: int) -> ProcessedDataItem:
    if isinstance(mesh["volume"], Data):
        edge_index = knn_graph(mesh['volume'].pos, knn)
        mesh['volume'].update(Data(edge_index=edge_index))

    if isinstance(mesh["surface"], Data):
        edge_index = knn_graph(mesh['surface'].pos, knn)
        mesh['surface'].update(Data(edge_index=edge_index))

    return mesh