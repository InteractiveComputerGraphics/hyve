import os
from time import time
import meshio
from torch import gt
import trimesh
from trimesh.proximity import closest_point
from trimesh.sample import sample_surface
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm
from hyve.dataset.dFaust import DFaust
from hyve.dataset.objaverse import Objaverse
from hyve.dataset.scannet import ScanNet
from hyve.postprocess import chamfer_distance, chamfer_distance_v2
import re


def make_grid_centers(bmin: np.ndarray, bmax: np.ndarray, res: Union[int, Tuple[int,int,int]]):
    """
    Compute grid centers for a 3D axis-aligned grid.
    Returns:
        xs, ys, zs: 1D arrays of centers along each axis
        voxel_size: 3-vector of voxel edge lengths
    """
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    if isinstance(res, int):
        res = (res, res, res)
    res = np.asarray(res, dtype=int)
    assert np.all(res > 0)
    voxel_size = (bmax - bmin) / res
    xs = bmin[0] + (np.arange(res[0]) + 0.5) * voxel_size[0]
    ys = bmin[1] + (np.arange(res[1]) + 0.5) * voxel_size[1]
    zs = bmin[2] + (np.arange(res[2]) + 0.5) * voxel_size[2]
    return xs, ys, zs, voxel_size


def _o3d_build_scene(mesh: trimesh.Trimesh):
    """
    Build Open3D RaycastingScene from a Trimesh mesh.
    """
    V = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
    F = o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int32))
    tmesh = o3d.t.geometry.TriangleMesh(V, F)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)
    return scene


def _o3d_point_to_mesh_distance(scene, points: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """
    Compute unsigned distances from points (N,3) to mesh in an Open3D RaycastingScene.
    """
    out = np.empty(points.shape[0], dtype=np.float32)
    for s in range(0, points.shape[0], chunk):
        e = min(s + chunk, points.shape[0])
        P = o3d.core.Tensor(points[s:e].astype(np.float32))
        d = scene.compute_distance(P)  # unsigned distances
        out[s:e] = d.numpy()
    return out

def _trimesh_point_to_mesh_distance(mesh: trimesh.Trimesh, points: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """
    Unsigned distances from points to mesh using trimesh.proximity.closest_point.
    """
    out = np.empty(points.shape[0], dtype=np.float64)
    for s in range(0, points.shape[0], chunk):
        e = min(s + chunk, points.shape[0])
        _, d, _ = closest_point(mesh, points[s:e])
        out[s:e] = d
    return out


def surface_thick_shell_voxelize_o3d(
    mesh: trimesh.Trimesh,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: Union[int, Tuple[int,int,int]],
    tau: float,
    crop_to_mesh_aabb: bool | Tuple[Tuple[int, int, int], Tuple[int, int, int]] = True
) -> Tuple[np.ndarray, np.ndarray, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """
    Create a boolean grid where a voxel is True if its center lies within tau of the mesh surface.

    Args:
        mesh: Trimesh mesh (non-watertight is OK).
        bbox_min, bbox_max: world-space bounds of evaluation (3,).
        resolution: grid resolution (int or (Nx,Ny,Nz)).
        tau: shell thickness in same units as mesh coordinates.
        crop_to_mesh_aabb: compute distances only near the mesh AABB (+/- tau) for speed.

    Returns:
        occ: boolean array of shape (Nx, Ny, Nz).
    """
    xs, ys, zs, vox = make_grid_centers(bbox_min, bbox_max, resolution)
    Nx, Ny, Nz = len(xs), len(ys), len(zs)
    occ = np.zeros((Nx, Ny, Nz), dtype=bool)

    # Precompute the region to process (mesh AABB expanded by tau)
    if isinstance(crop_to_mesh_aabb, bool) and crop_to_mesh_aabb:
        mmin, mmax = mesh.bounds  # (2,3)
        mmin = np.maximum(mmin - tau, bbox_min)
        mmax = np.minimum(mmax + tau, bbox_max)

        def idx_range(axis):
            # indices [i0, i1] inclusive that overlap [mmin, mmax] along this axis
            if axis == 0:
                coord = xs
                amin, amax = mmin[0], mmax[0]
            elif axis == 1:
                coord = ys
                amin, amax = mmin[1], mmax[1]
            else:
                coord = zs
                amin, amax = mmin[2], mmax[2]
            i0 = int(np.searchsorted(coord, amin - 1e-12, side='left'))
            i1 = int(np.searchsorted(coord, amax + 1e-12, side='right') - 1)
            i0 = max(0, min(i0, len(coord)-1))
            i1 = max(0, min(i1, len(coord)-1))
            if i1 < i0:
                return None
            return i0, i1

        xr = idx_range(0)
        yr = idx_range(1)
        zr = idx_range(2)
        if xr is None or yr is None or zr is None:
            return occ  # mesh is completely outside bbox
        xi0, xi1 = xr
        yi0, yi1 = yr
        zi0, zi1 = zr
    elif isinstance(crop_to_mesh_aabb, tuple):
        (xi0, yi0, zi0), (xi1, yi1, zi1) = crop_to_mesh_aabb
        assert 0 <= xi0 <= xi1 < Nx
        assert 0 <= yi0 <= yi1 < Ny
        assert 0 <= zi0 <= zi1 < Nz
    else:
        xi0, xi1 = 0, Nx - 1
        yi0, yi1 = 0, Ny - 1
        zi0, zi1 = 0, Nz - 1

    scene = _o3d_build_scene(mesh)

    # Process per z-slice to keep memory reasonable
    Xs = xs[xi0:xi1+1]
    Ys = ys[yi0:yi1+1]
    Zs = zs[zi0:zi1+1]
    # for kz, z in enumerate(zs[zi0:zi1+1], start=zi0):
    # Build all (x,y,z) points for this slice
    # XX, YY = np.meshgrid(Xs, Ys, indexing='xy')
    XX, YY, ZZ = np.meshgrid(Xs, Ys, Zs, indexing='xy')
    P = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    d = _o3d_point_to_mesh_distance(scene, P, chunk=1_000_000)
    # d = _trimesh_point_to_mesh_distance(mesh, P, chunk=100_000_000)
    shell = (d <= tau).reshape(XX.shape)
    # occ[xi0:xi1+1, yi0:yi1+1, kz] = shell
    occ = shell

    return d.reshape(XX.shape), occ, ((xi0, yi0, zi0), (xi1, yi1, zi1))

def iou_binary(A: np.ndarray, B: np.ndarray) -> float:
    """
    IoU for two boolean volumes A and B.
    """
    assert A.shape == B.shape and A.dtype == bool and B.dtype == bool
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    if union == 0:
        return 0.0  # both empty
    return float(inter) / float(union)

def sample_points_on_mesh(mesh: trimesh.Trimesh, n: int, seed: int = 42) -> np.ndarray:
    """
    Uniform-by-area sampling of points on a mesh surface (non-watertight is OK).
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    # rng = np.random.default_rng(seed)
    # return mesh.sample(n, evenly=False, random_state=rng)
    return sample_surface(mesh, n, seed=seed)[0]

def fscore_from_pointsets(
    P_pred: np.ndarray,
    P_gt: np.ndarray,
    tau: float | List[float]
) -> Tuple[float, float, float] | List[Tuple[float, float, float]]:
    """
    Compute F-score, precision, recall at threshold tau (same units as points).
    """
    tree_pred = cKDTree(P_pred)
    tree_gt   = cKDTree(P_gt)

    d_gt_to_pred, _ = tree_pred.query(P_gt, k=1, p=2, workers=-1)  # recall distances
    d_pred_to_gt, _ = tree_gt.query(P_pred, k=1, p=2, workers=-1)  # precision distances

    if isinstance(tau, float):
        tau = [tau]
    
    results = []
    for t in tau:
        recall = float((d_gt_to_pred < t).mean())
        precision = float((d_pred_to_gt < t).mean())
        if precision + recall == 0:
            F = 0.0
        else:
            F = 2 * precision * recall / (precision + recall)
        results.append((F, precision, recall))
    return results[0] if len(results) == 1 else results


def fscore_meshes(
    mesh_pred: trimesh.Trimesh,
    mesh_gt: trimesh.Trimesh,
    tau: float | List[float],
    n_samples: int = 100_000,
    seed: int = 42
) -> Tuple[float, float, float] | List[Tuple[float, float, float]]:
    """
    Sample points from each mesh and compute F-score at tau.
    """
    P_pred = sample_points_on_mesh(mesh_pred, n_samples, seed)
    P_gt   = sample_points_on_mesh(mesh_gt,   n_samples, seed+1)
    return fscore_from_pointsets(P_pred, P_gt, tau)

def compute_additional_metrics(pred_dir: str, gt_mesh_dir: str, dry_run: bool = False, overwrite: bool = False, fast_dev_run: Optional[int] = None, gt_points_loader=None):
    # Extract all consecutive numbers from the file name and then sort by all numbers
    def extract_numbers(filename: str) -> List[int]:
        return [int(num) for num in re.findall(r'\d+', filename)]
    def sort_key(filename: str) -> Tuple:
        return tuple(extract_numbers(filename))

    if not os.path.exists(os.path.join(pred_dir,"reconstructions")):
        print(f"Pred dir {pred_dir} does not contain reconstructions folder. Skipping...")
        return
    
    pred_meshes = sorted([f for f in os.listdir(os.path.join(pred_dir,"reconstructions")) if "256_sf" in f], key=sort_key)
    pred_meshes = [os.path.join(pred_dir,"reconstructions",f) for f in pred_meshes]

    gt_meshes = sorted([f for f in os.listdir(gt_mesh_dir) if f.endswith(".vtu") or f.endswith(".ply") or f.endswith(".obj")], key=sort_key)
    gt_meshes = [os.path.join(gt_mesh_dir,f) for f in gt_meshes]

    if len(pred_meshes) < len(gt_meshes):
        return
        # Try to find matching file indices
        pred_ids = [extract_numbers(os.path.basename(f))[-1] for f in pred_meshes]
        gt_ids = [extract_numbers(os.path.basename(f))[-1] for f in gt_meshes]
        matched_indices = [gt_ids.index(pid) for pid in pred_ids if pid in gt_ids]
        gt_meshes = [gt_meshes[i] for i in matched_indices]
        gt_points_loader = [gt_points_loader.dataset[i] for i in matched_indices] if gt_points_loader is not None else None

    assert len(pred_meshes) == len(gt_meshes)

    metrics = {"iou_01": [], "iou_001": [], "fscore_01": [], "fscore_001": [], "prec_01": [], "prec_001": [], "rec_01": [], "rec_001": [], "cd_l1": [], "cd_l2": []}

    if dry_run:
        print("Pred meshes: ", len(pred_meshes))
        print("GT meshes: ", len(gt_meshes))
        return
    
    if fast_dev_run is not None:
        pred_meshes = pred_meshes[:fast_dev_run]
        gt_meshes = gt_meshes[:fast_dev_run]

    output_metrics_file = os.path.join(pred_dir, "reconstructions", "metrics_revision.csv")

    if os.path.exists(output_metrics_file) and not overwrite:
        print(f"Metrics file {output_metrics_file} already exists. Skipping...")
        return

    iterable = zip(pred_meshes, gt_meshes)
    if gt_points_loader is not None:
        assert len(pred_meshes) == len(gt_points_loader)
        iterable = zip(pred_meshes, gt_meshes, gt_points_loader)

    for item in tqdm(iterable, total=len(pred_meshes)):
        if gt_points_loader is not None:
            pred_mesh, gt_mesh, gt_points = item
        else:
            pred_mesh, gt_mesh = item
            gt_points = None

        # print(pred_mesh)
        # print(gt_mesh)
        try:
            pred = meshio.read(pred_mesh)
        except OSError as e:
            print(f"Error reading predicted mesh {pred_mesh}: {e}. Skipping...")
            continue
        gt = meshio.read(gt_mesh)
        # print(pred.points.shape, gt.points.shape)
        # print(pred.cells_dict.keys(), gt.cells_dict.keys())
        # print(pred.cells_dict["triangle"].shape, gt.cells_dict["triangle"].shape)

        pred_tri = trimesh.Trimesh(vertices=pred.points, faces=pred.cells_dict["triangle"])
        gt_tri = trimesh.Trimesh(vertices=gt.points, faces=gt.cells_dict["triangle"])

        # Add timing to this
        start = time()
        gt_d, _, gt_bounds = surface_thick_shell_voxelize_o3d(gt_tri, [-1,-1,-1], [1,1,1], 256, 0.01, True)
        pred_d, _, _ = surface_thick_shell_voxelize_o3d(pred_tri, [-1,-1,-1], [1,1,1], 256, 0.01, gt_bounds)
        end = time()
        # print("Voxelization time: ", end - start)

        iou_01 = iou_binary(pred_d < 0.01, gt_d < 0.01)
        iou_001 = iou_binary(pred_d < 0.001, gt_d < 0.001)
        (fscore_01, prec_01, rec_01), (fscore_001, prec_001, rec_001)  = fscore_meshes(pred_tri, gt_tri, [0.01, 0.001], 100_000)
        # (fscore_001, prec_001, rec_001) = fscore_meshes(pred_tri, gt_tri, 0.001, 100_000)

        gt_sampled_points = gt.points
        if gt_points is not None:
            gt_sampled_points = gt_points['surface'].pos.numpy()

        # cd_l1_orig = chamfer_distance(pred.points, gt_sampled_points)
        cd_l1, cd_l2 = chamfer_distance_v2(pred.points, gt_sampled_points)
        # cd_l2 = chamfer_distance_v2(pred.points, gt_sampled_points, squared=True)

        # if cd_l1_orig != cd_l1:
        #     print("Warning: chamfer_distance and chamfer_distance_v2 give different results!")
        #     print("cd_l1_orig: ", cd_l1_orig)
        #     print("cd_l1: ", cd_l1)
        #     pass

        # print("IoU: ", iou_01, iou_001)
        # print("F-score: ", fscore_01, fscore_001)

        metrics["iou_01"].append(iou_01)
        metrics["iou_001"].append(iou_001)
        metrics["fscore_01"].append(fscore_01)
        metrics["fscore_001"].append(fscore_001)
        metrics["prec_01"].append(prec_01)
        metrics["prec_001"].append(prec_001)
        metrics["rec_01"].append(rec_01)
        metrics["rec_001"].append(rec_001)
        metrics["cd_l1"].append(cd_l1)
        metrics["cd_l2"].append(cd_l2)

        # break

    print("Average metrics: ")
    for k in metrics.keys():
        print(f"{k}: {np.mean(metrics[k])}")

    # Output metrics to csv file
    if fast_dev_run is None:
        df = pd.DataFrame(metrics)
        df.to_csv(output_metrics_file, index=False)

    pass

# Use same GT for ablation study
# paths['thingi10k_ablation']['gt'] = paths['thingi10k']['gt']

if __name__ == "__main__":
    print(f"Computing additional metrics.")
    # Path to predicted, reconstructed meshes
    pred_dir = ""
    # Path to ground truth surface meshes
    gt_dir = ""
    # Path to ground truth surface sampled points loader (optional)
    # Used to compute chamfer distance as a kind of "checksum" to see if the meshes are correct
    gt_points_loader = None
    compute_additional_metrics(pred_dir, gt_dir, dry_run=False, fast_dev_run=None, gt_points_loader=gt_points_loader)