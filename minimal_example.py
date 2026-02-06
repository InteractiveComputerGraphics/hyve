import torch
import numpy as np
import trimesh
from skimage.measure import marching_cubes
from hyve.model.hyve import HYVE
from hyve.dataset.single import Single
from hyve.preprocess import generate_grid
from einops import pack
import sys
import os

def main():
    # Ensure we can find the checkpoints and data
    checkpoint_path = "models/objaverse/pic-normals.ckpt"
    mesh_path = "datasets/honey/honey.ply"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please download datasets/models first.")
        return

    if not os.path.exists(mesh_path):
        print(f"Error: Mesh not found at {mesh_path}")
        print("Please download datasets first.")
        return

    # 1. Load the Model
    print(f"Loading model from {checkpoint_path}...")
    # Loading from checkpoint automatically handles hyperparameters
    model = HYVE.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda()

    # 2. Load the Dataset
    print(f"Loading data from {mesh_path}...")
    # Using Single dataset to preprocess (scale to unit cube, sample surface points, etc.)
    # Single dataset creates a dataloader with one Data object.
    # Be sure to check the logic here for your own data!
    dataset = Single(
        mesh_file=mesh_path, 
        subsample=200000, 
        # 4 was used with objaverse, but 8 with all other checkpoints
        # 8 also works for objaverse though if you want to try
        # knn_instead_of_mesh=4
        knn_instead_of_mesh=8
    )
    
    # Let's extract the surface data which is the relevant geometric data
    data_dict = dataset.data
    data = data_dict['surface']
    
    # Move to GPU
    data = data.cuda()
    
    # Add batch index (all zeros for single sample)
    data.batch = torch.zeros(data.pos.shape[0], dtype=torch.long, device=data.pos.device)

    # 3. Reference and Encode
    print("Encoding geometry...")
    with torch.no_grad():
        # Prepare input position
        pos = data.pos
        # If encoder uses normals, pack them
        if model.hparams.get('encoder_use_normals', False) and hasattr(data, 'x'):
            # data.x usually contains normals in Single dataset
            pos, _ = pack([pos, data.x], "n *")
        
        # Encode
        # encode(pos, edge_index, batch)
        # Note: edge_index comes from knn graph created in Single init
        latents = model.encode(pos, data.edge_index, data.batch)

    # 4. Inference on Dense Grid
    grid_res = 128
    print(f"Decoding on {grid_res}^3 grid...")
    
    with torch.no_grad():
        # Generate grid points in [-1, 1]
        sdf_pos = generate_grid(grid_res).cuda().float()
        
        # Process in chunks to avoid OOM
        chunk_size = 100000 # Adjust based on GPU memory
        sdf_vals_list = []
        
        num_chunks = (sdf_pos.shape[0] + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, sdf_pos.shape[0])
            
            chunk_pos = sdf_pos[start_idx:end_idx]

            # samples_batch corresponds to which batch item the query belongs to.
            # Here we have only 1 item in batch (index 0).
            chunk_samples_batch = torch.zeros(chunk_pos.shape[0], dtype=torch.long, device=model.device)
            
            # decode(samples, samples_batch, latents)
            # samples expects shape [Batch, Points, 3] 
            # so samples can be passed as [1, N, 3]
            chunk_pos_input = chunk_pos.unsqueeze(0) # [1, N, 3]
            lod_outputs = model.decode(chunk_pos_input, chunk_samples_batch, latents)
            
            # Use the highest level of detail (only the highest is actually computed to save memory by default)
            final_sdf = lod_outputs[-1]
            sdf_vals_list.append(final_sdf.squeeze().cpu())
            
        sdf_vals = torch.cat(sdf_vals_list, dim=0)

    # 5. Reconstruct Surface
    print("Running Marching Cubes...")
    sdf_volume = sdf_vals.numpy().reshape(grid_res, grid_res, grid_res)
    
    # Grid in model is [-1, 1]. Size is 2. Spacing is 2 / (res - 1).
    spacing = 2.0 / (grid_res - 1)
    
    # Extract mesh
    verts, faces, normals, values = marching_cubes(sdf_volume, level=0.0, spacing=(spacing, spacing, spacing))
    
    # Fix coordinates: marching_cubes returns coordinates starting at 0.
    # We need to shift to match [-1, 1] range.
    # The first point (index 0) corresponds to -1.
    verts = verts - 1.0
    
    # 6. Save Mesh
    output_filename = "honey_reconstructed.ply"
    print(f"Saving mesh to {output_filename}...")
    
    # Invert normals if needed (sometimes SDF models output inverted sign)
    normals = -normals
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export(output_filename)
    
    print("Done!")

if __name__ == "__main__":
    main()
