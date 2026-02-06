#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_INTERNAL_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_INTERNAL_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#include <ATen/ATen.h>
#include <iostream>

// CUDA kernel for linear interpolation
template <typename T, typename integer>
__global__ void point_to_grid_kernel(
    const T* __restrict__ features,
    const T* __restrict__ samples_flat,
    const integer* __restrict__ samples_batch,
    T* __restrict__ i_z,
    integer* __restrict__ i_000,
    T* __restrict__ w,
    integer* __restrict__ cluster_indices,
    T* __restrict__ counts,
    int ngrid,
    int npoints,
    int d
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    const T grid_delta = 2.0f / static_cast<T>(ngrid - 1);
    const int index_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };

    // Compute base index
    const integer i_000_3d[3] = {
        // -1.0 is the minimum value of the grid instead of 0.0
        // clamp to safe bounds to avoid out of bounds access
        min(max(static_cast<integer>((samples_flat[(3*tid + 0)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 1)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 2)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2))
    };

    // Compute local coordinates
    const T local_coord_3d[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute 1 - local coord
    const T inv_local_coord[3] = {
        static_cast<T>(1.0) - local_coord_3d[0], 
        static_cast<T>(1.0) - local_coord_3d[1], 
        static_cast<T>(1.0) - local_coord_3d[2]
    };

    // Compute weights
    T weights[8] = {
        inv_local_coord[0] * inv_local_coord[1] * inv_local_coord[2],
         local_coord_3d[0] * inv_local_coord[1] * inv_local_coord[2],
        inv_local_coord[0] *  local_coord_3d[1] * inv_local_coord[2],
         local_coord_3d[0] *  local_coord_3d[1] * inv_local_coord[2],
        inv_local_coord[0] * inv_local_coord[1] *  local_coord_3d[2],
         local_coord_3d[0] * inv_local_coord[1] *  local_coord_3d[2],
        inv_local_coord[0] *  local_coord_3d[1] *  local_coord_3d[2],
         local_coord_3d[0] *  local_coord_3d[1] *  local_coord_3d[2]
    };

    // Compute cluster indices
    integer cluster_indices_3d[8][3];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_3d[i][0] = i_000_3d[0] + index_offsets[i][0];
        cluster_indices_3d[i][1] = i_000_3d[1] + index_offsets[i][1];
        cluster_indices_3d[i][2] = i_000_3d[2] + index_offsets[i][2];
    }

    // Convert 3D cluster indices to flat indices
    integer cluster_indices_flat[8];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_flat[i] = cluster_indices_3d[i][0] + ngrid * (cluster_indices_3d[i][1] + ngrid * cluster_indices_3d[i][2]);
    }

    // Compute interpolation as weighted summation
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < d; ++j){
            // Add contribution to the grid
			// i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j] += weights[i] * features[(d*tid) + j];
            atomicAdd(static_cast<T*>(&i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j]), static_cast<T>(weights[i] * features[(d*tid) + j]));
		}
        // Increment counter
        atomicAdd(static_cast<T*>(&counts[cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)]), static_cast<T>(weights[i]));
    }

    // Store the results
    i_000[tid] = cluster_indices_flat[0];

    // Store weights and cluster indices for the backward pass
    for (int i = 0; i < 8; ++i) {
        w[(8*tid) + i] = weights[i];
        cluster_indices[(8*tid) + i] = cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid);
    }
}

// Forward CUDA operation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_cuda_forward(
    torch::Tensor samples_features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid
) {
    CHECK_INPUT(samples_features);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(samples_batch);

    TORCH_INTERNAL_ASSERT(samples_flat.size(1) == 3, "Flat list of input points should have 3 coordinates (x, y, z)");

    const int npoints = samples_flat.size(0);
    const int d = samples_features.size(1);
    const int n_batch = samples_batch.max().item<int>() + 1;

    torch::Tensor i_z = torch::zeros({ngrid*ngrid*ngrid*n_batch, d}, samples_flat.options());
    torch::Tensor i_000 = torch::zeros({npoints}, samples_batch.options());
    torch::Tensor w = torch::zeros({npoints * 8}, samples_flat.options());
    torch::Tensor cluster_indices = torch::zeros({npoints * 8}, samples_batch.options());
    torch::Tensor counts = torch::zeros({ngrid*ngrid*ngrid*n_batch, 1}, samples_flat.options());

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES(samples_flat.scalar_type(), "point_to_grid_forward_cuda", ([&]{
        point_to_grid_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
            samples_features.data_ptr<scalar_t>(),
            samples_flat.data_ptr<scalar_t>(),
            samples_batch.data_ptr<long>(),
            i_z.data_ptr<scalar_t>(),
            i_000.data_ptr<long>(),
            w.data_ptr<scalar_t>(),
            cluster_indices.data_ptr<long>(),
            counts.data_ptr<scalar_t>(),
            ngrid,
            npoints,
            d
        );
    }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    // return i_z;
    return std::make_tuple(i_z, counts, i_000, w, cluster_indices);
}

// CUDA kernel for the backward pass
template <typename T, typename integer>
__global__ void point_to_grid_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ grad_counts,
    const T* __restrict__ samples_flat,
    const T* __restrict__ features,
    const T* __restrict__ w,
    const integer* __restrict__ cluster_indices,
    T* __restrict__ grad_features,
    T* __restrict__ grad_lc,
    // T* __restrict__ grad_w,
    int ngrid,
    int npoints,
    int d
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    // Init grid delta
    const T grid_delta = 2.0f/ (ngrid - 1.0f);

    // Compute base index
    const integer i_000_3d[3] = {
        min(max(static_cast<integer>((samples_flat[(3*tid + 0)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 1)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 2)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2))
    };

    // Compute local coordinates
    // Init shorthands for lc and ilc
    const T lc_[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute 1 - local coord
    const T ilc_[3] = {
        static_cast<T>(1.0) - lc_[0], 
        static_cast<T>(1.0) - lc_[1], 
        static_cast<T>(1.0) - lc_[2]
    };

    // Initialize intermediate gradient for w
    T grad_w_[8];
    for (int i = 0; i < 8; ++i) {
        grad_w_[i] = 0.0f;
        grad_w_[i] += grad_counts[cluster_indices[(8*tid) + i]];
    }
    
    // Compute gradient of weighted summation
    for (int j = 0; j < d; ++j){
        grad_features[(d*tid) + j] = 0;
        for (int i = 0; i < 8; ++i) {
            // Derivative wrt w
			grad_w_[i] += grad_output[d*(cluster_indices[(8*tid) + i]) + j] * features[(d*tid) + j];

            // Derivative wrt features
            grad_features[(d*tid) + j] += w[(8*tid) + i] * grad_output[d*(cluster_indices[(8*tid) + i]) + j];
		}

        // Increment counter
    }

    // Compute gradients for lc (x, y, z)
    const T grad_lc_x = (-grad_w_[0] * ilc_[1] * ilc_[2] +
                          grad_w_[1] * ilc_[1] * ilc_[2] +
                         -grad_w_[2] *  lc_[1] * ilc_[2] +
                          grad_w_[3] *  lc_[1] * ilc_[2] +
                         -grad_w_[4] * ilc_[1] *  lc_[2] +
                          grad_w_[5] * ilc_[1] *  lc_[2] +
                         -grad_w_[6] *  lc_[1] *  lc_[2] +
                          grad_w_[7] *  lc_[1] *  lc_[2] 
                           );

    const T grad_lc_y = (ilc_[0] * -grad_w_[0] * ilc_[2] +
                          lc_[0] * -grad_w_[1] * ilc_[2] +
                         ilc_[0] *  grad_w_[2] * ilc_[2] +
                          lc_[0] *  grad_w_[3] * ilc_[2] +
                         ilc_[0] * -grad_w_[4] *  lc_[2] +
                          lc_[0] * -grad_w_[5] *  lc_[2] +
                         ilc_[0] *  grad_w_[6] *  lc_[2] +
                          lc_[0] *  grad_w_[7] *  lc_[2] 
                           // ... (similar pattern for other terms)
                           );
                    // ) * grad_output[(d * tid) + j];

    const T grad_lc_z = (ilc_[0] * ilc_[1] * -grad_w_[0] +
                          lc_[0] * ilc_[1] * -grad_w_[1] +
                         ilc_[0] *  lc_[1] * -grad_w_[2] +
                          lc_[0] *  lc_[1] * -grad_w_[3] +
                         ilc_[0] * ilc_[1] *  grad_w_[4] +
                          lc_[0] * ilc_[1] *  grad_w_[5] +
                         ilc_[0] *  lc_[1] *  grad_w_[6] +
                          lc_[0] *  lc_[1] *  grad_w_[7] 
                           // ... (similar pattern for other terms)
                           );
                    // ) * grad_output[(d * tid) + j];
        

    // Update gradients for lc
    grad_lc[(3 * tid) + 0] = grad_lc_x/grid_delta;
    grad_lc[(3 * tid) + 1] = grad_lc_y/grid_delta;
    grad_lc[(3 * tid) + 2] = grad_lc_z/grid_delta;

}

// Backward operation for the linear interpolation CUDA kernel
std::tuple<torch::Tensor, torch::Tensor> point_to_grid_cuda_backward(
    const torch::Tensor grad_output,
    const torch::Tensor grad_counts,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid,
    int npoints,
    int d
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_counts);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(features);
    CHECK_INPUT(w);
    CHECK_INPUT(cluster_indices);

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

	torch::Tensor grad_features = torch::zeros_like(features).to(grad_output.device()); //, features.options()); //
    torch::Tensor grad_lc = torch::zeros({npoints, 3}, samples_flat.options());
    // torch::Tensor grad_w = torch::zeros_like(w);

    // Call the backward kernel to compute gradients
    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "point_to_grid_backward_cuda", ([&]{
        point_to_grid_backward_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
                grad_output.data_ptr<scalar_t>(),
                grad_counts.data_ptr<scalar_t>(),
                samples_flat.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                cluster_indices.data_ptr<long>(),
                grad_features.data_ptr<scalar_t>(),
                grad_lc.data_ptr<scalar_t>(),
                // grad_w.data_ptr<scalar_t>(),
                ngrid,
                npoints,
                d
            );
        }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    return std::make_tuple(grad_features, grad_lc);
}

// CUDA kernel for linear interpolation
template <typename T, typename integer>
__global__ void point_to_grid_min_weight_kernel(
    const T* __restrict__ features,
    const T* __restrict__ samples_flat,
    const integer* __restrict__ samples_batch,
    T* __restrict__ i_z,
    integer* __restrict__ i_000,
    T* __restrict__ w,
    integer* __restrict__ cluster_indices,
    T* __restrict__ counts,
    int ngrid,
    int npoints,
    int d,
    T min_weight
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    const T grid_delta = 2.0f / static_cast<T>(ngrid - 1);
    const int index_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };

    // Compute base index
    const integer i_000_3d[3] = {
        min(max(static_cast<integer>((samples_flat[(3*tid + 0)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 1)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 2)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2))
    };

    // Compute local coordinates
    const T local_coord_3d[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute 1 - local coord
    const T inv_local_coord[3] = {
        static_cast<T>(1.0) - local_coord_3d[0], 
        static_cast<T>(1.0) - local_coord_3d[1], 
        static_cast<T>(1.0) - local_coord_3d[2]
    };

    // Compute weights
    T weights[8] = {
        inv_local_coord[0] * inv_local_coord[1] * inv_local_coord[2],
         local_coord_3d[0] * inv_local_coord[1] * inv_local_coord[2],
        inv_local_coord[0] *  local_coord_3d[1] * inv_local_coord[2],
         local_coord_3d[0] *  local_coord_3d[1] * inv_local_coord[2],
        inv_local_coord[0] * inv_local_coord[1] *  local_coord_3d[2],
         local_coord_3d[0] * inv_local_coord[1] *  local_coord_3d[2],
        inv_local_coord[0] *  local_coord_3d[1] *  local_coord_3d[2],
         local_coord_3d[0] *  local_coord_3d[1] *  local_coord_3d[2]
    };

    // Compute cluster indices
    integer cluster_indices_3d[8][3];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_3d[i][0] = i_000_3d[0] + index_offsets[i][0];
        cluster_indices_3d[i][1] = i_000_3d[1] + index_offsets[i][1];
        cluster_indices_3d[i][2] = i_000_3d[2] + index_offsets[i][2];
    }

    // Convert 3D cluster indices to flat indices
    integer cluster_indices_flat[8];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_flat[i] = cluster_indices_3d[i][0] + ngrid * (cluster_indices_3d[i][1] + ngrid * cluster_indices_3d[i][2]);
    }

    // Compute interpolation as weighted summation
    for (int i = 0; i < 8; ++i) {
        if (weights[i] < min_weight)
            continue;
        for (int j = 0; j < d; ++j){
            // Add contribution to the grid
			// i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j] += weights[i] * features[(d*tid) + j];
            atomicAdd(static_cast<T*>(&i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j]), static_cast<T>(weights[i] * features[(d*tid) + j]));
		}
        // Increment counter
        atomicAdd(static_cast<T*>(&counts[cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)]), static_cast<T>(weights[i]));
    }

    // Store the results
    i_000[tid] = cluster_indices_flat[0];

    // Store weights and cluster indices for the backward pass
    for (int i = 0; i < 8; ++i) {
        w[(8*tid) + i] = weights[i];
        cluster_indices[(8*tid) + i] = cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid);
    }
}

// Forward CUDA operation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_min_weight_cuda_forward(
    torch::Tensor samples_features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    float min_weight
) {
    CHECK_INPUT(samples_features);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(samples_batch);

    TORCH_INTERNAL_ASSERT(samples_flat.size(1) == 3, "Flat list of input points should have 3 coordinates (x, y, z)");

    const int npoints = samples_flat.size(0);
    const int d = samples_features.size(1);
    const int n_batch = samples_batch.max().item<int>() + 1;

    torch::Tensor i_z = torch::zeros({ngrid*ngrid*ngrid*n_batch, d}, samples_flat.options());
    torch::Tensor i_000 = torch::zeros({npoints}, samples_batch.options());
    torch::Tensor w = torch::zeros({npoints * 8}, samples_flat.options());
    torch::Tensor cluster_indices = torch::zeros({npoints * 8}, samples_batch.options());
    torch::Tensor counts = torch::zeros({ngrid*ngrid*ngrid*n_batch, 1}, samples_flat.options());

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES(samples_flat.scalar_type(), "point_to_grid_min_weight_forward_cuda", ([&]{
        point_to_grid_min_weight_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
            samples_features.data_ptr<scalar_t>(),
            samples_flat.data_ptr<scalar_t>(),
            samples_batch.data_ptr<long>(),
            i_z.data_ptr<scalar_t>(),
            i_000.data_ptr<long>(),
            w.data_ptr<scalar_t>(),
            cluster_indices.data_ptr<long>(),
            counts.data_ptr<scalar_t>(),
            ngrid,
            npoints,
            d,
            min_weight
        );
    }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    // return i_z;
    return std::make_tuple(i_z, counts, i_000, w, cluster_indices);
}

// CUDA kernel for the backward pass
template <typename T, typename integer>
__global__ void point_to_grid_min_weight_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ grad_counts,
    const T* __restrict__ samples_flat,
    const T* __restrict__ features,
    const T* __restrict__ w,
    const integer* __restrict__ cluster_indices,
    T* __restrict__ grad_features,
    T* __restrict__ grad_lc,
    // T* __restrict__ grad_w,
    int ngrid,
    int npoints,
    int d,
    T min_weight
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    // Init grid delta
    const T grid_delta = 2.0f/ (ngrid - 1.0f);

    // Compute base index
    const integer i_000_3d[3] = {
        min(max(static_cast<integer>((samples_flat[(3*tid + 0)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 1)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 2)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2))
    };

    // Compute local coordinates
    // Init shorthands for lc and ilc
    const T lc_[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute 1 - local coord
    const T ilc_[3] = {
        static_cast<T>(1.0) - lc_[0], 
        static_cast<T>(1.0) - lc_[1], 
        static_cast<T>(1.0) - lc_[2]
    };

    // Initialize intermediate gradient for w
    T grad_w_[8];
    for (int i = 0; i < 8; ++i) {
        if (w[(8*tid) + i] < min_weight){
            grad_w_[i] = 0.0f;
        } else {
            grad_w_[i] = grad_counts[cluster_indices[(8*tid) + i]];
        }
    }
    
    // Compute gradient of weighted summation
    for (int j = 0; j < d; ++j){
        grad_features[(d*tid) + j] = 0;
        for (int i = 0; i < 8; ++i) {
            if (w[(8*tid) + i] < min_weight)
                continue;

            // Derivative wrt w
			grad_w_[i] += grad_output[d*(cluster_indices[(8*tid) + i]) + j] * features[(d*tid) + j];

            // Derivative wrt features
            grad_features[(d*tid) + j] += w[(8*tid) + i] * grad_output[d*(cluster_indices[(8*tid) + i]) + j];
		}

        // Increment counter
    }

    // Compute gradients for lc (x, y, z)
    const T grad_lc_x = (-grad_w_[0] * ilc_[1] * ilc_[2] +
                          grad_w_[1] * ilc_[1] * ilc_[2] +
                         -grad_w_[2] *  lc_[1] * ilc_[2] +
                          grad_w_[3] *  lc_[1] * ilc_[2] +
                         -grad_w_[4] * ilc_[1] *  lc_[2] +
                          grad_w_[5] * ilc_[1] *  lc_[2] +
                         -grad_w_[6] *  lc_[1] *  lc_[2] +
                          grad_w_[7] *  lc_[1] *  lc_[2] 
                           );

    const T grad_lc_y = (ilc_[0] * -grad_w_[0] * ilc_[2] +
                          lc_[0] * -grad_w_[1] * ilc_[2] +
                         ilc_[0] *  grad_w_[2] * ilc_[2] +
                          lc_[0] *  grad_w_[3] * ilc_[2] +
                         ilc_[0] * -grad_w_[4] *  lc_[2] +
                          lc_[0] * -grad_w_[5] *  lc_[2] +
                         ilc_[0] *  grad_w_[6] *  lc_[2] +
                          lc_[0] *  grad_w_[7] *  lc_[2] 
                           // ... (similar pattern for other terms)
                           );
                    // ) * grad_output[(d * tid) + j];

    const T grad_lc_z = (ilc_[0] * ilc_[1] * -grad_w_[0] +
                          lc_[0] * ilc_[1] * -grad_w_[1] +
                         ilc_[0] *  lc_[1] * -grad_w_[2] +
                          lc_[0] *  lc_[1] * -grad_w_[3] +
                         ilc_[0] * ilc_[1] *  grad_w_[4] +
                          lc_[0] * ilc_[1] *  grad_w_[5] +
                         ilc_[0] *  lc_[1] *  grad_w_[6] +
                          lc_[0] *  lc_[1] *  grad_w_[7] 
                           // ... (similar pattern for other terms)
                           );
                    // ) * grad_output[(d * tid) + j];
        

    // Update gradients for lc
    grad_lc[(3 * tid) + 0] = grad_lc_x/grid_delta;
    grad_lc[(3 * tid) + 1] = grad_lc_y/grid_delta;
    grad_lc[(3 * tid) + 2] = grad_lc_z/grid_delta;

}

// Backward operation for the linear interpolation CUDA kernel
std::tuple<torch::Tensor, torch::Tensor> point_to_grid_min_weight_cuda_backward(
    const torch::Tensor grad_output,
    const torch::Tensor grad_counts,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid,
    int npoints,
    int d,
    float min_weight
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_counts);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(features);
    CHECK_INPUT(w);
    CHECK_INPUT(cluster_indices);

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

	torch::Tensor grad_features = torch::zeros_like(features).to(grad_output.device()); //, features.options()); //
    torch::Tensor grad_lc = torch::zeros({npoints, 3}, samples_flat.options());
    // torch::Tensor grad_w = torch::zeros_like(w);

    // Call the backward kernel to compute gradients
    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "point_to_grid_min_weight_backward_cuda", ([&]{
        point_to_grid_min_weight_backward_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
                grad_output.data_ptr<scalar_t>(),
                grad_counts.data_ptr<scalar_t>(),
                samples_flat.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                cluster_indices.data_ptr<long>(),
                grad_features.data_ptr<scalar_t>(),
                grad_lc.data_ptr<scalar_t>(),
                // grad_w.data_ptr<scalar_t>(),
                ngrid,
                npoints,
                d,
                min_weight
            );
        }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    return std::make_tuple(grad_features, grad_lc);
}

// CUDA kernel for linear interpolation
template <typename T, typename integer>
__global__ void point_to_grid_masked_kernel(
    const T* __restrict__ features,
    const T* __restrict__ samples_flat,
    const integer* __restrict__ samples_batch,
    T* __restrict__ i_z,
    integer* __restrict__ i_000,
    T* __restrict__ w,
    integer* __restrict__ cluster_indices,
    T* __restrict__ counts,
    int ngrid,
    int npoints,
    int d,
    const bool* __restrict__ mask
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    const T grid_delta = 2.0f / static_cast<T>(ngrid - 1);
    const int index_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };

    // Compute base index
    const integer i_000_3d[3] = {
        min(max(static_cast<integer>((samples_flat[(3*tid + 0)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 1)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 2)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2))
    };

    // Compute local coordinates
    const T local_coord_3d[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute 1 - local coord
    const T inv_local_coord[3] = {
        static_cast<T>(1.0) - local_coord_3d[0], 
        static_cast<T>(1.0) - local_coord_3d[1], 
        static_cast<T>(1.0) - local_coord_3d[2]
    };

    // Compute weights
    T weights[8] = {
        inv_local_coord[0] * inv_local_coord[1] * inv_local_coord[2],
         local_coord_3d[0] * inv_local_coord[1] * inv_local_coord[2],
        inv_local_coord[0] *  local_coord_3d[1] * inv_local_coord[2],
         local_coord_3d[0] *  local_coord_3d[1] * inv_local_coord[2],
        inv_local_coord[0] * inv_local_coord[1] *  local_coord_3d[2],
         local_coord_3d[0] * inv_local_coord[1] *  local_coord_3d[2],
        inv_local_coord[0] *  local_coord_3d[1] *  local_coord_3d[2],
         local_coord_3d[0] *  local_coord_3d[1] *  local_coord_3d[2]
    };

    // Compute cluster indices
    integer cluster_indices_3d[8][3];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_3d[i][0] = i_000_3d[0] + index_offsets[i][0];
        cluster_indices_3d[i][1] = i_000_3d[1] + index_offsets[i][1];
        cluster_indices_3d[i][2] = i_000_3d[2] + index_offsets[i][2];
    }

    // Convert 3D cluster indices to flat indices
    integer cluster_indices_flat[8];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_flat[i] = cluster_indices_3d[i][0] + ngrid * (cluster_indices_3d[i][1] + ngrid * cluster_indices_3d[i][2]);
    }

    // Compute interpolation as weighted summation
    for (int i = 0; i < 8; ++i) {
        if (!mask[(8*tid) + i])
            continue;
        for (int j = 0; j < d; ++j){
            // Add contribution to the grid
			// i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j] += weights[i] * features[(d*tid) + j];
            atomicAdd(static_cast<T*>(&i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j]), static_cast<T>(weights[i] * features[(d*tid) + j]));
		}
        // Increment counter
        atomicAdd(static_cast<T*>(&counts[cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)]), static_cast<T>(weights[i]));
    }

    // Store the results
    i_000[tid] = cluster_indices_flat[0];

    // Store weights and cluster indices for the backward pass
    for (int i = 0; i < 8; ++i) {
        w[(8*tid) + i] = weights[i];
        cluster_indices[(8*tid) + i] = cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid);
    }
}

// Forward CUDA operation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_masked_cuda_forward(
    torch::Tensor samples_features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    torch::Tensor mask
) {
    CHECK_INPUT(samples_features);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(samples_batch);
    CHECK_INPUT(mask);

    TORCH_INTERNAL_ASSERT(samples_flat.size(1) == 3, "Flat list of input points should have 3 coordinates (x, y, z)");

    const int npoints = samples_flat.size(0);
    const int d = samples_features.size(1);
    const int n_batch = samples_batch.max().item<int>() + 1;

    torch::Tensor i_z = torch::zeros({ngrid*ngrid*ngrid*n_batch, d}, samples_flat.options());
    torch::Tensor i_000 = torch::zeros({npoints}, samples_batch.options());
    torch::Tensor w = torch::zeros({npoints * 8}, samples_flat.options());
    torch::Tensor cluster_indices = torch::zeros({npoints * 8}, samples_batch.options());
    torch::Tensor counts = torch::zeros({ngrid*ngrid*ngrid*n_batch, 1}, samples_flat.options());

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES(samples_flat.scalar_type(), "point_to_grid_masked_forward_cuda", ([&]{
        point_to_grid_masked_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
            samples_features.data_ptr<scalar_t>(),
            samples_flat.data_ptr<scalar_t>(),
            samples_batch.data_ptr<long>(),
            i_z.data_ptr<scalar_t>(),
            i_000.data_ptr<long>(),
            w.data_ptr<scalar_t>(),
            cluster_indices.data_ptr<long>(),
            counts.data_ptr<scalar_t>(),
            ngrid,
            npoints,
            d,
            mask.data_ptr<bool>()
        );
    }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    // return i_z;
    return std::make_tuple(i_z, counts, i_000, w, cluster_indices);
}

// CUDA kernel for the backward pass
template <typename T, typename integer>
__global__ void point_to_grid_masked_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ grad_counts,
    const T* __restrict__ samples_flat,
    const T* __restrict__ features,
    const T* __restrict__ w,
    const integer* __restrict__ cluster_indices,
    T* __restrict__ grad_features,
    T* __restrict__ grad_lc,
    // T* __restrict__ grad_w,
    int ngrid,
    int npoints,
    int d,
    const bool* __restrict__ mask
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    // Init grid delta
    const T grid_delta = 2.0f/ (ngrid - 1.0f);

    // Compute base index
    const integer i_000_3d[3] = {
        min(max(static_cast<integer>((samples_flat[(3*tid + 0)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 1)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 2)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2))
    };

    // Compute local coordinates
    // Init shorthands for lc and ilc
    const T lc_[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute 1 - local coord
    const T ilc_[3] = {
        static_cast<T>(1.0) - lc_[0], 
        static_cast<T>(1.0) - lc_[1], 
        static_cast<T>(1.0) - lc_[2]
    };

    // Initialize intermediate gradient for w
    T grad_w_[8];
    for (int i = 0; i < 8; ++i) {
        if (!mask[(8*tid) + i]){
            grad_w_[i] = 0.0f;
        } else {
            grad_w_[i] = grad_counts[cluster_indices[(8*tid) + i]];
        }
    }
    
    // Compute gradient of weighted summation
    for (int j = 0; j < d; ++j){
        grad_features[(d*tid) + j] = 0;
        for (int i = 0; i < 8; ++i) {
            if (!mask[(8*tid) + i])
                continue;

            // Derivative wrt w
			grad_w_[i] += grad_output[d*(cluster_indices[(8*tid) + i]) + j] * features[(d*tid) + j];

            // Derivative wrt features
            grad_features[(d*tid) + j] += w[(8*tid) + i] * grad_output[d*(cluster_indices[(8*tid) + i]) + j];
		}

        // Increment counter
    }

    // Compute gradients for lc (x, y, z)
    const T grad_lc_x = (-grad_w_[0] * ilc_[1] * ilc_[2] +
                          grad_w_[1] * ilc_[1] * ilc_[2] +
                         -grad_w_[2] *  lc_[1] * ilc_[2] +
                          grad_w_[3] *  lc_[1] * ilc_[2] +
                         -grad_w_[4] * ilc_[1] *  lc_[2] +
                          grad_w_[5] * ilc_[1] *  lc_[2] +
                         -grad_w_[6] *  lc_[1] *  lc_[2] +
                          grad_w_[7] *  lc_[1] *  lc_[2] 
                           );

    const T grad_lc_y = (ilc_[0] * -grad_w_[0] * ilc_[2] +
                          lc_[0] * -grad_w_[1] * ilc_[2] +
                         ilc_[0] *  grad_w_[2] * ilc_[2] +
                          lc_[0] *  grad_w_[3] * ilc_[2] +
                         ilc_[0] * -grad_w_[4] *  lc_[2] +
                          lc_[0] * -grad_w_[5] *  lc_[2] +
                         ilc_[0] *  grad_w_[6] *  lc_[2] +
                          lc_[0] *  grad_w_[7] *  lc_[2] 
                           // ... (similar pattern for other terms)
                           );
                    // ) * grad_output[(d * tid) + j];

    const T grad_lc_z = (ilc_[0] * ilc_[1] * -grad_w_[0] +
                          lc_[0] * ilc_[1] * -grad_w_[1] +
                         ilc_[0] *  lc_[1] * -grad_w_[2] +
                          lc_[0] *  lc_[1] * -grad_w_[3] +
                         ilc_[0] * ilc_[1] *  grad_w_[4] +
                          lc_[0] * ilc_[1] *  grad_w_[5] +
                         ilc_[0] *  lc_[1] *  grad_w_[6] +
                          lc_[0] *  lc_[1] *  grad_w_[7] 
                           // ... (similar pattern for other terms)
                           );
                    // ) * grad_output[(d * tid) + j];
        

    // Update gradients for lc
    grad_lc[(3 * tid) + 0] = grad_lc_x/grid_delta;
    grad_lc[(3 * tid) + 1] = grad_lc_y/grid_delta;
    grad_lc[(3 * tid) + 2] = grad_lc_z/grid_delta;

}

// Backward operation for the linear interpolation CUDA kernel
std::tuple<torch::Tensor, torch::Tensor> point_to_grid_masked_cuda_backward(
    const torch::Tensor grad_output,
    const torch::Tensor grad_counts,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid,
    int npoints,
    int d,
    const torch::Tensor mask
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_counts);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(features);
    CHECK_INPUT(w);
    CHECK_INPUT(cluster_indices);
    CHECK_INPUT(mask);

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

	torch::Tensor grad_features = torch::zeros_like(features).to(grad_output.device()); //, features.options()); //
    torch::Tensor grad_lc = torch::zeros({npoints, 3}, samples_flat.options());
    // torch::Tensor grad_w = torch::zeros_like(w);

    // Call the backward kernel to compute gradients
    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "point_to_grid_masked_backward_cuda", ([&]{
        point_to_grid_masked_backward_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
                grad_output.data_ptr<scalar_t>(),
                grad_counts.data_ptr<scalar_t>(),
                samples_flat.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                cluster_indices.data_ptr<long>(),
                grad_features.data_ptr<scalar_t>(),
                grad_lc.data_ptr<scalar_t>(),
                // grad_w.data_ptr<scalar_t>(),
                ngrid,
                npoints,
                d,
                mask.data_ptr<bool>()
            );
        }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    return std::make_tuple(grad_features, grad_lc);
}


// CUDA kernel for linear interpolation
template <typename T, typename integer>
__global__ void point_to_grid_masked_weighted_kernel(
    const T* __restrict__ features,
    const T* __restrict__ samples_flat,
    const integer* __restrict__ samples_batch,
    T* __restrict__ i_z,
    integer* __restrict__ i_000,
    T* __restrict__ w,
    integer* __restrict__ cluster_indices,
    T* __restrict__ counts,
    int ngrid,
    int npoints,
    int d,
    T mask_value,
    const T* __restrict__ mask
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    const T grid_delta = 2.0f / static_cast<T>(ngrid - 1);
    const int index_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };

    // Compute base index
    const integer i_000_3d[3] = {
        min(max(static_cast<integer>((samples_flat[(3*tid + 0)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 1)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 2)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2))
    };

    // Compute local coordinates
    const T local_coord_3d[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute 1 - local coord
    const T inv_local_coord[3] = {
        static_cast<T>(1.0) - local_coord_3d[0], 
        static_cast<T>(1.0) - local_coord_3d[1], 
        static_cast<T>(1.0) - local_coord_3d[2]
    };

    // Compute weights
    T weights[8] = {
        inv_local_coord[0] * inv_local_coord[1] * inv_local_coord[2],
         local_coord_3d[0] * inv_local_coord[1] * inv_local_coord[2],
        inv_local_coord[0] *  local_coord_3d[1] * inv_local_coord[2],
         local_coord_3d[0] *  local_coord_3d[1] * inv_local_coord[2],
        inv_local_coord[0] * inv_local_coord[1] *  local_coord_3d[2],
         local_coord_3d[0] * inv_local_coord[1] *  local_coord_3d[2],
        inv_local_coord[0] *  local_coord_3d[1] *  local_coord_3d[2],
         local_coord_3d[0] *  local_coord_3d[1] *  local_coord_3d[2]
    };

    // Compute cluster indices
    integer cluster_indices_3d[8][3];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_3d[i][0] = i_000_3d[0] + index_offsets[i][0];
        cluster_indices_3d[i][1] = i_000_3d[1] + index_offsets[i][1];
        cluster_indices_3d[i][2] = i_000_3d[2] + index_offsets[i][2];
    }

    // Convert 3D cluster indices to flat indices
    integer cluster_indices_flat[8];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_flat[i] = cluster_indices_3d[i][0] + ngrid * (cluster_indices_3d[i][1] + ngrid * cluster_indices_3d[i][2]);
    }

    // Compute interpolation as weighted summation
    for (int i = 0; i < 8; ++i) {
        // Make it more likely that a point is used my multiplying a value less than 1
        // Points with higher weights therefore have a lower chance of being masked
        if ((mask[(8*tid) + i] * (1 - weights[i])) >= mask_value)
            continue;
        for (int j = 0; j < d; ++j){
            // Add contribution to the grid
			// i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j] += weights[i] * features[(d*tid) + j];
            atomicAdd(static_cast<T*>(&i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j]), static_cast<T>(weights[i] * features[(d*tid) + j]));
		}
        // Increment counter
        atomicAdd(static_cast<T*>(&counts[cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)]), static_cast<T>(weights[i]));
    }

    // Store the results
    i_000[tid] = cluster_indices_flat[0];

    // Store weights and cluster indices for the backward pass
    for (int i = 0; i < 8; ++i) {
        w[(8*tid) + i] = weights[i];
        cluster_indices[(8*tid) + i] = cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid);
    }
}

// Forward CUDA operation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_masked_weighted_cuda_forward(
    torch::Tensor samples_features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    float mask_value
) {
    CHECK_INPUT(samples_features);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(samples_batch);

    TORCH_INTERNAL_ASSERT(samples_flat.size(1) == 3, "Flat list of input points should have 3 coordinates (x, y, z)");

    const int npoints = samples_flat.size(0);
    const int d = samples_features.size(1);
    const int n_batch = samples_batch.max().item<int>() + 1;

    torch::Tensor i_z = torch::zeros({ngrid*ngrid*ngrid*n_batch, d}, samples_flat.options());
    torch::Tensor i_000 = torch::zeros({npoints}, samples_batch.options());
    torch::Tensor w = torch::zeros({npoints * 8}, samples_flat.options());
    torch::Tensor cluster_indices = torch::zeros({npoints * 8}, samples_batch.options());
    torch::Tensor counts = torch::zeros({ngrid*ngrid*ngrid*n_batch, 1}, samples_flat.options());
    // torch::manual_seed(0);
    torch::Tensor mask = torch::rand({npoints, 8}, samples_flat.options());

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES(samples_flat.scalar_type(), "point_to_grid_masked_weighted_forward_cuda", ([&]{
        point_to_grid_masked_weighted_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
            samples_features.data_ptr<scalar_t>(),
            samples_flat.data_ptr<scalar_t>(),
            samples_batch.data_ptr<long>(),
            i_z.data_ptr<scalar_t>(),
            i_000.data_ptr<long>(),
            w.data_ptr<scalar_t>(),
            cluster_indices.data_ptr<long>(),
            counts.data_ptr<scalar_t>(),
            ngrid,
            npoints,
            d,
            mask_value,
            mask.data_ptr<scalar_t>()
        );
    }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    // return i_z;
    return std::make_tuple(i_z, counts, i_000, w, cluster_indices, mask);
}

// CUDA kernel for the backward pass
template <typename T, typename integer>
__global__ void point_to_grid_masked_weighted_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ grad_counts,
    const T* __restrict__ samples_flat,
    const T* __restrict__ features,
    const T* __restrict__ w,
    const integer* __restrict__ cluster_indices,
    T* __restrict__ grad_features,
    T* __restrict__ grad_lc,
    // T* __restrict__ grad_w,
    int ngrid,
    int npoints,
    int d,
    float mask_value,
    const T* __restrict__ mask
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    // Init grid delta
    const T grid_delta = 2.0f/ (ngrid - 1.0f);

    // Compute base index
    const integer i_000_3d[3] = {
        min(max(static_cast<integer>((samples_flat[(3*tid + 0)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 1)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 2)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2))
    };

    // Compute local coordinates
    // Init shorthands for lc and ilc
    const T lc_[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute 1 - local coord
    const T ilc_[3] = {
        static_cast<T>(1.0) - lc_[0], 
        static_cast<T>(1.0) - lc_[1], 
        static_cast<T>(1.0) - lc_[2]
    };

    // Initialize intermediate gradient for w
    T grad_w_[8];
    for (int i = 0; i < 8; ++i) {
        if ((mask[(8*tid) + i] * (1 - w[(8*tid) + i])) >= mask_value){
            grad_w_[i] = 0.0f;
        } else {
            grad_w_[i] = grad_counts[cluster_indices[(8*tid) + i]];
        }
    }
    
    // Compute gradient of weighted summation
    for (int j = 0; j < d; ++j){
        grad_features[(d*tid) + j] = 0;
        for (int i = 0; i < 8; ++i) {
            if ((mask[(8*tid) + i] * (1 - w[(8*tid) + i])) >= mask_value)
                continue;

            // Derivative wrt w
			grad_w_[i] += grad_output[d*(cluster_indices[(8*tid) + i]) + j] * features[(d*tid) + j];

            // Derivative wrt features
            grad_features[(d*tid) + j] += w[(8*tid) + i] * grad_output[d*(cluster_indices[(8*tid) + i]) + j];
		}

        // Increment counter
    }

    // Compute gradients for lc (x, y, z)
    const T grad_lc_x = (-grad_w_[0] * ilc_[1] * ilc_[2] +
                          grad_w_[1] * ilc_[1] * ilc_[2] +
                         -grad_w_[2] *  lc_[1] * ilc_[2] +
                          grad_w_[3] *  lc_[1] * ilc_[2] +
                         -grad_w_[4] * ilc_[1] *  lc_[2] +
                          grad_w_[5] * ilc_[1] *  lc_[2] +
                         -grad_w_[6] *  lc_[1] *  lc_[2] +
                          grad_w_[7] *  lc_[1] *  lc_[2] 
                           );

    const T grad_lc_y = (ilc_[0] * -grad_w_[0] * ilc_[2] +
                          lc_[0] * -grad_w_[1] * ilc_[2] +
                         ilc_[0] *  grad_w_[2] * ilc_[2] +
                          lc_[0] *  grad_w_[3] * ilc_[2] +
                         ilc_[0] * -grad_w_[4] *  lc_[2] +
                          lc_[0] * -grad_w_[5] *  lc_[2] +
                         ilc_[0] *  grad_w_[6] *  lc_[2] +
                          lc_[0] *  grad_w_[7] *  lc_[2] 
                           // ... (similar pattern for other terms)
                           );
                    // ) * grad_output[(d * tid) + j];

    const T grad_lc_z = (ilc_[0] * ilc_[1] * -grad_w_[0] +
                          lc_[0] * ilc_[1] * -grad_w_[1] +
                         ilc_[0] *  lc_[1] * -grad_w_[2] +
                          lc_[0] *  lc_[1] * -grad_w_[3] +
                         ilc_[0] * ilc_[1] *  grad_w_[4] +
                          lc_[0] * ilc_[1] *  grad_w_[5] +
                         ilc_[0] *  lc_[1] *  grad_w_[6] +
                          lc_[0] *  lc_[1] *  grad_w_[7] 
                           // ... (similar pattern for other terms)
                           );
                    // ) * grad_output[(d * tid) + j];
        

    // Update gradients for lc
    grad_lc[(3 * tid) + 0] = grad_lc_x/grid_delta;
    grad_lc[(3 * tid) + 1] = grad_lc_y/grid_delta;
    grad_lc[(3 * tid) + 2] = grad_lc_z/grid_delta;

}

// Backward operation for the linear interpolation CUDA kernel
std::tuple<torch::Tensor, torch::Tensor> point_to_grid_masked_weighted_cuda_backward(
    const torch::Tensor grad_output,
    const torch::Tensor grad_counts,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid,
    int npoints,
    int d,
    float mask_value,
    const torch::Tensor mask
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_counts);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(features);
    CHECK_INPUT(w);
    CHECK_INPUT(cluster_indices);
    CHECK_INPUT(mask);

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

	torch::Tensor grad_features = torch::zeros_like(features).to(grad_output.device()); //, features.options()); //
    torch::Tensor grad_lc = torch::zeros({npoints, 3}, samples_flat.options());
    // torch::Tensor grad_w = torch::zeros_like(w);

    // Call the backward kernel to compute gradients
    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "point_to_grid_masked_weighted_backward_cuda", ([&]{
        point_to_grid_masked_weighted_backward_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
                grad_output.data_ptr<scalar_t>(),
                grad_counts.data_ptr<scalar_t>(),
                samples_flat.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                cluster_indices.data_ptr<long>(),
                grad_features.data_ptr<scalar_t>(),
                grad_lc.data_ptr<scalar_t>(),
                // grad_w.data_ptr<scalar_t>(),
                ngrid,
                npoints,
                d,
                mask_value,
                mask.data_ptr<scalar_t>()
            );
        }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    return std::make_tuple(grad_features, grad_lc);
}

////////////////////////////////////////////////
/// Linear interpolation with slope
////////////////////////////////////////////////
template <typename T, typename integer>
__global__ void point_to_grid_sloped_kernel(
    const T* __restrict__ features,
    const T* __restrict__ samples_flat,
    const integer* __restrict__ samples_batch,
    T* __restrict__ i_z,
    integer* __restrict__ i_000,
    T* __restrict__ w,
    integer* __restrict__ cluster_indices,
    T* __restrict__ counts,
    int ngrid,
    int npoints,
    int d,
    T slope // should be between 0 and 1
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    const T grid_delta = 2.0f / static_cast<T>(ngrid - 1);
    const int index_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };

    // Compute base index
    const integer i_000_3d[3] = {
        // -1.0 is the minimum value of the grid instead of 0.0
        // clamp to safe bounds to avoid out of bounds access
        min(max(static_cast<integer>((samples_flat[(3*tid + 0)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 1)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 2)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2))
    };

    // Compute local coordinates
    const T local_coord_3d[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute sloped local coordinates
    const T local_coord_sloped[3] = {
        (1-slope) + slope * local_coord_3d[0],
        (1-slope) + slope * local_coord_3d[1],
        (1-slope) + slope * local_coord_3d[2]
    };

    // Compute 1 - slope * local coord
    const T inv_local_coord_sloped[3] = {
        static_cast<T>(1.0) - slope * local_coord_3d[0], 
        static_cast<T>(1.0) - slope * local_coord_3d[1], 
        static_cast<T>(1.0) - slope * local_coord_3d[2]
    };

    // Compute weights
    T weights[8] = {
        inv_local_coord_sloped[0] * inv_local_coord_sloped[1] * inv_local_coord_sloped[2],
            local_coord_sloped[0] * inv_local_coord_sloped[1] * inv_local_coord_sloped[2],
        inv_local_coord_sloped[0] *     local_coord_sloped[1] * inv_local_coord_sloped[2],
            local_coord_sloped[0] *     local_coord_sloped[1] * inv_local_coord_sloped[2],
        inv_local_coord_sloped[0] * inv_local_coord_sloped[1] *     local_coord_sloped[2],
            local_coord_sloped[0] * inv_local_coord_sloped[1] *     local_coord_sloped[2],
        inv_local_coord_sloped[0] *     local_coord_sloped[1] *     local_coord_sloped[2],
            local_coord_sloped[0] *     local_coord_sloped[1] *     local_coord_sloped[2]
    };

    // Compute cluster indices
    integer cluster_indices_3d[8][3];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_3d[i][0] = i_000_3d[0] + index_offsets[i][0];
        cluster_indices_3d[i][1] = i_000_3d[1] + index_offsets[i][1];
        cluster_indices_3d[i][2] = i_000_3d[2] + index_offsets[i][2];
    }

    // Convert 3D cluster indices to flat indices
    integer cluster_indices_flat[8];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_flat[i] = cluster_indices_3d[i][0] + ngrid * (cluster_indices_3d[i][1] + ngrid * cluster_indices_3d[i][2]);
    }

    // Compute interpolation as weighted summation
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < d; ++j){
            // Add contribution to the grid
			// i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j] += weights[i] * features[(d*tid) + j];
            atomicAdd(static_cast<T*>(&i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j]), static_cast<T>(weights[i] * features[(d*tid) + j]));
		}
        // Increment counter
        atomicAdd(static_cast<T*>(&counts[cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)]), static_cast<T>(weights[i]));
    }

    // Store the results
    i_000[tid] = cluster_indices_flat[0];

    // Store weights and cluster indices for the backward pass
    for (int i = 0; i < 8; ++i) {
        w[(8*tid) + i] = weights[i];
        cluster_indices[(8*tid) + i] = cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid);
    }
}

// Forward CUDA operation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_sloped_cuda_forward(
    torch::Tensor samples_features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    float slope
) {
    CHECK_INPUT(samples_features);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(samples_batch);

    TORCH_INTERNAL_ASSERT(samples_flat.size(1) == 3, "Flat list of input points should have 3 coordinates (x, y, z)");

    const int npoints = samples_flat.size(0);
    const int d = samples_features.size(1);
    const int n_batch = samples_batch.max().item<int>() + 1;

    torch::Tensor i_z = torch::zeros({ngrid*ngrid*ngrid*n_batch, d}, samples_flat.options());
    torch::Tensor i_000 = torch::zeros({npoints}, samples_batch.options());
    torch::Tensor w = torch::zeros({npoints * 8}, samples_flat.options());
    torch::Tensor cluster_indices = torch::zeros({npoints * 8}, samples_batch.options());
    torch::Tensor counts = torch::zeros({ngrid*ngrid*ngrid*n_batch, 1}, samples_flat.options());

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES(samples_flat.scalar_type(), "point_to_grid_sloped_forward_cuda", ([&]{
        point_to_grid_sloped_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
            samples_features.data_ptr<scalar_t>(),
            samples_flat.data_ptr<scalar_t>(),
            samples_batch.data_ptr<long>(),
            i_z.data_ptr<scalar_t>(),
            i_000.data_ptr<long>(),
            w.data_ptr<scalar_t>(),
            cluster_indices.data_ptr<long>(),
            counts.data_ptr<scalar_t>(),
            ngrid,
            npoints,
            d,
            slope
        );
    }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    // return i_z;
    return std::make_tuple(i_z, counts, i_000, w, cluster_indices);
}

// CUDA kernel for the backward pass
template <typename T, typename integer>
__global__ void point_to_grid_sloped_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ grad_counts,
    const T* __restrict__ samples_flat,
    const T* __restrict__ features,
    const T* __restrict__ w,
    const integer* __restrict__ cluster_indices,
    T* __restrict__ grad_features,
    T* __restrict__ grad_lc,
    // T* __restrict__ grad_w,
    int ngrid,
    int npoints,
    int d,
    T slope
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    // Init grid delta
    const T grid_delta = 2.0f/ (ngrid - 1.0f);

    // Compute base index
    const integer i_000_3d[3] = {
        min(max(static_cast<integer>((samples_flat[(3*tid + 0)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 1)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 2)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2))
    };

    // Compute local coordinates
    // Init shorthands for lc and ilc
    const T lc_[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute sloped local coordinates
    const T lc_s[3] = {
        (1-slope) + slope * lc_[0],
        (1-slope) + slope * lc_[1],
        (1-slope) + slope * lc_[2]
    };

    // Compute 1 - local coord
    const T ilc_s[3] = {
        static_cast<T>(1.0) - slope * lc_[0], 
        static_cast<T>(1.0) - slope * lc_[1], 
        static_cast<T>(1.0) - slope * lc_[2]
    };

    // Initialize intermediate gradient for w
    T grad_w_[8];
    for (int i = 0; i < 8; ++i) {
        grad_w_[i] = 0.0f;
        grad_w_[i] += grad_counts[cluster_indices[(8*tid) + i]];
    }
    
    // Compute gradient of weighted summation
    for (int j = 0; j < d; ++j){
        grad_features[(d*tid) + j] = 0;
        for (int i = 0; i < 8; ++i) {
            // Derivative wrt w
			grad_w_[i] += grad_output[d*(cluster_indices[(8*tid) + i]) + j] * features[(d*tid) + j];

            // Derivative wrt features
            grad_features[(d*tid) + j] += w[(8*tid) + i] * grad_output[d*(cluster_indices[(8*tid) + i]) + j];
		}

        // Increment counter
    }

    // Compute gradients for lc (x, y, z)
    const T grad_lc_x = (-slope * grad_w_[0] * ilc_s[1] * ilc_s[2] +
                          slope * grad_w_[1] * ilc_s[1] * ilc_s[2] +
                         -slope * grad_w_[2] *  lc_s[1] * ilc_s[2] +
                          slope * grad_w_[3] *  lc_s[1] * ilc_s[2] +
                         -slope * grad_w_[4] * ilc_s[1] *  lc_s[2] +
                          slope * grad_w_[5] * ilc_s[1] *  lc_s[2] +
                         -slope * grad_w_[6] *  lc_s[1] *  lc_s[2] +
                          slope * grad_w_[7] *  lc_s[1] *  lc_s[2] 
                           );

    const T grad_lc_y = (ilc_s[0] * -slope * grad_w_[0] * ilc_s[2] +
                          lc_s[0] * -slope * grad_w_[1] * ilc_s[2] +
                         ilc_s[0] *  slope * grad_w_[2] * ilc_s[2] +
                          lc_s[0] *  slope * grad_w_[3] * ilc_s[2] +
                         ilc_s[0] * -slope * grad_w_[4] *  lc_s[2] +
                          lc_s[0] * -slope * grad_w_[5] *  lc_s[2] +
                         ilc_s[0] *  slope * grad_w_[6] *  lc_s[2] +
                          lc_s[0] *  slope * grad_w_[7] *  lc_s[2] 
                           // ... (similar pattern for other terms)
                           );
                    // ) * grad_output[(d * tid) + j];

    const T grad_lc_z = (ilc_s[0] * ilc_s[1] * -slope * grad_w_[0] +
                          lc_s[0] * ilc_s[1] * -slope * grad_w_[1] +
                         ilc_s[0] *  lc_s[1] * -slope * grad_w_[2] +
                          lc_s[0] *  lc_s[1] * -slope * grad_w_[3] +
                         ilc_s[0] * ilc_s[1] *  slope * grad_w_[4] +
                          lc_s[0] * ilc_s[1] *  slope * grad_w_[5] +
                         ilc_s[0] *  lc_s[1] *  slope * grad_w_[6] +
                          lc_s[0] *  lc_s[1] *  slope * grad_w_[7] 
                           // ... (similar pattern for other terms)
                           );
                    // ) * grad_output[(d * tid) + j];
        

    // Update gradients for lc
    grad_lc[(3 * tid) + 0] = grad_lc_x/grid_delta;
    grad_lc[(3 * tid) + 1] = grad_lc_y/grid_delta;
    grad_lc[(3 * tid) + 2] = grad_lc_z/grid_delta;

}

// Backward operation for the linear interpolation CUDA kernel
std::tuple<torch::Tensor, torch::Tensor> point_to_grid_sloped_cuda_backward(
    const torch::Tensor grad_output,
    const torch::Tensor grad_counts,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid,
    int npoints,
    int d,
    float slope
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_counts);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(features);
    CHECK_INPUT(w);
    CHECK_INPUT(cluster_indices);

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

	torch::Tensor grad_features = torch::zeros_like(features).to(grad_output.device()); //, features.options()); //
    torch::Tensor grad_lc = torch::zeros({npoints, 3}, samples_flat.options());
    // torch::Tensor grad_w = torch::zeros_like(w);

    // Call the backward kernel to compute gradients
    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "point_to_grid_sloped_backward_cuda", ([&]{
        point_to_grid_sloped_backward_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
                grad_output.data_ptr<scalar_t>(),
                grad_counts.data_ptr<scalar_t>(),
                samples_flat.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                cluster_indices.data_ptr<long>(),
                grad_features.data_ptr<scalar_t>(),
                grad_lc.data_ptr<scalar_t>(),
                // grad_w.data_ptr<scalar_t>(),
                ngrid,
                npoints,
                d,
                slope
            );
        }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    return std::make_tuple(grad_features, grad_lc);
}

// CUDA kernel for linear interpolation
template <typename T, typename integer>
__global__ void point_to_grid_attn_mask_kernel(
    const T* __restrict__ features,
    const T* __restrict__ samples_flat,
    const integer* __restrict__ samples_batch,
    T* __restrict__ i_z,
    integer* __restrict__ i_000,
    T* __restrict__ w,
    integer* __restrict__ cluster_indices,
    T* __restrict__ counts,
    bool* attention_mask,
    int ngrid,
    int npoints,
    int d
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    const T grid_delta = 2.0f / static_cast<T>(ngrid - 1);
    const int index_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };

    // Compute base index
    const integer i_000_3d[3] = {
        // -1.0 is the minimum value of the grid instead of 0.0
        // clamp to safe bounds to avoid out of bounds access
        min(max(static_cast<integer>((samples_flat[(3*tid + 0)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 1)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2)),
        min(max(static_cast<integer>((samples_flat[(3*tid + 2)] - (-1.0)) / grid_delta), static_cast<integer>(0)), static_cast<integer>(ngrid- 2))
    };

    // Compute local coordinates
    const T local_coord_3d[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute 1 - local coord
    const T inv_local_coord[3] = {
        static_cast<T>(1.0) - local_coord_3d[0], 
        static_cast<T>(1.0) - local_coord_3d[1], 
        static_cast<T>(1.0) - local_coord_3d[2]
    };

    // Compute weights
    T weights[8] = {
        inv_local_coord[0] * inv_local_coord[1] * inv_local_coord[2],
         local_coord_3d[0] * inv_local_coord[1] * inv_local_coord[2],
        inv_local_coord[0] *  local_coord_3d[1] * inv_local_coord[2],
         local_coord_3d[0] *  local_coord_3d[1] * inv_local_coord[2],
        inv_local_coord[0] * inv_local_coord[1] *  local_coord_3d[2],
         local_coord_3d[0] * inv_local_coord[1] *  local_coord_3d[2],
        inv_local_coord[0] *  local_coord_3d[1] *  local_coord_3d[2],
         local_coord_3d[0] *  local_coord_3d[1] *  local_coord_3d[2]
    };

    // Compute cluster indices
    integer cluster_indices_3d[8][3];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_3d[i][0] = i_000_3d[0] + index_offsets[i][0];
        cluster_indices_3d[i][1] = i_000_3d[1] + index_offsets[i][1];
        cluster_indices_3d[i][2] = i_000_3d[2] + index_offsets[i][2];
    }

    // Convert 3D cluster indices to flat indices
    integer cluster_indices_flat[8];
    for (int i = 0; i < 8; ++i) {
        cluster_indices_flat[i] = cluster_indices_3d[i][0] + ngrid * (cluster_indices_3d[i][1] + ngrid * cluster_indices_3d[i][2]);
    }

    // Compute interpolation as weighted summation
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < d; ++j){
            // Add contribution to the grid
			// i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j] += weights[i] * features[(d*tid) + j];
            atomicAdd(static_cast<T*>(&i_z[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j]), static_cast<T>(weights[i] * features[(d*tid) + j]));
		}
        // Increment counter
        atomicAdd(static_cast<T*>(&counts[cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)]), static_cast<T>(weights[i]));
    }

    // Store the results
    i_000[tid] = cluster_indices_flat[0];

    // Store weights and cluster indices for the backward pass
    for (int i = 0; i < 8; ++i) {
        w[(8*tid) + i] = weights[i];
        cluster_indices[(8*tid) + i] = cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid);
        attention_mask[npoints*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + tid] = true;
    }
}

// Forward CUDA operation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_attn_mask_cuda_forward(
    torch::Tensor samples_features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid
) {
    CHECK_INPUT(samples_features);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(samples_batch);

    TORCH_INTERNAL_ASSERT(samples_flat.size(1) == 3, "Flat list of input points should have 3 coordinates (x, y, z)");

    const int npoints = samples_flat.size(0);
    const int d = samples_features.size(1);
    const int n_batch = samples_batch.max().item<int>() + 1;

    torch::Tensor i_z = torch::zeros({ngrid*ngrid*ngrid*n_batch, d}, samples_flat.options());
    torch::Tensor i_000 = torch::zeros({npoints}, samples_batch.options());
    torch::Tensor w = torch::zeros({npoints * 8}, samples_flat.options());
    torch::Tensor cluster_indices = torch::zeros({npoints * 8}, samples_batch.options());
    torch::Tensor counts = torch::zeros({ngrid*ngrid*ngrid*n_batch, 1}, samples_flat.options());
    torch::Tensor attention_mask = torch::zeros({ngrid*ngrid*ngrid*n_batch, npoints}, samples_flat.options().dtype(torch::kBool));

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES(samples_flat.scalar_type(), "point_to_grid_attn_mask_forward_cuda", ([&]{
        point_to_grid_attn_mask_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
            samples_features.data_ptr<scalar_t>(),
            samples_flat.data_ptr<scalar_t>(),
            samples_batch.data_ptr<long>(),
            i_z.data_ptr<scalar_t>(),
            i_000.data_ptr<long>(),
            w.data_ptr<scalar_t>(),
            cluster_indices.data_ptr<long>(),
            counts.data_ptr<scalar_t>(),
            attention_mask.data_ptr<bool>(),
            ngrid,
            npoints,
            d
        );
    }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    // return i_z;
    return std::make_tuple(i_z, counts, i_000, w, cluster_indices, attention_mask);
}