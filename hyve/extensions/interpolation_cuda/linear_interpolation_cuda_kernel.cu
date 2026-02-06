#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#include <ATen/ATen.h>
#include <iostream>

// CUDA kernel for linear interpolation
template <typename T, typename integer>
__global__ void linear_interpolation_kernel(
    const T* __restrict__ features,
    const T* __restrict__ samples_flat,
    const integer* __restrict__ samples_batch,
    T* __restrict__ i_z,
    integer* __restrict__ i_000,
    T* __restrict__ w,
    integer* __restrict__ cluster_indices,
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
	for (int j = 0; j < d; ++j){
		T result = 0.0f;
		for (int i = 0; i < 8; ++i) {
			result += weights[i] * features[d*(cluster_indices_flat[i] + samples_batch[tid] * (ngrid * ngrid * ngrid)) + j];
		}
		i_z[(d*tid) + j] = result;
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> linear_interpolation_cuda_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid
) {
    CHECK_INPUT(features);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(samples_batch);

    AT_ASSERTM(samples_flat.size(1) == 3, "Flat list of input points should have 3 coordinates (x, y, z)");

    const int npoints = samples_flat.size(0);
    const int d = features.size(1);

    torch::Tensor i_z = torch::zeros({npoints, d}, samples_flat.options());
    torch::Tensor i_000 = torch::zeros({npoints}, samples_batch.options());
    torch::Tensor w = torch::zeros({npoints * 8}, samples_flat.options());
    torch::Tensor cluster_indices = torch::zeros({npoints * 8}, samples_batch.options());

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES(samples_flat.scalar_type(), "linear_interpolation_forward_cuda", ([&]{
        linear_interpolation_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
            features.data_ptr<scalar_t>(),
            samples_flat.data_ptr<scalar_t>(),
            samples_batch.data_ptr<long>(),
            i_z.data_ptr<scalar_t>(),
            i_000.data_ptr<long>(),
            w.data_ptr<scalar_t>(),
            cluster_indices.data_ptr<long>(),
            ngrid,
            npoints,
            d
        );
    }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    // return i_z;
    return std::make_tuple(i_z, i_000, w, cluster_indices);
}

// CUDA kernel for the backward pass
template <typename T, typename integer>
__global__ void linear_interpolation_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ samples_flat,
    const T* __restrict__ features,
    const T* __restrict__ w,
    const integer* __restrict__ cluster_indices,
    T* __restrict__ grad_features,
    T* __restrict__ grad_lc,
    T* __restrict__ grad_w,
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
    }

    for (int i = 0; i < 8; ++i) {
        const T weight = w[(8 * tid) + i];
        const int cluster_index = cluster_indices[(8 * tid) + i];

        // Compute gradients for features
        for (int j = 0; j < d; ++j) {
            const T grad_feature_sum = weight * grad_output[(d * tid) + j];
            atomicAdd(static_cast<T*>(&grad_features[(d*cluster_index) + j]), static_cast<T>(grad_feature_sum));
            // atomicAdd(&grad_features[(d*cluster_index) + j], grad_feature_sum);

            // Compute gradient grad_w
            grad_w_[i] += grad_output[(d * tid) + j] * features[(d*cluster_index) + j];
        }
        grad_w[(8*tid)+i] = grad_w_[i];
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
                           // ... (similar pattern for other terms)
                           );
                    // ) * grad_output[(d * tid) + j];

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



    // Update gradients for features
    // for (int j = 0; j < d; ++j) {
    //     grad_features[(d * cluster_index) + j] = grad_feature_sum[j];
    // }
}

// Backward operation for the linear interpolation CUDA kernel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> linear_interpolation_cuda_backward(
    const torch::Tensor grad_output,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid,
    int npoints,
    int d
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(features);
    CHECK_INPUT(w);
    CHECK_INPUT(cluster_indices);

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

	torch::Tensor grad_features = torch::zeros_like(features).to(grad_output.device()); //, features.options()); //
    torch::Tensor grad_lc = torch::zeros({npoints, 3}, samples_flat.options());
    torch::Tensor grad_w = torch::zeros_like(w);

    // Call the backward kernel to compute gradients
    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "interpolation_backward_cuda", ([&]{
        linear_interpolation_backward_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
                grad_output.data_ptr<scalar_t>(),
                samples_flat.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                cluster_indices.data_ptr<long>(),
                grad_features.data_ptr<scalar_t>(),
                grad_lc.data_ptr<scalar_t>(),
                grad_w.data_ptr<scalar_t>(),
                ngrid,
                npoints,
                d
            );
        }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    return std::make_tuple(grad_features, grad_lc, grad_w);
}

// CUDA kernel for the backward pass of the backward function
template <typename T, typename integer>
__global__ void linear_interpolation_backward_backward_kernel(
    const T* __restrict__ grad_grad_features,
    const T* __restrict__ grad_grad_samples_flat,
    const T* __restrict__ grad_output,
    const T* __restrict__ samples_flat,
    const T* __restrict__ features,
    const T* __restrict__ w,
    const T* __restrict__ grad_w,
    const integer* __restrict__ cluster_indices,
    T* __restrict__ grad_grad_output,
    T* __restrict__ grad_samples_flat,
    T* __restrict__ grad_features,
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
    const T lc[3] = {
        min(max((samples_flat[(3*tid) + 0] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[0]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 1] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[1]), static_cast<T>(0.0)), static_cast<T>(1.0)),
        min(max((samples_flat[(3*tid) + 2] - static_cast<T>(-1.0)) / grid_delta - static_cast<T>(i_000_3d[2]), static_cast<T>(0.0)), static_cast<T>(1.0))
    };

    // Compute 1 - local coord
    const T ilc[3] = {
        static_cast<T>(1.0) - lc[0], 
        static_cast<T>(1.0) - lc[1], 
        static_cast<T>(1.0) - lc[2]
    };

    // Get grad_lc values
    const T d_d_samples[3] = {
        grad_grad_samples_flat[(3*tid) + 0]/grid_delta,
        grad_grad_samples_flat[(3*tid) + 1]/grid_delta,
        grad_grad_samples_flat[(3*tid) + 2]/grid_delta
    };

    // Compute gradients of lc wrt grad w
    const T d_lc_d_w[8] = {-d_d_samples[0] * ilc[1] * ilc[2] + ilc[0] * -d_d_samples[1] * ilc[2] + ilc[0] * ilc[1] * -d_d_samples[2],
                            d_d_samples[0] * ilc[1] * ilc[2] +  lc[0] * -d_d_samples[1] * ilc[2] +  lc[0] * ilc[1] * -d_d_samples[2],
                           -d_d_samples[0] *  lc[1] * ilc[2] + ilc[0] *  d_d_samples[1] * ilc[2] + ilc[0] *  lc[1] * -d_d_samples[2],
                            d_d_samples[0] *  lc[1] * ilc[2] +  lc[0] *  d_d_samples[1] * ilc[2] +  lc[0] *  lc[1] * -d_d_samples[2],
                           -d_d_samples[0] * ilc[1] *  lc[2] + ilc[0] * -d_d_samples[1] *  lc[2] + ilc[0] * ilc[1] *  d_d_samples[2],
                            d_d_samples[0] * ilc[1] *  lc[2] +  lc[0] * -d_d_samples[1] *  lc[2] +  lc[0] * ilc[1] *  d_d_samples[2],
                           -d_d_samples[0] *  lc[1] *  lc[2] + ilc[0] *  d_d_samples[1] *  lc[2] + ilc[0] *  lc[1] *  d_d_samples[2],
                            d_d_samples[0] *  lc[1] *  lc[2] +  lc[0] *  d_d_samples[1] *  lc[2] +  lc[0] *  lc[1] *  d_d_samples[2]
                          };
    
    // Init local grad_w
    T grad_w_[8];
    for (int i = 0; i < 8; i++)
        grad_w_[i] = grad_w_[(8*tid) + i];
    
    // Compute gradient of d_lc with respect to lc
    T d_lc_d_lc[3];
    //              1 wrt 0                                 2 wrt 0
    d_lc_d_lc[0] = -grad_w[0] * -d_d_samples[1] *  ilc[2] + -grad_w[0] *  ilc[1] * -d_d_samples[2] + 
                    grad_w[1] * -d_d_samples[1] *  ilc[2] +  grad_w[1] *  ilc[1] * -d_d_samples[2] + 
                   -grad_w[2] *  d_d_samples[1] *  ilc[2] + -grad_w[2] *   lc[1] * -d_d_samples[2] +
                    grad_w[3] *  d_d_samples[1] *  ilc[2] +  grad_w[3] *   lc[1] * -d_d_samples[2] +
                   -grad_w[4] * -d_d_samples[1] *   lc[2] + -grad_w[4] *  ilc[1] *  d_d_samples[2] +
                    grad_w[5] * -d_d_samples[1] *   lc[2] +  grad_w[5] *  ilc[1] *  d_d_samples[2] +
                   -grad_w[6] *  d_d_samples[1] *   lc[2] + -grad_w[6] *   lc[1] *  d_d_samples[2] +
                    grad_w[7] *  d_d_samples[1] *   lc[2] +  grad_w[7] *   lc[1] *  d_d_samples[2];
    
    //              0 wrt 1                                 2 wrt 1
    d_lc_d_lc[1] = -d_d_samples[0] * -grad_w[0] *  ilc[2] +  ilc[0] * -grad_w[0] * -d_d_samples[2] + 
                    d_d_samples[0] * -grad_w[1] *  ilc[2] +   lc[0] * -grad_w[1] * -d_d_samples[2] + 
                   -d_d_samples[0] *  grad_w[2] *  ilc[2] +  ilc[0] *  grad_w[2] * -d_d_samples[2] +
                    d_d_samples[0] *  grad_w[3] *  ilc[2] +   lc[0] *  grad_w[3] * -d_d_samples[2] +
                   -d_d_samples[0] * -grad_w[4] *   lc[2] +  ilc[0] * -grad_w[4] *  d_d_samples[2] +
                    d_d_samples[0] * -grad_w[5] *   lc[2] +   lc[0] * -grad_w[5] *  d_d_samples[2] +
                   -d_d_samples[0] *  grad_w[6] *   lc[2] +  ilc[0] *  grad_w[6] *  d_d_samples[2] +
                    d_d_samples[0] *  grad_w[7] *   lc[2] +   lc[0] *  grad_w[7] *  d_d_samples[2];

    //              0 wrt 2                                 1 wrt 2
    d_lc_d_lc[2] = -d_d_samples[0] *  ilc[1] * -grad_w[0] +  ilc[0] * -d_d_samples[1] * -grad_w[0] + 
                    d_d_samples[0] *  ilc[1] * -grad_w[1] +   lc[0] * -d_d_samples[1] * -grad_w[1] + 
                   -d_d_samples[0] *   lc[1] * -grad_w[2] +  ilc[0] *  d_d_samples[1] * -grad_w[2] +
                    d_d_samples[0] *   lc[1] * -grad_w[3] +   lc[0] *  d_d_samples[1] * -grad_w[3] +
                   -d_d_samples[0] *  ilc[1] *  grad_w[4] +  ilc[0] * -d_d_samples[1] *  grad_w[4] +
                    d_d_samples[0] *  ilc[1] *  grad_w[5] +   lc[0] * -d_d_samples[1] *  grad_w[5] +
                   -d_d_samples[0] *   lc[1] *  grad_w[6] +  ilc[0] *  d_d_samples[1] *  grad_w[6] +
                    d_d_samples[0] *   lc[1] *  grad_w[7] +   lc[0] *  d_d_samples[1] *  grad_w[7];

    // Compute interpolation as weighted summation
    T d_features_d_w[8];
    for (int i = 0; i < 8; i++)
        d_features_d_w[i] = 0;
    
	for (int j = 0; j < d; ++j){
        T grad_features_ = static_cast<T>(0.0);
        T grad_samples_flat_  = static_cast<T>(0.0);
        T grad_grad_output_  = static_cast<T>(0.0);
		for (int i = 0; i < 8; ++i) {
            const T weight = w[(8*tid) + i];
            const int cluster_index = cluster_indices[(8*tid)+i];

            // Compute derivative of d_features wrt grad_output
            grad_grad_output_ += weight * grad_grad_features[(d*cluster_index) + j];

            // Compute derivative of d_features wrt w
            d_features_d_w[i] += grad_grad_features[(d*cluster_index) + j] * grad_output[(d*tid) + j];

            // Compute derivative of grad_w wrt grad_output
			grad_grad_output_ += d_lc_d_w[i] * features[(d*cluster_index) + j];

			// Compute derivative of grad w wrt features
            const T sum_ = d_lc_d_w[i] * grad_output[(d * tid) + j];
            atomicAdd(static_cast<T*>(&grad_features[(d*cluster_index) + j]), static_cast<T>(sum_));
		}

		grad_grad_output[(d*tid) + j] = grad_grad_output_;
	}

    T d_w_d_lc[3];

    d_w_d_lc[0] = -d_features_d_w[0] *  ilc[1] *  ilc[2] + 
                   d_features_d_w[1] *  ilc[1] *  ilc[2] + 
                  -d_features_d_w[2] *   lc[1] *  ilc[2] + 
                   d_features_d_w[3] *   lc[1] *  ilc[2] + 
                  -d_features_d_w[4] *  ilc[1] *   lc[2] + 
                   d_features_d_w[5] *  ilc[1] *   lc[2] + 
                  -d_features_d_w[6] *   lc[1] *   lc[2] + 
                   d_features_d_w[7] *   lc[1] *   lc[2];

    d_w_d_lc[1] =  ilc[0] * -d_features_d_w[0] *  ilc[2] + 
                    lc[0] * -d_features_d_w[1] *  ilc[2] + 
                   ilc[0] *  d_features_d_w[2] *  ilc[2] + 
                    lc[0] *  d_features_d_w[3] *  ilc[2] + 
                   ilc[0] * -d_features_d_w[4] *   lc[2] + 
                    lc[0] * -d_features_d_w[5] *   lc[2] + 
                   ilc[0] *  d_features_d_w[6] *   lc[2] + 
                    lc[0] *  d_features_d_w[7] *   lc[2];

    d_w_d_lc[2] =  ilc[0] *  ilc[1] * -d_features_d_w[0] + 
                    lc[0] *  ilc[1] * -d_features_d_w[1] + 
                   ilc[0] *   lc[1] * -d_features_d_w[2] + 
                    lc[0] *   lc[1] * -d_features_d_w[3] + 
                   ilc[0] *  ilc[1] *  d_features_d_w[4] + 
                    lc[0] *  ilc[1] *  d_features_d_w[5] + 
                   ilc[0] *   lc[1] *  d_features_d_w[6] + 
                    lc[0] *   lc[1] *  d_features_d_w[7];

    for (int i = 0; i < 3; i++)
    {
        grad_samples_flat[(3*tid) + i] = d_w_d_lc[i]/grid_delta + d_lc_d_lc[i]/grid_delta;
    }
    
    
}

// Backward of the backward operation for the linear interpolation CUDA kernel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> linear_interpolation_cuda_backward_backward(
    const torch::Tensor grad_grad_features,
    const torch::Tensor grad_grad_samples_flat,
    const torch::Tensor grad_output,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor grad_w,
    const torch::Tensor cluster_indices,
    int ngrid,
    int npoints,
    int d
) {
    CHECK_INPUT(grad_grad_features);
    CHECK_INPUT(grad_grad_samples_flat);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(samples_flat);
    CHECK_INPUT(features);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_w);
    CHECK_INPUT(cluster_indices);

    // Define the number of threads per block and compute the number of blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (npoints + threadsPerBlock - 1) / threadsPerBlock;

    torch::Tensor grad_grad_output = torch::zeros_like(grad_output).to(grad_output.device());
    torch::Tensor grad_samples_flat = torch::zeros_like(samples_flat);
    torch::Tensor grad_features = torch::zeros_like(features);

    // Call the backward of the backward kernel to compute gradients
    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "interpolation_backward_backward_cuda", ([&]{
        linear_interpolation_backward_backward_kernel<scalar_t, long><<<blocksPerGrid, threadsPerBlock>>>(
                grad_grad_features.data_ptr<scalar_t>(),
                grad_grad_samples_flat.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                samples_flat.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                grad_w.data_ptr<scalar_t>(),
                cluster_indices.data_ptr<long>(),
                grad_grad_output.data_ptr<scalar_t>(),
                grad_samples_flat.data_ptr<scalar_t>(),
                grad_features.data_ptr<scalar_t>(),
                ngrid,
                npoints,
                d
            );
        }));

    // Synchronize to ensure all CUDA operations are complete
    cudaDeviceSynchronize();

    return std::make_tuple(grad_grad_output, grad_samples_flat, grad_features);
}
