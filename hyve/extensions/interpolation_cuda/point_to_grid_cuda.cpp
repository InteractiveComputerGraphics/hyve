#include <torch/extension.h>

// Forward declaration of the CUDA kernel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_cuda_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid
);

// Forward operation definition
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid
) {
    return point_to_grid_cuda_forward(features, samples_flat, samples_batch, ngrid);
}

// Forward declaration of the backward kernel
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
);

// Backward operation for the forward pass
std::tuple<torch::Tensor, torch::Tensor>  point_to_grid_backward(
    const torch::Tensor grad_output,
    const torch::Tensor grad_counts,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid
) {
    const int d = features.size(1);
    const int npoints = features.size(0);

    // Call the backward kernel to compute gradients
    return point_to_grid_cuda_backward(
        grad_output,
        grad_counts,
        samples_flat,
        features,
        w,
        cluster_indices,
        ngrid,
        npoints,
        d
    );
}

// Forward declaration of the CUDA kernel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_min_weight_cuda_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    float min_weight
);

// Forward operation definition
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_min_weight_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    float min_weight
) {
    return point_to_grid_min_weight_cuda_forward(features, samples_flat, samples_batch, ngrid, min_weight);
}

// Forward declaration of the backward kernel
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
);

// Backward operation for the forward pass
std::tuple<torch::Tensor, torch::Tensor>  point_to_grid_min_weight_backward(
    const torch::Tensor grad_output,
    const torch::Tensor grad_counts,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid,
    float min_weight
) {
    const int d = features.size(1);
    const int npoints = features.size(0);

    // Call the backward kernel to compute gradients
    return point_to_grid_min_weight_cuda_backward(
        grad_output,
        grad_counts,
        samples_flat,
        features,
        w,
        cluster_indices,
        ngrid,
        npoints,
        d,
        min_weight
    );
}

// Forward declaration of the CUDA kernel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_masked_cuda_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    torch::Tensor mask
);

// Forward operation definition
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_masked_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    torch::Tensor mask
) {
    return point_to_grid_masked_cuda_forward(features, samples_flat, samples_batch, ngrid, mask);
}

// Forward declaration of the backward kernel
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
);

// Backward operation for the forward pass
std::tuple<torch::Tensor, torch::Tensor>  point_to_grid_masked_backward(
    const torch::Tensor grad_output,
    const torch::Tensor grad_counts,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid,
    const torch::Tensor mask
) {
    const int d = features.size(1);
    const int npoints = features.size(0);

    // Call the backward kernel to compute gradients
    return point_to_grid_masked_cuda_backward(
        grad_output,
        grad_counts,
        samples_flat,
        features,
        w,
        cluster_indices,
        ngrid,
        npoints,
        d,
        mask
    );
}


// Forward declaration of the CUDA kernel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_masked_weighted_cuda_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    float mask_value
);

// Forward operation definition
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_masked_weighted_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    float mask_value
) {
    return point_to_grid_masked_weighted_cuda_forward(features, samples_flat, samples_batch, ngrid, mask_value);
}

// Forward declaration of the backward kernel
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
);

// Backward operation for the forward pass
std::tuple<torch::Tensor, torch::Tensor>  point_to_grid_masked_weighted_backward(
    const torch::Tensor grad_output,
    const torch::Tensor grad_counts,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid,
    float mask_value,
    const torch::Tensor mask
) {
    const int d = features.size(1);
    const int npoints = features.size(0);

    // Call the backward kernel to compute gradients
    return point_to_grid_masked_weighted_cuda_backward(
        grad_output,
        grad_counts,
        samples_flat,
        features,
        w,
        cluster_indices,
        ngrid,
        npoints,
        d,
        mask_value,
        mask
    );
}

// Forward declaration of the sloped CUDA kernel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_sloped_cuda_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    float slope
);

// Forward operation definition
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_sloped_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid,
    float slope
) {
    return point_to_grid_sloped_cuda_forward(features, samples_flat, samples_batch, ngrid, slope);
}

// Forward declaration of the backward kernel
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
);

// Backward operation for the forward pass
std::tuple<torch::Tensor, torch::Tensor>  point_to_grid_sloped_backward(
    const torch::Tensor grad_output,
    const torch::Tensor grad_counts,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid,
    float slope
) {
    const int d = features.size(1);
    const int npoints = features.size(0);

    // Call the backward kernel to compute gradients
    return point_to_grid_sloped_cuda_backward(
        grad_output,
        grad_counts,
        samples_flat,
        features,
        w,
        cluster_indices,
        ngrid,
        npoints,
        d,
        slope
    );
}

// Forward declaration of the CUDA kernel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_attn_mask_cuda_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid
);

// Forward operation definition
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> point_to_grid_attn_mask_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid
) {
    return point_to_grid_attn_mask_cuda_forward(features, samples_flat, samples_batch, ngrid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &point_to_grid_forward, "Point To Grid Forward (CUDA)");
    m.def("backward", &point_to_grid_backward, "Point To Grid Backward (CUDA)");
    m.def("forward_minw", &point_to_grid_min_weight_forward, "Point To Grid Forward (CUDA) with Min Weight");
    m.def("backward_minw", &point_to_grid_min_weight_backward, "Point To Grid Backward (CUDA) with Min Weight");
    m.def("forward_masked", &point_to_grid_masked_forward, "Point To Grid Forward (CUDA) with mask");
    m.def("backward_masked", &point_to_grid_masked_backward, "Point To Grid Backward (CUDA) with mask");
    m.def("forward_masked_weighted", &point_to_grid_masked_weighted_forward, "Point To Grid Forward (CUDA) with mask");
    m.def("backward_masked_weighted", &point_to_grid_masked_weighted_backward, "Point To Grid Backward (CUDA) with mask");
    m.def("forward_sloped", &point_to_grid_sloped_forward, "Point To Grid Forward (CUDA) with slope");
    m.def("backward_sloped", &point_to_grid_sloped_backward, "Point To Grid Backward (CUDA) with slope");
    m.def("forward_attn_mask", &point_to_grid_attn_mask_forward, "Point To Grid Forward (CUDA) with computation of attention mask");
}