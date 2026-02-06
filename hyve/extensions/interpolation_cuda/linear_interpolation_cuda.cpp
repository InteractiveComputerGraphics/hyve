#include <torch/extension.h>

// Forward declaration of the CUDA kernel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> linear_interpolation_cuda_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid
);

// Forward operation definition
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> linear_interpolation_forward(
    torch::Tensor features,
    torch::Tensor samples_flat,
    torch::Tensor samples_batch,
    int ngrid
) {
    return linear_interpolation_cuda_forward(features, samples_flat, samples_batch, ngrid);
}

// Forward declaration of the backward kernel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> linear_interpolation_cuda_backward(
    const torch::Tensor grad_output,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid,
    int npoints,
    int d
);

// Backward operation for the forward pass
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>  linear_interpolation_backward(
    const torch::Tensor grad_output,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor cluster_indices,
    int ngrid
) {
    const int d = features.size(1);
    const int npoints = grad_output.size(0);

    // Call the backward kernel to compute gradients
    return linear_interpolation_cuda_backward(
        grad_output,
        samples_flat,
        features,
        w,
        cluster_indices,
        ngrid,
        npoints,
        d
    );
}

// Forward declaration of the double backward kernel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>  linear_interpolation_cuda_backward_backward(
    const torch::Tensor grad_grad_features,
    const torch::Tensor grad_grad_lc,
    const torch::Tensor grad_output,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor grad_w,
    const torch::Tensor cluster_indices,
    int ngrid,
    int npoints,
    int d
);

// Backward operation for the backward pass
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> linear_interpolation_backward_backward(
    const torch::Tensor grad_grad_features,
    const torch::Tensor grad_grad_lc,
    const torch::Tensor grad_output,
    const torch::Tensor samples_flat,
    const torch::Tensor features,
    const torch::Tensor w,
    const torch::Tensor grad_w,
    const torch::Tensor cluster_indices,
    int ngrid
){
    const int d = features.size(1);
    const int npoints = grad_output.size(0);

    // Call the backward kernel to compute gradients
    return linear_interpolation_cuda_backward_backward(
        grad_grad_features,
        grad_grad_lc,
        grad_output,
        samples_flat,
        features,
        w,
        grad_w,
        cluster_indices,
        ngrid,
        npoints,
        d
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &linear_interpolation_forward, "Linear Interpolation Forward (CUDA)");
    m.def("backward", &linear_interpolation_backward, "Linear Interpolation Backward (CUDA)");
    m.def("double_backward", &linear_interpolation_backward_backward, "Linear Interpolation Double Backward (CUDA)");
}