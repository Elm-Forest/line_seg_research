#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define PI 3.14159265358979323846

// template initTab to support float/double
template <typename scalar_t>
void initTab(std::vector<scalar_t>& tabSin, std::vector<scalar_t>& tabCos, int numangle, int numrho, int H, int W)
{
    scalar_t irho = static_cast<scalar_t>(std::sqrt(H * H + W * W) + 1) / static_cast<scalar_t>(numrho - 1);
    scalar_t itheta = static_cast<scalar_t>(PI) / static_cast<scalar_t>(numangle);
    scalar_t angle = 0;
    for (int i = 0; i < numangle; ++i)
    {
        tabCos[i] = std::cos(angle) / irho;
        tabSin[i] = std::sin(angle) / irho;
        angle += itheta;
    }
}

// CUDA forward declarations (still extern "C")
std::vector<torch::Tensor> line_accum_cuda_forward(
    const torch::Tensor feat,
    const float* tabCos,
    const float* tabSin,
    torch::Tensor output,
    int numangle,
    int numrho);

std::vector<torch::Tensor> line_accum_cuda_backward(
    torch::Tensor grad_outputs,
    torch::Tensor grad_in,
    torch::Tensor feat,
    const float* tabCos,
    const float* tabSin,
    int numangle,
    int numrho);

std::vector<at::Tensor> line_accum_forward(
    const at::Tensor feat,
    at::Tensor output,
    int numangle,
    int numrho) {

    CHECK_INPUT(feat);
    CHECK_INPUT(output);

    const int H = feat.size(2);
    const int W = feat.size(3);

    at::ScalarType dtype = feat.scalar_type();

    std::vector<at::Tensor> out;

    AT_DISPATCH_FLOATING_TYPES(dtype, "initTab_forward", ([&] {
        std::vector<scalar_t> tabSin(numangle), tabCos(numangle);
        initTab<scalar_t>(tabSin, tabCos, numangle, numrho, H, W);
        out = line_accum_cuda_forward(
            feat,
            reinterpret_cast<const float*>(tabCos.data()),
            reinterpret_cast<const float*>(tabSin.data()),
            output,
            numangle,
            numrho);
    }));

    CHECK_CONTIGUOUS(out[0]);
    return out;
}

std::vector<torch::Tensor> line_accum_backward(
    torch::Tensor grad_outputs,
    torch::Tensor grad_inputs,
    torch::Tensor feat,
    int numangle,
    int numrho) {

    CHECK_INPUT(grad_outputs);
    CHECK_INPUT(grad_inputs);
    CHECK_INPUT(feat);

    const int H = feat.size(2);
    const int W = feat.size(3);

    at::ScalarType dtype = feat.scalar_type();
    std::vector<torch::Tensor> out;

    AT_DISPATCH_FLOATING_TYPES(dtype, "initTab_backward", ([&] {
        std::vector<scalar_t> tabSin(numangle), tabCos(numangle);
        initTab<scalar_t>(tabSin, tabCos, numangle, numrho, H, W);
        out = line_accum_cuda_backward(
            grad_outputs,
            grad_inputs,
            feat,
            reinterpret_cast<const float*>(tabCos.data()),
            reinterpret_cast<const float*>(tabSin.data()),
            numangle,
            numrho);
    }));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &line_accum_forward, "line features accumulating forward (CUDA)");
    m.def("backward", &line_accum_backward, "line features accumulating backward (CUDA)");
}