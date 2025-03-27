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

// CUDA 前向声明
std::vector<torch::Tensor> line_inverse_cuda_forward(
    const torch::Tensor accumulator,
    torch::Tensor output,
    const int numangle,
    const int numrho);

std::vector<torch::Tensor> line_inverse_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_accumulator,
    const torch::Tensor output,
    const float* tabCos,
    const float* tabSin,
    const int numangle,
    const int numrho);

// 辅助函数：初始化角度表
void initTab(std::vector<float>& tabSin, std::vector<float>& tabCos, const int numangle, const int numrho, const int H, const int W)
{
    float irho = float(int(std::sqrt(H * H + W * W) + 1)) / float((numrho - 1));
    float itheta = PI / numangle;
    float angle = 0;
    for (int i = 0; i < numangle; ++i)
    {
        tabCos[i] = std::cos(angle) / irho;
        tabSin[i] = std::sin(angle) / irho;
        angle += itheta;
    }
}

// C++ 接口：反 Hough 前向
std::vector<at::Tensor> line_inverse_forward(
    const at::Tensor accumulator,  // 输入的 Hough 累加器，[N, C, numangle, numrho]
    at::Tensor output,             // 输出图像，[N, C, H, W]
    const int numangle,
    const int numrho) {

    CHECK_INPUT(accumulator);
    CHECK_INPUT(output);

    auto out = line_inverse_cuda_forward(accumulator, output, numangle, numrho);
    CHECK_INPUT(out[0]);
    return out;
}

// C++ 接口：反 Hough 反向
std::vector<torch::Tensor> line_inverse_backward(
    torch::Tensor grad_output,         // 输出图像梯度，[N, C, H, W]
    torch::Tensor grad_accumulator,      // 累加器梯度，[N, C, numangle, numrho]
    const torch::Tensor output,        // 前向输出（可选）
    const int numangle,
    const int numrho) {

    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_accumulator);
    CHECK_INPUT(output);

    std::vector<float> tabSin(numangle), tabCos(numangle);
    const int H = output.size(2);
    const int W = output.size(3);
    initTab(tabSin, tabCos, numangle, numrho, H, W);

    return line_inverse_cuda_backward(
        grad_output,
        grad_accumulator,
        output,
        tabCos.data(),
        tabSin.data(),
        numangle,
        numrho);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &line_inverse_forward, "inverse Hough transform forward (CUDA)");
    m.def("backward", &line_inverse_backward, "inverse Hough transform backward (CUDA)");
}
