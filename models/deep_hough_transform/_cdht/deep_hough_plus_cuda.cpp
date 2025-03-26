#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <iostream>

// ============ CUDA 声明 =============
std::vector<torch::Tensor> line_accum_cuda_forward(
    const torch::Tensor feat,
    const float* tabCos,
    const float* tabSin,
    torch::Tensor output,
    const int numangle,
    const int numrho);

std::vector<torch::Tensor> line_accum_cuda_backward(
    torch::Tensor grad_outputs,
    torch::Tensor grad_in,
    torch::Tensor feat,
    const float* tabCos,
    const float* tabSin,
    const int numangle,
    const int numrho);

std::vector<torch::Tensor> line_accum_cuda_inverse(
    const torch::Tensor hough_in,
    torch::Tensor image_out,
    const float* tabCos,
    const float* tabSin,
    const int numangle,
    const int numrho);

std::vector<torch::Tensor> line_accum_cuda_drawfull(
    const torch::Tensor hough_in,
    torch::Tensor image_out,
    const float* tabCos,
    const float* tabSin,
    int   numangle,
    int   numrho,
    float threshold,
    float irho,
    int   batch_size,
    int   channels
);

// ============= 常用宏 ==============
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ============= Tab Init =============
static void initTab(std::vector<float>& tabSin,
                    std::vector<float>& tabCos,
                    int numangle, int numrho,
                    int H, int W)
{
    float diag = std::sqrt(H*H + W*W) + 1;
    float irho = diag / float(numrho -1);
    float itheta = M_PI / float(numangle);

    float angle = 0.0f;
    for(int i=0; i< numangle; i++){
        tabCos[i] = std::cos(angle)/ irho;
        tabSin[i] = std::sin(angle)/ irho;
        angle += itheta;
    }
}

// ============= forward接口 =============
std::vector<at::Tensor> line_accum_forward(
    const at::Tensor feat,
    at::Tensor output,
    const int numangle,
    const int numrho)
{
    CHECK_INPUT(feat);
    CHECK_INPUT(output);

    int H = feat.size(2);
    int W = feat.size(3);
    std::vector<float> tabSin(numangle), tabCos(numangle);
    initTab(tabSin, tabCos, numangle, numrho, H, W);

    auto out = line_accum_cuda_forward(
        feat, tabCos.data(), tabSin.data(),
        output, numangle, numrho
    );
    return out;
}

// ============= backward接口 =============
std::vector<torch::Tensor> line_accum_backward(
    torch::Tensor grad_outputs,
    torch::Tensor grad_in,
    torch::Tensor feat,
    const int numangle,
    const int numrho)
{
    CHECK_INPUT(grad_outputs);
    CHECK_INPUT(grad_in);
    CHECK_INPUT(feat);

    int H = feat.size(2);
    int W = feat.size(3);
    std::vector<float> tabSin(numangle), tabCos(numangle);
    initTab(tabSin, tabCos, numangle, numrho, H, W);

    return line_accum_cuda_backward(
        grad_outputs, grad_in, feat,
        tabCos.data(), tabSin.data(),
        numangle, numrho
    );
}

// ============= inverse接口 =============
std::vector<torch::Tensor> line_accum_inverse(
    const torch::Tensor hough_in,
    torch::Tensor image_out,
    const int numangle,
    const int numrho)
{
    CHECK_INPUT(hough_in);
    CHECK_INPUT(image_out);

    int H = image_out.size(2);
    int W = image_out.size(3);
    std::vector<float> tabSin(numangle), tabCos(numangle);
    initTab(tabSin, tabCos, numangle, numrho, H, W);

    return line_accum_cuda_inverse(
        hough_in, image_out,
        tabCos.data(), tabSin.data(),
        numangle, numrho
    );
}

// ============= drawfull接口 =============
std::vector<torch::Tensor> line_accum_drawfull(
    const torch::Tensor hough_in,
    torch::Tensor image_out,
    float threshold,
    const int numangle,
    const int numrho)
{
    CHECK_INPUT(hough_in);
    CHECK_INPUT(image_out);

    int H = image_out.size(2);
    int W = image_out.size(3);
    int N = hough_in.size(0);
    int C = hough_in.size(1);

    // 生成 tab
    std::vector<float> tabSin(numangle), tabCos(numangle);
    initTab(tabSin, tabCos, numangle, numrho, H, W);

    // 计算 irho
    float diag = std::sqrt(H*H + W*W) + 1;
    float irho = diag / float(numrho -1);

    return line_accum_cuda_drawfull(
        hough_in, image_out,
        tabCos.data(), tabSin.data(),
        numangle, numrho,
        threshold, irho,
        N, C
    );
}

// ============= PYBIND11 模块 =============
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",   &line_accum_forward,   "hough forward (CUDA)");
    m.def("backward",  &line_accum_backward,  "hough backward (CUDA)");
    m.def("inverse",   &line_accum_inverse,   "hough inverse (CUDA)");
    m.def("drawfull",  &line_accum_drawfull,  "hough drawfull (CUDA)");
}
