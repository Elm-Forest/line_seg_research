#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <math.h>
#include <iostream>

// 声明在 deep_inverse_hough_cuda_kernel.cu 中的接口
std::vector<torch::Tensor> line_inverse_accum_cuda_forward(
    const torch::Tensor hough,
    const torch::Tensor tabCos_t,
    const torch::Tensor tabSin_t,
    torch::Tensor output,
    const int numangle,
    const int numrho);

std::vector<torch::Tensor> line_inverse_accum_cuda_backward(
    torch::Tensor grad_out,
    torch::Tensor grad_in,
    const torch::Tensor tabCos_t,
    const torch::Tensor tabSin_t,
    const int numangle,
    const int numrho);

// 检查宏
#define CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define PI 3.14159265358979323846

// 在 CPU 生成 tabSin, tabCos, 并拷贝到与 hough 相同类型的 GPU Tensor
template <typename scalar_t>
void initTabCPU(std::vector<scalar_t> &tabSin,
                std::vector<scalar_t> &tabCos,
                const int numangle,
                const int numrho,
                const int H,
                const int W)
{
    float diag = std::sqrt((float)H*H + (float)W*W);
    float irho = float(int(diag + 1.0f)) / float(numrho - 1);
    float itheta = PI / numangle;
    float angle  = 0.f;
    for (int i = 0; i < numangle; i++)
    {
        double c = std::cos(angle) / irho;
        double s = std::sin(angle) / irho;
        tabCos[i] = (scalar_t)c;
        tabSin[i] = (scalar_t)s;
        angle += itheta;
    }
}

/**
 * 前向接口（逆 Hough 的正向）：
 *   输入： hough [N, C, numangle, numrho]
 *   输出： output [N, C, H, W]
 *
 *   要根据 output 的 H, W 来生成合适的 tabCos, tabSin。
 */
std::vector<at::Tensor> inverse_accum_forward(
    const at::Tensor hough,   // [N, C, numangle, numrho]
    at::Tensor output,        // [N, C, H, W]
    const int numangle,
    const int numrho)
{
    CHECK_INPUT(hough);
    CHECK_INPUT(output);

    // H, W 从 output 维度获取
    int H = output.size(2);
    int W = output.size(3);

    // 根据 dtype 生成 CPU tabCos, tabSin
    auto opts = hough.options();
    if (hough.scalar_type() == at::ScalarType::Float) {
        std::vector<float> tabSin(numangle), tabCos(numangle);
        initTabCPU<float>(tabSin, tabCos, numangle, numrho, H, W);

        auto tabSin_t = torch::empty({numangle}, opts);
        auto tabCos_t = torch::empty({numangle}, opts);

        tabSin_t.copy_(torch::from_blob(tabSin.data(), {numangle}, torch::dtype(at::kFloat)));
        tabCos_t.copy_(torch::from_blob(tabCos.data(), {numangle}, torch::dtype(at::kFloat)));

        return line_inverse_accum_cuda_forward(
            hough, tabCos_t, tabSin_t, output,
            numangle, numrho
        );
    }
    else if (hough.scalar_type() == at::ScalarType::Double) {
        std::vector<double> tabSin(numangle), tabCos(numangle);
        initTabCPU<double>(tabSin, tabCos, numangle, numrho, H, W);

        auto tabSin_t = torch::empty({numangle}, opts);
        auto tabCos_t = torch::empty({numangle}, opts);

        tabSin_t.copy_(torch::from_blob(tabSin.data(), {numangle}, torch::dtype(at::kDouble)));
        tabCos_t.copy_(torch::from_blob(tabCos.data(), {numangle}, torch::dtype(at::kDouble)));

        return line_inverse_accum_cuda_forward(
            hough, tabCos_t, tabSin_t, output,
            numangle, numrho
        );
    }
    else {
        AT_ERROR("inverse_accum_forward only supports float/double!");
    }
}

/**
 * 反向接口（逆 Hough 的反向）：
 *   输入:  grad_out [N, C, H, W]
 *   输出:  grad_in  [N, C, numangle, numrho]
 */
std::vector<at::Tensor> inverse_accum_backward(
    torch::Tensor grad_out, // [N, C, H, W]
    torch::Tensor grad_in,  // [N, C, numangle, numrho]
    const int numangle,
    const int numrho)
{
    CHECK_INPUT(grad_out);
    CHECK_INPUT(grad_in);

    // 从 grad_out 中获取 H, W
    int H = grad_out.size(2);
    int W = grad_out.size(3);

    auto opts = grad_out.options();
    if (grad_out.scalar_type() == at::ScalarType::Float) {
        std::vector<float> tabSin(numangle), tabCos(numangle);
        initTabCPU<float>(tabSin, tabCos, numangle, numrho, H, W);

        auto tabSin_t = torch::empty({numangle}, opts);
        auto tabCos_t = torch::empty({numangle}, opts);

        tabSin_t.copy_(torch::from_blob(tabSin.data(), {numangle}, torch::dtype(at::kFloat)));
        tabCos_t.copy_(torch::from_blob(tabCos.data(), {numangle}, torch::dtype(at::kFloat)));

        return line_inverse_accum_cuda_backward(
            grad_out, grad_in,
            tabCos_t, tabSin_t,
            numangle, numrho
        );
    }
    else if (grad_out.scalar_type() == at::ScalarType::Double) {
        std::vector<double> tabSin(numangle), tabCos(numangle);
        initTabCPU<double>(tabSin, tabCos, numangle, numrho, H, W);

        auto tabSin_t = torch::empty({numangle}, opts);
        auto tabCos_t = torch::empty({numangle}, opts);

        tabSin_t.copy_(torch::from_blob(tabSin.data(), {numangle}, torch::dtype(at::kDouble)));
        tabCos_t.copy_(torch::from_blob(tabCos.data(), {numangle}, torch::dtype(at::kDouble)));

        return line_inverse_accum_cuda_backward(
            grad_out, grad_in,
            tabCos_t, tabSin_t,
            numangle, numrho
        );
    }
    else {
        AT_ERROR("inverse_accum_backward only supports float/double!");
    }
}


// PyBind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &inverse_accum_forward,
          "Deep Inverse Hough forward (CUDA) [supports float/double]");
    m.def("backward", &inverse_accum_backward,
          "Deep Inverse Hough backward (CUDA) [supports float/double]");
}
