#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>

// 声明在 deep_hough_cuda_kernel.cu 中实现的函数
std::vector<torch::Tensor> line_accum_cuda_forward(
    const torch::Tensor feat,
    const torch::Tensor tabCos_t,
    const torch::Tensor tabSin_t,
    torch::Tensor output,
    const int numangle,
    const int numrho);

std::vector<torch::Tensor> line_accum_cuda_backward(
    torch::Tensor grad_outputs,
    torch::Tensor grad_in,
    const torch::Tensor tabCos_t,
    const torch::Tensor tabSin_t,
    const int numangle,
    const int numrho);


// ---- 常用检查宏
#define CHECK_CUDA(x)      TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)     CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define PI 3.14159265358979323846

// 使用与 feat 相同精度在 CPU 上生成 tabCos, tabSin，并拷到 GPU Tensor
template <typename scalar_t>
void initTabCPU(std::vector<scalar_t> &tabSin,
                std::vector<scalar_t> &tabCos,
                const int numangle,
                const int numrho,
                const int H,
                const int W)
{
    // 计算极坐标分辨率
    float diag = std::sqrt((float)H * H + (float)W * W);
    float irho = float( int(diag + 1.0) ) / float(numrho - 1);
    float itheta = PI / numangle;
    float angle  = 0.0f;
    for (int i = 0; i < numangle; ++i) {
        double c = std::cos((double)angle) / irho;
        double s = std::sin((double)angle) / irho;
        tabCos[i] = (scalar_t)c;
        tabSin[i] = (scalar_t)s;
        angle += itheta;
    }
}

// 前向: feat -> hough
//   feat  : [N, C, H, W] (float/double)
//   output: [N, C, numangle, numrho]
std::vector<at::Tensor> line_accum_forward(
    const at::Tensor feat,       // [N, C, H, W]
    at::Tensor       output,     // [N, C, numangle, numrho]
    const int numangle,
    const int numrho)
{
    CHECK_INPUT(feat);
    CHECK_INPUT(output);

    int H = feat.size(2);
    int W = feat.size(3);

    // 1) 根据 feat 的 dtype（float/double），在 CPU 生成 tabCos, tabSin
    // 2) 拷到 GPU Tensor
    auto opts = feat.options();  // 保持相同 device + dtype

    // 创建 CPU 向量
    if (feat.scalar_type() == at::ScalarType::Float) {
        std::vector<float> tabCos(numangle), tabSin(numangle);
        initTabCPU<float>(tabSin, tabCos, numangle, numrho, H, W);

        // 拷到 GPU
        auto tabCos_t = torch::empty({numangle}, opts);
        auto tabSin_t = torch::empty({numangle}, opts);
        tabCos_t.copy_(torch::from_blob(tabCos.data(), {numangle}, torch::dtype(at::kFloat)));
        tabSin_t.copy_(torch::from_blob(tabSin.data(), {numangle}, torch::dtype(at::kFloat)));

        return line_accum_cuda_forward(
            feat, tabCos_t, tabSin_t, output,
            numangle, numrho
        );
    }
    else if (feat.scalar_type() == at::ScalarType::Double) {
        std::vector<double> tabCos(numangle), tabSin(numangle);
        initTabCPU<double>(tabSin, tabCos, numangle, numrho, H, W);

        // GPU tensor (double)
        auto tabCos_t = torch::empty({numangle}, opts);
        auto tabSin_t = torch::empty({numangle}, opts);
        tabCos_t.copy_(torch::from_blob(tabCos.data(), {numangle}, torch::dtype(at::kDouble)));
        tabSin_t.copy_(torch::from_blob(tabSin.data(), {numangle}, torch::dtype(at::kDouble)));

        return line_accum_cuda_forward(
            feat, tabCos_t, tabSin_t, output,
            numangle, numrho
        );
    }
    else {
        AT_ERROR("line_accum_forward only support float or double!");
    }
}


// 反向: grad_outputs -> grad_in
//   grad_outputs: [N, C, numangle, numrho]
//   grad_in     : [N, C, H, W]
std::vector<torch::Tensor> line_accum_backward(
    torch::Tensor grad_outputs,
    torch::Tensor grad_in,
    torch::Tensor feat,       // 主要用来获取 H, W
    const int numangle,
    const int numrho)
{
    CHECK_INPUT(grad_outputs);
    CHECK_INPUT(grad_in);
    CHECK_INPUT(feat);

    int H = feat.size(2);
    int W = feat.size(3);

    auto opts = feat.options();

    // 根据 dtype 再次生成同样的 tabCos, tabSin
    if (feat.scalar_type() == at::ScalarType::Float) {
        std::vector<float> tabCos(numangle), tabSin(numangle);
        initTabCPU<float>(tabSin, tabCos, numangle, numrho, H, W);

        auto tabCos_t = torch::empty({numangle}, opts);
        auto tabSin_t = torch::empty({numangle}, opts);
        tabCos_t.copy_(torch::from_blob(tabCos.data(), {numangle}, torch::dtype(at::kFloat)));
        tabSin_t.copy_(torch::from_blob(tabSin.data(), {numangle}, torch::dtype(at::kFloat)));

        return line_accum_cuda_backward(
            grad_outputs, grad_in,
            tabCos_t, tabSin_t,
            numangle, numrho
        );
    }
    else if (feat.scalar_type() == at::ScalarType::Double) {
        std::vector<double> tabCos(numangle), tabSin(numangle);
        initTabCPU<double>(tabSin, tabCos, numangle, numrho, H, W);

        auto tabCos_t = torch::empty({numangle}, opts);
        auto tabSin_t = torch::empty({numangle}, opts);
        tabCos_t.copy_(torch::from_blob(tabCos.data(), {numangle}, torch::dtype(at::kDouble)));
        tabSin_t.copy_(torch::from_blob(tabSin.data(), {numangle}, torch::dtype(at::kDouble)));

        return line_accum_cuda_backward(
            grad_outputs, grad_in,
            tabCos_t, tabSin_t,
            numangle, numrho
        );
    }
    else {
        AT_ERROR("line_accum_backward only support float or double!");
    }
}


// ---- PyBind 接口
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &line_accum_forward,  "line features accumulating forward (CUDA) [supports float/double]");
    m.def("backward", &line_accum_backward,"line features accumulating backward (CUDA) [supports float/double]");
}
