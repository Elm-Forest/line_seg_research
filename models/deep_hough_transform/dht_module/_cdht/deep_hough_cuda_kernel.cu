#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <iostream>

// ---------------------------------------
// 原子加法包装（若需要对 double 做特殊处理，可自行添加判断）
template <typename scalar_t>
__device__ __forceinline__ void atomicAddWrapper(scalar_t* address, scalar_t val) {
    // 对 float / double 皆可用（需 GPU 计算能力 >= 6.0 才支持原生 double atomicAdd）
    atomicAdd(address, val);
}

// -------------
// Forward Kernel
// -------------
// feat    : [N, C, H, W]
// tabCos  : [numangle]
// tabSin  : [numangle]
// output  : [N, C, numangle, numrho]
template <typename scalar_t>
__global__
void line_accum_forward_kernel(
    const scalar_t* __restrict__ feat,
    const scalar_t* __restrict__ tabCos,
    const scalar_t* __restrict__ tabSin,
    scalar_t* output,
    const int imWidth,
    const int imHeight,
    const int threadW,
    const int threadH,
    const int threadK,
    const int channelSize,
    const int batchSize,
    const int numangle,
    const int numrho)
{
    // blockIdx.x -> channel, blockIdx.y -> batch
    // threadIdx.x, threadIdx.y, threadIdx.z -> 对应 x, y, angle 的子范围
    int batch   = blockIdx.y;
    int channel = blockIdx.x;

    int x = threadIdx.x * threadW;
    int y = threadIdx.y * threadH;
    int k = threadIdx.z * threadK;

    // 计算 feat 中 (batch, channel, y, x) 对应的基地址
    // feat 是 [N, C, H, W]
    int imgStartIdx = batch * channelSize * imWidth * imHeight
                    + channel * imWidth * imHeight
                    + y * imWidth
                    + x;

    // 角度起始
    int angleStartIdx = k;

    // 边界检查
    if (x >= imWidth || y >= imHeight || channel >= channelSize || batch >= batchSize || k >= numangle) {
        return;
    }

    for (int idY = 0; idY < threadH; idY++) {
        int yy = y + idY;
        if (yy >= imHeight) break;
        // 当前行 feat 索引
        int imgIndex = imgStartIdx + idY * imWidth;

        for (int idX = 0; idX < threadW; idX++) {
            int xx = x + idX;
            if (xx >= imWidth) break;

            // 取该像素的值
            scalar_t pixelVal = feat[imgIndex + idX];

            // 对每个角度累加
            for (int idK = 0; idK < threadK; idK++) {
                int angleIndex = angleStartIdx + idK;
                if (angleIndex >= numangle) break;

                // 计算在 Hough 空间中的 rho
                int cx = xx - imWidth  / 2;
                int cy = yy - imHeight / 2;
                scalar_t rF = scalar_t(cx) * tabCos[angleIndex] + scalar_t(cy) * tabSin[angleIndex];
                int r = (int)round((double)rF);  // cast to double 做 round，避免误差

                r += (numrho / 2);
                // 如果超出范围则跳过
                if (r < 0 || r >= numrho) continue;

                // output 下标: [N, C, angle, rho]
                int outIndex = batch * channelSize * numangle * numrho
                             + channel * numangle * numrho
                             + angleIndex * numrho
                             + r;

                // 原子加
                atomicAddWrapper(&output[outIndex], pixelVal);
            }
        }
    }
}

// -------------
// Backward Kernel
// -------------
// grad_in  : [N, C, H, W]
// grad_out : [N, C, numangle, numrho]
// 其中 grad_out 是对 output 的梯度，向 grad_in 累加
template <typename scalar_t>
__global__
void line_accum_backward_kernel(
    scalar_t* __restrict__ grad_in,
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ tabCos,
    const scalar_t* __restrict__ tabSin,
    const int imWidth,
    const int imHeight,
    const int threadW,
    const int threadH,
    const int threadK,
    const int channelSize,
    const int batchSize,
    const int numangle,
    const int numrho)
{
    int batch   = blockIdx.y;
    int channel = blockIdx.x;

    int x = threadIdx.x * threadW;
    int y = threadIdx.y * threadH;
    int k = threadIdx.z * threadK;

    // grad_in : [N, C, H, W]
    int imgStartIdx = batch * channelSize * imWidth * imHeight
                    + channel * imWidth * imHeight
                    + y * imWidth
                    + x;

    // 边界检查
    if (x >= imWidth || y >= imHeight || channel >= channelSize || batch >= batchSize || k >= numangle) {
        return;
    }

    for (int idY = 0; idY < threadH; idY++) {
        int yy = y + idY;
        if (yy >= imHeight) break;
        int imgIndex = imgStartIdx + idY * imWidth;

        for (int idX = 0; idX < threadW; idX++) {
            int xx = x + idX;
            if (xx >= imWidth) break;

            // 针对每个角度
            for (int idK = 0; idK < threadK; idK++) {
                int angleIdx = k + idK;
                if (angleIdx >= numangle) break;

                int cx = xx - imWidth / 2;
                int cy = yy - imHeight / 2;
                scalar_t rF = scalar_t(cx) * tabCos[angleIdx] + scalar_t(cy) * tabSin[angleIdx];
                int r = (int)round((double)rF);

                r += (numrho / 2);
                if (r < 0 || r >= numrho) continue;

                // [N, C, angle, rho]
                int outIndex = batch * channelSize * numangle * numrho
                             + channel * numangle * numrho
                             + angleIdx * numrho
                             + r;

                // 取该处的梯度
                scalar_t gVal = grad_out[outIndex];

                // 累加回 grad_in
                atomicAddWrapper(&grad_in[imgIndex + idX], gVal);
            }
        }
    }
}


// --------------------------------------------------------
// 前向包装: 调用 line_accum_forward_kernel<scalar_t>
// --------------------------------------------------------
std::vector<torch::Tensor> line_accum_cuda_forward(
    const torch::Tensor feat,
    const torch::Tensor tabCos_t,  // GPU tensor [numangle]
    const torch::Tensor tabSin_t,  // GPU tensor [numangle]
    torch::Tensor output,          // [N, C, numangle, numrho]
    const int numangle,
    const int numrho)
{
    const auto batch_size   = feat.size(0);
    const auto channel_size = feat.size(1);
    const auto imH = feat.size(2);
    const auto imW = feat.size(3);

    // 设定 blocks / threads 大小(可调)
    int blockSizeX = std::min(8, (int)imW);
    int blockSizeY = std::min(8, (int)imH);
    int blockSizeZ = std::min(8, (int)numangle);

    // 每个线程处理多少像素/角度
    const int threadW = ceil((double)imW      / blockSizeX);
    const int threadH = ceil((double)imH      / blockSizeY);
    const int threadK = ceil((double)numangle / blockSizeZ);

    dim3 blocks(channel_size, batch_size);
    dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    // 根据 feat 的实际类型进行分发
    AT_DISPATCH_FLOATING_TYPES(feat.scalar_type(), "line_accum_cuda_forward", ([&] {
        const scalar_t* feat_ptr    = feat.data_ptr<scalar_t>();
        const scalar_t* cos_ptr     = tabCos_t.data_ptr<scalar_t>();
        const scalar_t* sin_ptr     = tabSin_t.data_ptr<scalar_t>();
        scalar_t* output_ptr        = output.data_ptr<scalar_t>();

        line_accum_forward_kernel<scalar_t><<<blocks, threads>>>(
            feat_ptr,
            cos_ptr,
            sin_ptr,
            output_ptr,
            imW,
            imH,
            threadW,
            threadH,
            threadK,
            channel_size,
            batch_size,
            numangle,
            numrho
        );
        cudaDeviceSynchronize();
    }));

    return { output };
}


// --------------------------------------------------------
// 反向包装: 调用 line_accum_backward_kernel<scalar_t>
// --------------------------------------------------------
std::vector<torch::Tensor> line_accum_cuda_backward(
    torch::Tensor grad_outputs,  // [N, C, numangle, numrho]
    torch::Tensor grad_in,       // [N, C, H, W], accum
    const torch::Tensor tabCos_t,
    const torch::Tensor tabSin_t,
    const int numangle,
    const int numrho)
{
    const auto batch_size   = grad_in.size(0);
    const auto channel_size = grad_in.size(1);
    const auto imH = grad_in.size(2);
    const auto imW = grad_in.size(3);

    int blockSizeX = std::min(8, (int)imW);
    int blockSizeY = std::min(8, (int)imH);
    int blockSizeZ = std::min(8, (int)numangle);

    const int threadW = ceil((double)imW      / blockSizeX);
    const int threadH = ceil((double)imH      / blockSizeY);
    const int threadK = ceil((double)numangle / blockSizeZ);

    dim3 blocks(channel_size, batch_size);
    dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    AT_DISPATCH_FLOATING_TYPES(grad_in.scalar_type(), "line_accum_cuda_backward", ([&] {
        scalar_t* grad_in_ptr         = grad_in.data_ptr<scalar_t>();
        const scalar_t* grad_out_ptr  = grad_outputs.data_ptr<scalar_t>();
        const scalar_t* cos_ptr       = tabCos_t.data_ptr<scalar_t>();
        const scalar_t* sin_ptr       = tabSin_t.data_ptr<scalar_t>();

        line_accum_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_in_ptr,
            grad_out_ptr,
            cos_ptr,
            sin_ptr,
            imW,
            imH,
            threadW,
            threadH,
            threadK,
            channel_size,
            batch_size,
            numangle,
            numrho
        );
        cudaDeviceSynchronize();
    }));

    return { grad_in };
}
