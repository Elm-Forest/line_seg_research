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
// 原子加法包装：对 float/double 通用
// （需要 GPU 计算能力 >= 6.0 才能原生支持 atomicAdd(double* , double) ）
// ---------------------------------------
template <typename scalar_t>
__device__ __forceinline__ void atomicAddWrapper(scalar_t* address, scalar_t val) {
    atomicAdd(address, val);
}

/**
 * Inverse Hough 正向： Hough -> Image
 *   hough  : [N, C, numangle, numrho]
 *   tabCos : [numangle]
 *   tabSin : [numangle]
 *   output : [N, C, H, W]  (被累加)
 *
 *  逻辑：对于 (angle, rho) 所对应的像素 (x,y)，将 hough 里的值加到 output(x,y) 上
 *        这里实现的方式是遍历 (x, y, angle) 并查找对应 rho 后加到 output。
 */
template <typename scalar_t>
__global__
void line_inverse_accum_forward_kernel(
    const scalar_t* __restrict__ hough,
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
    // blockIdx.x = channel, blockIdx.y = batch
    int batch   = blockIdx.y;
    int channel = blockIdx.x;

    // threadIdx.x, threadIdx.y, threadIdx.z -> x, y, angle 子范围
    int x = threadIdx.x * threadW;
    int y = threadIdx.y * threadH;
    int k = threadIdx.z * threadK;

    // 在 output (N, C, H, W) 中的基准
    int outBaseIdx = batch * channelSize * imWidth * imHeight
                   + channel * imWidth * imHeight
                   + y * imWidth
                   + x;

    // 边界检查
    if (x >= imWidth || y >= imHeight ||
        channel >= channelSize || batch >= batchSize ||
        k >= numangle)
    {
        return;
    }

    for (int idY = 0; idY < threadH; idY++) {
        int curY = y + idY;
        if (curY >= imHeight) break;
        int outYbase = outBaseIdx + idY * imWidth;

        for (int idX = 0; idX < threadW; idX++) {
            int curX = x + idX;
            if (curX >= imWidth) break;

            int outIdx = outYbase + idX;

            // 对该 (x, y) 遍历子范围里的 angle
            for (int idK = 0; idK < threadK; idK++) {
                int angleIdx = k + idK;
                if (angleIdx >= numangle) break;

                int xx = curX - imWidth  / 2;
                int yy = curY - imHeight / 2;
                // 计算 rho 索引
                scalar_t rF = scalar_t(xx) * tabCos[angleIdx] + scalar_t(yy) * tabSin[angleIdx];
                int r = (int)round((double)rF);
                r += (numrho / 2);

                if (r < 0 || r >= numrho) {
                    continue;
                }

                // hough 下标: [N, C, angleIdx, r]
                int houghIndex = batch * channelSize * numangle * numrho
                               + channel * numangle * numrho
                               + angleIdx * numrho
                               + r;

                // 将 hough[houghIndex] 累加到 output[outIdx]
                scalar_t val = hough[houghIndex];
                atomicAddWrapper(&output[outIdx], val);
            }
        }
    }
}

/**
 * Inverse Hough 反向：从 output 的梯度 -> hough 的梯度
 *   grad_in  : [N, C, numangle, numrho]
 *   grad_out : [N, C, H, W]
 */
template <typename scalar_t>
__global__
void line_inverse_accum_backward_kernel(
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

    if (x >= imWidth || y >= imHeight ||
        channel >= channelSize || batch >= batchSize ||
        k >= numangle)
    {
        return;
    }

    // grad_out: [N, C, H, W]
    int outBaseIdx = batch * channelSize * imWidth * imHeight
                   + channel * imWidth * imHeight
                   + y * imWidth
                   + x;

    for (int idY = 0; idY < threadH; idY++) {
        int curY = y + idY;
        if (curY >= imHeight) break;
        int outYbase = outBaseIdx + idY * imWidth;

        for (int idX = 0; idX < threadW; idX++) {
            int curX = x + idX;
            if (curX >= imWidth) break;

            int outIdx = outYbase + idX;
            // 取当前像素梯度
            scalar_t gradVal = grad_out[outIdx];

            for (int idK = 0; idK < threadK; idK++) {
                int angleIdx = k + idK;
                if (angleIdx >= numangle) break;

                int xx = curX - imWidth  / 2;
                int yy = curY - imHeight / 2;

                scalar_t rF = scalar_t(xx) * tabCos[angleIdx] + scalar_t(yy) * tabSin[angleIdx];
                int r = (int)round((double)rF);
                r += (numrho / 2);

                if (r < 0 || r >= numrho) {
                    continue;
                }

                // 对应 grad_in [N, C, angle, r]
                int houghIndex = batch * channelSize * numangle * numrho
                               + channel * numangle * numrho
                               + angleIdx * numrho
                               + r;

                atomicAddWrapper(&grad_in[houghIndex], gradVal);
            }
        }
    }
}


// -------------------------------
// 前向包装函数: 调用 kernel (Hough->Image)
// -------------------------------
std::vector<torch::Tensor> line_inverse_accum_cuda_forward(
    const torch::Tensor hough,   // [N, C, numangle, numrho]
    const torch::Tensor tabCos_t,// [numangle]
    const torch::Tensor tabSin_t,// [numangle]
    torch::Tensor output,        // [N, C, H, W], 被累加
    const int numangle,
    const int numrho)
{
    const int batch_size   = hough.size(0);
    const int channel_size = hough.size(1);
    // output: [N, C, H, W]
    const int imH = output.size(2);
    const int imW = output.size(3);

    // 设定 block / thread 大小
    int blockSizeX = std::min(8, imW);
    int blockSizeY = std::min(8, imH);
    int blockSizeZ = std::min(8, numangle);

    const int threadW = (int)ceil((double)imW / blockSizeX);
    const int threadH = (int)ceil((double)imH / blockSizeY);
    const int threadK = (int)ceil((double)numangle / blockSizeZ);

    dim3 blocks(channel_size, batch_size);
    dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    AT_DISPATCH_FLOATING_TYPES(hough.scalar_type(), "line_inverse_accum_cuda_forward", ([&] {
        const scalar_t* hough_ptr = hough.data_ptr<scalar_t>();
        const scalar_t* cos_ptr   = tabCos_t.data_ptr<scalar_t>();
        const scalar_t* sin_ptr   = tabSin_t.data_ptr<scalar_t>();
        scalar_t* out_ptr         = output.data_ptr<scalar_t>();

        line_inverse_accum_forward_kernel<scalar_t><<<blocks, threads>>>(
            hough_ptr,
            cos_ptr,
            sin_ptr,
            out_ptr,
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

    return {output};
}

// -------------------------------
// 反向包装函数: (dOutput->dHough)
// -------------------------------
std::vector<torch::Tensor> line_inverse_accum_cuda_backward(
    torch::Tensor grad_out,     // [N, C, H, W]
    torch::Tensor grad_in,      // [N, C, numangle, numrho]
    const torch::Tensor tabCos_t,
    const torch::Tensor tabSin_t,
    const int numangle,
    const int numrho)
{
    const int batch_size   = grad_in.size(0);
    const int channel_size = grad_in.size(1);
    // grad_out: [N, C, H, W]
    // grad_in : [N, C, numangle, numrho]
    // 需要 H, W 来进行坐标计算
    const int imH = grad_out.size(2);
    const int imW = grad_out.size(3);

    int blockSizeX = std::min(8, imW);
    int blockSizeY = std::min(8, imH);
    int blockSizeZ = std::min(8, numangle);

    const int threadW = (int)ceil((double)imW / blockSizeX);
    const int threadH = (int)ceil((double)imH / blockSizeY);
    const int threadK = (int)ceil((double)numangle / blockSizeZ);

    dim3 blocks(channel_size, batch_size);
    dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    AT_DISPATCH_FLOATING_TYPES(grad_in.scalar_type(), "line_inverse_accum_cuda_backward", ([&] {
        scalar_t* grad_in_ptr     = grad_in.data_ptr<scalar_t>();
        const scalar_t* grad_o_ptr= grad_out.data_ptr<scalar_t>();
        const scalar_t* cos_ptr   = tabCos_t.data_ptr<scalar_t>();
        const scalar_t* sin_ptr   = tabSin_t.data_ptr<scalar_t>();

        line_inverse_accum_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_in_ptr,
            grad_o_ptr,
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

    return {grad_in};
}
