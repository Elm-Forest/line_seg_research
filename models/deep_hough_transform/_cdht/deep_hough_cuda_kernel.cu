#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>

template<typename scalar_t>
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
    int batch = blockIdx.y;
    int channel = blockIdx.x;
    int x = threadIdx.x * threadW;
    int y = threadIdx.y * threadH;
    int k = threadIdx.z * threadK;

    int imgStartIdx = batch*channelSize*imWidth*imHeight +
                      channel*imWidth*imHeight +
                      y*imWidth + x;

    int angleStartIdx = k;

    if (x < imWidth && y < imHeight && channel < channelSize && batch < batchSize && k < numangle)
    {
        int imgIndex = imgStartIdx;
        for (int idY = 0; idY < threadH; ++idY)
        {
            imgIndex = imgStartIdx + idY * imWidth;
            if (y + idY < imHeight)
            {
                for (int idX = 0; idX < threadW; ++idX)
                {
                    if (x + idX < imWidth)
                    {
                        for (int idK = 0; idK < threadK; ++idK)
                        {
                            int angleIndex = angleStartIdx + idK;
                            if (angleIndex < numangle)
                            {
                                int xx = x + idX - imWidth / 2;
                                int yy = y + idY - imHeight / 2;
                                int r = static_cast<int>(roundf(
                                    static_cast<float>(xx) * static_cast<float>(tabCos[angleIndex]) +
                                    static_cast<float>(yy) * static_cast<float>(tabSin[angleIndex])
                                ));
                                r += numrho / 2;
                                int outIndex = batch * channelSize * numangle * numrho +
                                               channel * numangle * numrho +
                                               angleIndex * numrho + r;
                                scalar_t val = feat[imgIndex];
                                atomicAdd(&output[outIndex], val);  // 注意 atomicAdd 只支持 float/double，编译器自动分派
                            }
                        }
                        imgIndex++;
                    }
                }
            }
        }
    }
}

template<typename scalar_t>
__global__
void line_accum_backward_kernel(
    scalar_t* grad_in,
    const scalar_t* grad_out,
    const scalar_t* tabCos,
    const scalar_t* tabSin,
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
    int batch = blockIdx.y;
    int channel = blockIdx.x;
    int x = threadIdx.x * threadW;
    int y = threadIdx.y * threadH;
    int k = threadIdx.z * threadK;

    int imgStartIdx = batch*channelSize*imWidth*imHeight +
                      channel*imWidth*imHeight +
                      y*imWidth + x;

    int angleStartIdx = k;

    if (x < imWidth && y < imHeight && channel < channelSize && batch < batchSize && k < numangle)
    {
        int imgIndex = imgStartIdx;
        for (int idY = 0; idY < threadH; ++idY)
        {
            imgIndex = imgStartIdx + idY * imWidth;
            if (y + idY < imHeight)
            {
                for (int idX = 0; idX < threadW; ++idX)
                {
                    if (x + idX < imWidth)
                    {
                        for (int idK = 0; idK < threadK; ++idK)
                        {
                            int angleIndex = angleStartIdx + idK;
                            if (angleIndex < numangle)
                            {
                                int xx = x + idX - imWidth / 2;
                                int yy = y + idY - imHeight / 2;
                                int r = static_cast<int>(roundf(
                                    static_cast<float>(xx) * static_cast<float>(tabCos[angleIndex]) +
                                    static_cast<float>(yy) * static_cast<float>(tabSin[angleIndex])
                                ));
                                r += numrho / 2;
                                int outIndex = batch * channelSize * numangle * numrho +
                                               channel * numangle * numrho +
                                               angleIndex * numrho + r;
                                scalar_t val = grad_out[outIndex];
                                atomicAdd(&grad_in[imgIndex], val);
                            }
                        }
                        imgIndex++;
                    }
                }
            }
        }
    }
}

// ----------
// WRAPPERS
// ----------

std::vector<torch::Tensor> line_accum_cuda_forward(
    const torch::Tensor feat,
    const float* tabCos,
    const float* tabSin,
    torch::Tensor output,
    const int numangle,
    const int numrho)
{
    const int B = feat.size(0);
    const int C = feat.size(1);
    const int H = feat.size(2);
    const int W = feat.size(3);

    int blockSizeX = std::min(8, W);
    int threadW = (W + blockSizeX - 1) / blockSizeX;

    int blockSizeY = std::min(8, H);
    int threadH = (H + blockSizeY - 1) / blockSizeY;

    int blockSizeZ = std::min(8, numangle);
    int threadK = (numangle + blockSizeZ - 1) / blockSizeZ;

    dim3 blocks(C, B);
    dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    float *d_tabCos, *d_tabSin;
    cudaMalloc(&d_tabCos, sizeof(float) * numangle);
    cudaMalloc(&d_tabSin, sizeof(float) * numangle);
    cudaMemcpy(d_tabCos, tabCos, sizeof(float) * numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabSin, tabSin, sizeof(float) * numangle, cudaMemcpyHostToDevice);

    AT_DISPATCH_FLOATING_TYPES(feat.scalar_type(), "line_accum_forward_kernel", ([&] {
        line_accum_forward_kernel<scalar_t><<<blocks, threads>>>(
            feat.data_ptr<scalar_t>(),
            reinterpret_cast<scalar_t*>(d_tabCos),
            reinterpret_cast<scalar_t*>(d_tabSin),
            output.data_ptr<scalar_t>(),
            W, H, threadW, threadH, threadK,
            C, B, numangle, numrho
        );
    }));

    cudaFree(d_tabCos);
    cudaFree(d_tabSin);
    return {output};
}

std::vector<torch::Tensor> line_accum_cuda_backward(
    torch::Tensor grad_outputs,
    torch::Tensor grad_in,
    torch::Tensor feat,
    const float* tabCos,
    const float* tabSin,
    const int numangle,
    const int numrho)
{
    const int B = feat.size(0);
    const int C = feat.size(1);
    const int H = feat.size(2);
    const int W = feat.size(3);

    int blockSizeX = std::min(8, W);
    int threadW = (W + blockSizeX - 1) / blockSizeX;

    int blockSizeY = std::min(8, H);
    int threadH = (H + blockSizeY - 1) / blockSizeY;

    int blockSizeZ = std::min(8, numangle);
    int threadK = (numangle + blockSizeZ - 1) / blockSizeZ;

    dim3 blocks(C, B);
    dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    float *d_tabCos, *d_tabSin;
    cudaMalloc(&d_tabCos, sizeof(float) * numangle);
    cudaMalloc(&d_tabSin, sizeof(float) * numangle);
    cudaMemcpy(d_tabCos, tabCos, sizeof(float) * numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabSin, tabSin, sizeof(float) * numangle, cudaMemcpyHostToDevice);

    AT_DISPATCH_FLOATING_TYPES(grad_outputs.scalar_type(), "line_accum_backward_kernel", ([&] {
        line_accum_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_in.data_ptr<scalar_t>(),
            grad_outputs.data_ptr<scalar_t>(),
            reinterpret_cast<scalar_t*>(d_tabCos),
            reinterpret_cast<scalar_t*>(d_tabSin),
            W, H, threadW, threadH, threadK,
            C, B, numangle, numrho
        );
    }));

    cudaFree(d_tabCos);
    cudaFree(d_tabSin);
    return {grad_in};
}
