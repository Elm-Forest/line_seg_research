#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <iostream>

//------------------------------------------------------
// (A) Forward Kernel
//------------------------------------------------------
__global__
void line_accum_forward_kernel(
    const float* __restrict__ feat,   // [N, C, H, W]
    const float* __restrict__ tabCos, // [numangle]
    const float* __restrict__ tabSin, // [numangle]
    float* output,                    // [N, C, numangle, numrho]
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
    int x       = threadIdx.x * threadW;
    int y       = threadIdx.y * threadH;
    int k       = threadIdx.z * threadK;

    int imgStartIdx = batch*channelSize*imWidth*imHeight +
                      channel*imWidth*imHeight +
                      y*imWidth + x;

    int angleStartIdx = k;

    if ((x < imWidth) && (y < imHeight) &&
        (channel < channelSize) && (batch < batchSize) && (k < numangle))
    {
        int imgIndex;
        int angleIndex;
        int outIndex;
        for (int idY = 0; idY < threadH; idY++)
        {
            imgIndex = imgStartIdx + idY * imWidth;
            if (y + idY < imHeight)
            {
                for (int idX = 0; idX < threadW; idX++)
                {
                    if (x + idX < imWidth)
                    {
                        for (int idK = 0; idK < threadK; idK++)
                        {
                            angleIndex = angleStartIdx + idK;
                            if (angleIndex < numangle)
                            {
                                int xx = (x + idX) - imWidth  / 2;
                                int yy = (y + idY) - imHeight / 2;
                                // 计算极坐标 r
                                int rr = __float2int_rn(float(xx) * tabCos[angleIndex] +
                                                        float(yy) * tabSin[angleIndex]);
                                rr += (numrho / 2);

                                outIndex = batch*channelSize*numangle*numrho
                                         + channel*numangle*numrho
                                         + angleIndex*numrho
                                         + rr;
                                float val = feat[imgIndex];
                                atomicAdd(&(output[outIndex]), val);
                            }
                            else
                                break;
                        }
                        imgIndex++;
                    }
                    else
                        break;
                }
            }
            else
                break;
        }
    }
}

//------------------------------------------------------
// (B) Backward Kernel
//------------------------------------------------------
__global__
void line_accum_backward_kernel(
    float* grad_in,             // [N, C, H, W]
    const float* __restrict__ grad_out, // [N, C, numangle, numrho]
    const float* __restrict__ tabCos,
    const float* __restrict__ tabSin,
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
    int x       = threadIdx.x * threadW;
    int y       = threadIdx.y * threadH;
    int k       = threadIdx.z * threadK;

    int imgStartIdx = batch*channelSize*imWidth*imHeight +
                      channel*imWidth*imHeight +
                      y*imWidth + x;

    int angleStartIdx = k;

    if ((x < imWidth) && (y < imHeight) &&
        (channel < channelSize) && (batch < batchSize) && (k < numangle))
    {
        int imgIndex;
        int angleIndex;
        int outIndex;
        for (int idY = 0; idY < threadH; idY++)
        {
            imgIndex = imgStartIdx + idY * imWidth;
            if (y + idY < imHeight)
            {
                for (int idX = 0; idX < threadW; idX++)
                {
                    if (x + idX < imWidth)
                    {
                        for (int idK = 0; idK < threadK; idK++)
                        {
                            angleIndex = angleStartIdx + idK;
                            if (angleIndex < numangle)
                            {
                                int xx = (x + idX) - imWidth  / 2;
                                int yy = (y + idY) - imHeight / 2;
                                int rr = __float2int_rn(float(xx) * tabCos[angleIndex] +
                                                        float(yy) * tabSin[angleIndex]);
                                rr += (numrho / 2);

                                outIndex = batch*channelSize*numangle*numrho
                                         + channel*numangle*numrho
                                         + angleIndex*numrho
                                         + rr;

                                float val = grad_out[outIndex];
                                // 累加回原像素
                                atomicAdd(&(grad_in[imgIndex]), val);
                            }
                            else
                                break;
                        }
                        imgIndex++;
                    }
                    else
                        break;
                }
            }
            else
                break;
        }
    }
}

//------------------------------------------------------
// (C) Inverse Hough Kernel
//------------------------------------------------------
__global__
void line_accum_inverse_kernel(
    const float* __restrict__ hough_in, // [N, C, numangle, numrho]
    float* image_out,                   // [N, C, H, W]
    const float* __restrict__ tabCos,
    const float* __restrict__ tabSin,
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
    int x       = threadIdx.x * threadW;
    int y       = threadIdx.y * threadH;
    int k       = threadIdx.z * threadK;

    int imgStartIdx = batch*channelSize*imWidth*imHeight +
                      channel*imWidth*imHeight +
                      y*imWidth + x;

    if ((x < imWidth) && (y < imHeight) &&
        (channel < channelSize) && (batch < batchSize) && (k < numangle))
    {
        for (int idY = 0; idY < threadH; idY++)
        {
            int imgIndex = imgStartIdx + idY*imWidth;
            if (y + idY < imHeight)
            {
                for (int idX = 0; idX < threadW; idX++)
                {
                    if (x + idX < imWidth)
                    {
                        for (int idK = 0; idK < threadK; idK++)
                        {
                            int angleIndex = k + idK;
                            if (angleIndex < numangle)
                            {
                                int xx = (x + idX) - imWidth  / 2;
                                int yy = (y + idY) - imHeight / 2;
                                int rr = __float2int_rn(float(xx) * tabCos[angleIndex] +
                                                        float(yy) * tabSin[angleIndex]);
                                rr += (numrho / 2);

                                int houghIdx = batch*channelSize*numangle*numrho
                                             + channel*numangle*numrho
                                             + angleIndex*numrho
                                             + rr;

                                float val = hough_in[houghIdx];
                                atomicAdd(&(image_out[imgIndex]), val);
                            }
                            else
                                break;
                        }
                        imgIndex++;
                    }
                    else
                        break;
                }
            }
            else
                break;
        }
    }
}

//------------------------------------------------------
// (D) Forward Wrapper
//------------------------------------------------------
std::vector<torch::Tensor> line_accum_cuda_forward(
    const torch::Tensor feat,
    const float* tabCos,
    const float* tabSin,
    torch::Tensor output,
    const int numangle,
    const int numrho)
{
    const int batch_size   = feat.size(0);
    const int channels     = feat.size(1);
    const int imH          = feat.size(2);
    const int imW          = feat.size(3);

    int blockSizeX = std::min(8, imW);
    const int threadW = static_cast<int>(std::ceil(float(imW) / blockSizeX));

    int blockSizeY = std::min(8, imH);
    const int threadH = static_cast<int>(std::ceil(float(imH) / blockSizeY));

    int blockSizeZ = std::min(8, numangle);
    const int threadK = static_cast<int>(std::ceil(float(numangle) / blockSizeZ));

    const dim3 blocks(channels, batch_size);
    const dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    float* d_tabCos;
    float* d_tabSin;
    cudaMalloc((void**)&d_tabCos, sizeof(float)*numangle);
    cudaMalloc((void**)&d_tabSin, sizeof(float)*numangle);

    cudaMemcpy(d_tabCos, tabCos, sizeof(float)*numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabSin, tabSin, sizeof(float)*numangle, cudaMemcpyHostToDevice);

    line_accum_forward_kernel<<<blocks, threads>>>(
        feat.data_ptr<float>(),
        d_tabCos,
        d_tabSin,
        output.data_ptr<float>(),
        imW,
        imH,
        threadW,
        threadH,
        threadK,
        channels,
        batch_size,
        numangle,
        numrho
    );
    cudaFree(d_tabCos);
    cudaFree(d_tabSin);

    return {output};
}

//------------------------------------------------------
// (E) Backward Wrapper
//------------------------------------------------------
std::vector<torch::Tensor> line_accum_cuda_backward(
    torch::Tensor grad_outputs,
    torch::Tensor grad_in,
    torch::Tensor feat,
    const float* tabCos,
    const float* tabSin,
    const int numangle,
    const int numrho)
{
    const int batch_size = feat.size(0);
    const int channels   = feat.size(1);
    const int imH        = feat.size(2);
    const int imW        = feat.size(3);

    int blockSizeX = std::min(8, imW);
    const int threadW = static_cast<int>(std::ceil(float(imW) / blockSizeX));

    int blockSizeY = std::min(8, imH);
    const int threadH = static_cast<int>(std::ceil(float(imH) / blockSizeY));

    int blockSizeZ = std::min(8, numangle);
    const int threadK = static_cast<int>(std::ceil(float(numangle) / blockSizeZ));

    const dim3 blocks(channels, batch_size);
    const dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    float* d_tabCos;
    float* d_tabSin;
    cudaMalloc((void **)&d_tabCos, sizeof(float)*numangle);
    cudaMalloc((void **)&d_tabSin, sizeof(float)*numangle);

    cudaMemcpy(d_tabCos, tabCos, sizeof(float)*numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabSin, tabSin, sizeof(float)*numangle, cudaMemcpyHostToDevice);

    line_accum_backward_kernel<<<blocks, threads>>>(
        grad_in.data_ptr<float>(),
        grad_outputs.data_ptr<float>(),
        d_tabCos,
        d_tabSin,
        imW,
        imH,
        threadW,
        threadH,
        threadK,
        channels,
        batch_size,
        numangle,
        numrho
    );
    cudaFree(d_tabCos);
    cudaFree(d_tabSin);

    return {grad_in};
}

//------------------------------------------------------
// (F) Inverse Wrapper
//------------------------------------------------------
std::vector<torch::Tensor> line_accum_cuda_inverse(
    const torch::Tensor hough_in,
    torch::Tensor image_out,
    const float* tabCos,
    const float* tabSin,
    const int numangle,
    const int numrho)
{
    const int batch_size = image_out.size(0);
    const int channels   = image_out.size(1);
    const int imH        = image_out.size(2);
    const int imW        = image_out.size(3);

    int blockSizeX = std::min(8, imW);
    const int threadW = static_cast<int>(std::ceil(float(imW) / blockSizeX));

    int blockSizeY = std::min(8, imH);
    const int threadH = static_cast<int>(std::ceil(float(imH) / blockSizeY));

    int blockSizeZ = std::min(8, numangle);
    const int threadK = static_cast<int>(std::ceil(float(numangle) / blockSizeZ));

    const dim3 blocks(channels, batch_size);
    const dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    float* d_tabCos;
    float* d_tabSin;
    cudaMalloc((void**)&d_tabCos, sizeof(float)*numangle);
    cudaMalloc((void**)&d_tabSin, sizeof(float)*numangle);

    cudaMemcpy(d_tabCos, tabCos, sizeof(float)*numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabSin, tabSin, sizeof(float)*numangle, cudaMemcpyHostToDevice);

    line_accum_inverse_kernel<<<blocks, threads>>>(
        hough_in.data_ptr<float>(),
        image_out.data_ptr<float>(),
        d_tabCos,
        d_tabSin,
        imW,
        imH,
        threadW,
        threadH,
        threadK,
        channels,
        batch_size,
        numangle,
        numrho
    );
    cudaFree(d_tabCos);
    cudaFree(d_tabSin);

    return {image_out};
}

//------------------------------------------------------
// (G) DrawFull Kernel
//------------------------------------------------------

__device__
bool compute_line_param_range(
    float x0, float y0,
    float dx, float dy,
    float halfW, float halfH,
    float& tmin, float& tmax)
{
    tmin = -1e10f;
    tmax =  1e10f;

    // X clip
    if (fabsf(dx) > 1e-9f) {
        float t1 = (-halfW - x0) / dx;
        float t2 = ((halfW - 1.f) - x0) / dx;
        float mm1 = fminf(t1, t2);
        float mm2 = fmaxf(t1, t2);
        if (mm1 > tmin) tmin = mm1;
        if (mm2 < tmax) tmax = mm2;
    } else {
        // dx=0 => vertical line
        if (x0 < -halfW || x0 > (halfW -1)) return false;
    }

    // Y clip
    if (fabsf(dy) > 1e-9f) {
        float t1 = (-halfH - y0) / dy;
        float t2 = ((halfH -1.f) - y0) / dy;
        float mm1 = fminf(t1, t2);
        float mm2 = fmaxf(t1, t2);
        if (mm1 > tmin) tmin = mm1;
        if (mm2 < tmax) tmax = mm2;
    } else {
        // dy=0 => horizontal line
        if (y0 < -halfH || y0 > (halfH -1)) return false;
    }

    return (tmin <= tmax);
}

__global__
void line_accum_drawfull_kernel(
    const float* __restrict__ hough_in,   // [N, C, numangle, numrho]
    float* image_out,                     // [N, C, H, W]
    const float* __restrict__ tabCos,     // [numangle]
    const float* __restrict__ tabSin,     // [numangle]
    const int imW, const int imH,
    const float threshold,
    const int numangle, const int numrho,
    const float irho,
    const int totalBC,
    const int batch_size,
    const int channels)
{
    // blockIdx.x * blockDim.x + threadIdx.x => angle a
    // blockIdx.y * blockDim.y + threadIdx.y => rho   r
    // blockIdx.z => bc in [0.. totalBC-1]

    int a = blockIdx.x*blockDim.x + threadIdx.x;
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int bc = blockIdx.z;

    if (bc >= totalBC) return;
    int b = bc / channels;
    int c = bc % channels;

    if (a>=numangle || r>=numrho) return;

    int houghIdx = b*(channels*numangle*numrho)
                 + c*(numangle*numrho)
                 + a*(numrho)
                 + r;

    float val = hough_in[houghIdx];
    if (val <= threshold) return;

    float halfW = float(imW)*0.5f;
    float halfH = float(imH)*0.5f;

    // r_phys
    float r_phys = (r - (numrho/2))* irho;
    float cth = tabCos[a]* irho;
    float sth = tabSin[a]* irho;
    float x0 = cth*r_phys;
    float y0 = sth*r_phys;

    float dx = -sth;
    float dy =  cth;

    float tmin, tmax;
    bool ok = compute_line_param_range(x0, y0, dx, dy, halfW, halfH, tmin, tmax);
    if (!ok) return;

    float x1 = x0 + tmin*dx;
    float y1 = y0 + tmin*dy;
    float x2 = x0 + tmax*dx;
    float y2 = y0 + tmax*dy;

    float seg_len = hypotf(x2 - x1, y2 - y1);
    int steps = (int)(seg_len + 1);

    for (int i=0; i<=steps; i++){
        float alpha = (steps==0)? 0.f: float(i)/float(steps);
        float cx = x1 + alpha*(x2-x1);
        float cy = y1 + alpha*(y2-y1);

        float px = cx + halfW;
        float py = cy + halfH;

        int ix = __float2int_rn(px);
        int iy = __float2int_rn(py);

        if (ix>=0 && ix<imW && iy>=0 && iy<imH){
            int outIdx = b*(channels*imH*imW)
                        + c*(imH*imW)
                        + iy*imW
                        + ix;
            atomicAdd(&image_out[outIdx], val);
        }
    }
}

std::vector<torch::Tensor> line_accum_cuda_drawfull(
    const torch::Tensor hough_in,
    torch::Tensor image_out,
    const float* tabCos,
    const float* tabSin,
    int numangle,
    int numrho,
    float threshold,
    float irho,
    int batch_size,
    int channels)
{
    int H = image_out.size(2);
    int W = image_out.size(3);

    int totalBC = batch_size*channels;

    // 2D block for angle, rho
    dim3 block(16,16);
    dim3 grid(
        (numangle+ block.x-1)/ block.x,
        (numrho  + block.y-1)/ block.y,
        totalBC
    );

    float* d_tabCos;
    float* d_tabSin;
    cudaMalloc((void**)&d_tabCos, sizeof(float)*numangle);
    cudaMalloc((void**)&d_tabSin, sizeof(float)*numangle);

    cudaMemcpy(d_tabCos, tabCos, sizeof(float)*numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabSin, tabSin, sizeof(float)*numangle, cudaMemcpyHostToDevice);

    line_accum_drawfull_kernel<<<grid, block>>>(
        hough_in.data_ptr<float>(),
        image_out.data_ptr<float>(),
        d_tabCos,
        d_tabSin,
        W, H,
        threshold,
        numangle,
        numrho,
        irho,
        totalBC,
        batch_size,
        channels
    );
    cudaFree(d_tabCos);
    cudaFree(d_tabSin);

    return {image_out};
}
