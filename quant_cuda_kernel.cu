#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "src/fastertransformer/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "src/fastertransformer/utils/activation_types.h"

#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm.h"

using T_A = half;
using T_B = cutlass::uint4b_t;
using T_C = T_A;

using namespace fastertransformer;

void run(T_A* d_A, T_B* d_B, T_C* d_C, T_C* d_scales, T_C* d_biases, int m, int n, int k) {
    auto* runner = new CutlassFpAIntBGemmRunner<T_A, T_B>();

    char* workspace_ptr;
    size_t workspace_bytes = runner->getWorkspaceSize(m, n, k);
    cudaMalloc((char**) &workspace_ptr, workspace_bytes);

    runner->gemm_bias_act(d_A, d_B, d_scales, d_biases, d_C, m, n, k, ActivationType::Identity, workspace_ptr, workspace_bytes, 0);

    cudaDeviceSynchronize();

    cudaFree(workspace_ptr);

    delete runner;
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int height,
    int width
);

__global__ void VecQuant4MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros,
    int height,
    int width
);

__global__ void BatchVecQuant4MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros,
    int height,
    int width,
    int batchsize
);

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT =  32;

void vecquant4matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_cuda", ([&] {
      VecQuant4MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<scalar_t>(),
        height, width
      );
    })
  );
}

void vecquant4matmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batchsize = (vec.dim() == 1) ? 1 : vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);

  int m = batchsize;
  int k = height * 8;
  int n = width;

  run((half*) vec.data_ptr(), (cutlass::uint4b_t*) mat.data_ptr(), (half*) mul.data_ptr(), (half*) scales.data_ptr(), 
    (half*) mul.data_ptr(), m, k, n);
  
}

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int height,
    int width
) {
  int row = BLOCKHEIGHT * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t scale = scales[col];
  scalar_t zero = zeros[col];

  scalar_t res = 0;
  int i = width * row + col;
  int k = 0;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
    res += (scale * scalar_t((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp2 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp2 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp2 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp2 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp2 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp2 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp2 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp2 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    k += 10;
  }

  atomicAdd(&mul[col], res);
}

__global__ void VecQuant4MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros,
    int height,
    int width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[64][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0x7), __int2half_rn(val >> 3)
    );
  }

  half2 scale = __float2half2_rn(scales[col]);
  half2 zero = __float2half2_rn(-zeros[col]);

  int i = width * row + col;
  int k = 0;

  float res = 0;
  half2 res2;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    res2 = {};
    tmp1 = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
    res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 5], res2);
    tmp2 >>= 4;
    k += 6;
    res2 = __hfma2(__hfma2(deq2[(tmp2 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
    res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 4], res2);
    tmp1 >>= 2;
    k += 5;
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
    i += width;
    k += 5;
    res += __half2float(res2.x) + __half2float(res2.y);
  }

  atomicAdd(&mul[col], res);
}

__global__ void BatchVecQuant4MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros,
    int height,
    int width,
    int batchsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  half scale = __float2half_rn(scales[col]);
  half zero = __float2half_rn(-zeros[col]);

  __shared__ half2 blockvec[blockwidth2];

  for (int b = 0; b < batchsize; b++) {
    if (threadIdx.x < blockwidth2)
      blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * blockwidth2 + threadIdx.x + b * height / 4 * 16];
    int i = width * row + col;
    int k = 0;

    float res = 0;

    __syncthreads();

    unsigned int tmp;

    while (k < blockwidth2) {
      tmp = as_unsigned(mat[i]);
      half t = {};

      t = __hfma(__hfma(__int2half_rn((tmp >> 0) & 0xf), scale, zero), blockvec[k + 0].x, t);
      t = __hfma(__hfma(__int2half_rn((tmp >> 4) & 0xf), scale, zero), blockvec[k + 0].y, t);
      t = __hfma(__hfma(__int2half_rn((tmp >> 8) & 0xf), scale, zero), blockvec[k + 1].x, t);
      t = __hfma(__hfma(__int2half_rn((tmp >> 12) & 0xf), scale, zero), blockvec[k + 1].y, t);
      t = __hfma(__hfma(__int2half_rn((tmp >> 16) & 0xf), scale, zero), blockvec[k + 2].x, t);
      t = __hfma(__hfma(__int2half_rn((tmp >> 20) & 0xf), scale, zero), blockvec[k + 2].y, t);
      t = __hfma(__hfma(__int2half_rn((tmp >> 24) & 0xf), scale, zero), blockvec[k + 3].x, t);
      t = __hfma(__hfma(__int2half_rn((tmp >> 28) & 0xf), scale, zero), blockvec[k + 3].y, t);

      res += __half2float(t);

      i += width;
      k += 4;
    }

    atomicAdd(&mul[col + b * width], res);
  }
}
