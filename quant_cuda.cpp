#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void vecquant8matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant8matmul_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
); 

void vecquant8matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_cuda(vec, mat, mul, scales, zeros);
}

void vecquant8matmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_faster_cuda(vec, mat, mul, scales, zeros);
}

void preprocess_weights_for_mixed_gemm(
  torch::Tensor preprocessed_quantized_weight, torch::Tensor row_major_quantized_weight, size_t nbits
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant8matmul", &vecquant8matmul, "Vector 8-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant8matmul_faster", &vecquant8matmul_faster, "Vector 8-bit Quantized Matrix Multiplication (CUDA), faster version");
  m.def("preprocess_weights_for_mixed_gemm", &preprocess_weights_for_mixed_gemm, "Transformation from a row major weight matrix to the required layout of mixed type GEMM");
}
