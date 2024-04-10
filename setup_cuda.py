from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        name='quant_cuda', sources=['quant_cuda.cpp', 'quant_cuda_kernel.cu'],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-I./', '-I./3rdparty/cutlass-extension', \
            '-I./3rdparty/cutlass-extension/3rdparty/cutlass/include', '-I/usr/local/cuda/include', \
            '-L./3rdparty/cutlass-extension/build/lib', '-ltransformer-shared', '-lcutlass_heuristic']}
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
