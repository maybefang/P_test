from setuptools import setup, Extension
from torch.utils import cpp_extension

import glob
all_files = glob.glob("./*.cu")+glob.glob("./*.cc")#+glob.glob("./*.h")
'''
setup(name='gemm',
    ext_modules=[cpp_extension.CUDAExtension('gemm',["gemm.cc","gemm_kernel.cu"])],
    cmdclass={'build_ext':cpp_extension.BuildExtension})
'''
setup(name='gemm',
    ext_modules=[cpp_extension.CUDAExtension('gemm',all_files)],
    cmdclass={'build_ext':cpp_extension.BuildExtension})