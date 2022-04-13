from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='phello',
    ext_modules=[cpp_extension.CUDAExtension('phello',['im2col.cc'])],
    cmdclass={'build_ext':cpp_extension.BuildExtension})
