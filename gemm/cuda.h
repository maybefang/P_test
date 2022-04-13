#ifndef CUDA_H
#define CUDA_H
void check_error(cudaError_t status);
cublasHandle_t blas_handle();
#endif