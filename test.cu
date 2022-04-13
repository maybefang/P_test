// CUDA runtime 库 + CUBLAS 库 
#include "cuda_runtime.h"
#include "cublas_v2.h"
 
#include <time.h>
#include <iostream> 
#include<unistd.h>

//void calculate_kernel(float* d_input, float* d_weight, float* d_output, float* bias, int64_t size, int64_t M, int64_t N, int64_t K) 
void calculate_dataingpu(float* h_input, float* h_weight, float* h_output, int64_t size, int64_t M, int64_t N, int64_t K) 
{   
    // 定义状态变量
    cublasStatus_t status;
    

    // 创建并初始化 CUBLAS 库对象
    cublasHandle_t handle;
    status = cublasCreate_v2(&handle);
    
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            std::cout << "CUBLAS 对象实例化出错" << std::endl;
        }
        getchar ();
        //return EXIT_FAILURE;
        
        return;
    }

    float *d_input, *d_weight;
    cudaMalloc (
        (void**)&d_input,   
        K*M * sizeof(float)   
     );
    cudaMalloc (
        (void**)&d_weight,
        N*K * sizeof(float)
    );


    float *d_output;
    //为d_output申请空间
    cudaMalloc (
        (void**)&d_output,        // 指向开辟的空间的指针
        M*N * sizeof(float)     //　需要开辟空间的字节数
     );
 
    
    cublasSetVector (
        K*M,    
        sizeof(float), 
        h_input,   
        1,    
        d_input,  
        1    
    );
    cublasSetVector (
        N*K,
        sizeof(float),
        h_weight,
        1,
        d_weight,
        1
    );

    float elapsedTime;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // 同步函数
    cudaDeviceSynchronize();
 
    // 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
    float a=1; float b=0;
    // 矩阵相乘。该函数必然将数组解析成列优先数组
    cublasSgemm (
        handle,    // blas 库对象 
        CUBLAS_OP_T,    // 矩阵 A 属性参数
        CUBLAS_OP_T,    // 矩阵 B 属性参数
        M,    // A, C 的行数 
        N,    // B, C 的列数
        K,    // A 的列数和 B 的行数
        &a,    // 运算式的 α 值
        d_input,    // A 在显存中的地址
        K,    // lda，因为是列优先，所以此处传入每列多少元素
        d_weight,    // B 在显存中的地址
        N,    // ldb，同lda
        &b,    // 运算式的 β 值
        d_output,    // C 在显存中的地址(结果矩阵)
        M    // ldc
    );
    
    // 同步函数
    cudaDeviceSynchronize();
    //测时间停止参数
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    //print("Time cost: %3.1f ms", elapsedTime);
    std::cout<<"in time:"<<elapsedTime<<std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cublasGetVector (
        M*N,    
        sizeof(float),    
        d_output,    
        1,    
        h_output,    
        1    
    );
    cudaFree(d_output);
    // 释放 CUBLAS 库对象
    cublasDestroy (handle);
    //std::cout<<"最后释放库对象"<<std::endl;
}
int main(){
    float a[12]={1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0};
    float b[3]={1.0,2.0,3.0};
    //torch::Tensor atensor = torch::Tensor(a,{4,3}).cuda();
    //torch::Tensor btensor = torch::Tensor(b,{3}).cuda();
    float c[4]={0};
    //测时间参数初始化
    float elapsedTime;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    calculate_dataingpu(a, b, c, 5, 4, 1, 3);
    //测时间停止参数
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    //print("Time cost: %3.1f ms", elapsedTime);
    std::cout<<"time:"<<elapsedTime<<" ms"<<std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}