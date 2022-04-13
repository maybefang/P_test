// CUDA runtime 库 + CUBLAS 库 
#include "cuda_runtime.h"
#include "cublas_v2.h"
 
#include <time.h>
#include <iostream> 
#include<torch/extension.h>
#include <thrust/device_vector.h>

#include <typeinfo>

#include "cuda.h"
/*
//正常计算output，且测量时间
void calculate_dataingpu(float* d_input, float* d_weight, float* d_output, int64_t size, int64_t M, int64_t N, int64_t K) 
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
 
    // 同步函数
    cudaDeviceSynchronize();
 
    //测时间参数初始化
    float elapsedTime;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

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
    //std::cout<<"in .cu time:"<<elapsedTime<<" ms"<<std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 释放 CUBLAS 库对象
    cublasDestroy (handle);
    //std::cout<<"最后释放库对象"<<std::endl;

}
*/
/*
//正常计算output，没有测量时间
void calculate_dataingpu(float* d_input, float* d_weight, float* d_output, int64_t size, int64_t M, int64_t N, int64_t K) 
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


    // 释放 CUBLAS 库对象
    cublasDestroy (handle);

}*/

//尝试在gpu中生成返回值空间
//torch::Tensor calculate_dataingpu(torch::Tensor d_input, torch::Tensor d_weight, int64_t size, int64_t M, int64_t N, int64_t K) 
void calculate_dataingpu(float* d_input, float* d_weight,float* d_output, int64_t size, int64_t M, int64_t N, int64_t K) 
{   
    
    //thrust::device_vector<float> d_output(M*N,0);
    //float * d_output_ptr = thrust::raw_pointer_cast(d_output.data());
    
    //float * d_output_ptr = thrust::raw_pointer_cast(&d_output[0]);
    
    //float elapsedTime;

    //float* pp;
    //cudaMalloc (
    //    (void**)&pp,        // 指向开辟的空间的指针
    //    M*N * sizeof(float)     //　需要开辟空间的字节数
    // );

    //std::cout << typeid(thrust::raw_pointer_cast(&d_output[0])).name() << std::endl;

    // 定义状态变量
    //cublasStatus_t status;
    
    // 同步函数
    //cudaDeviceSynchronize();
 
    //测时间参数初始化
    // cudaEvent_t start,stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);

    // 创建并初始化 CUBLAS 库对象
    cublasHandle_t handle=blas_handle();
    
    //cublasHandle_t handle;
    //status = cublasCreate_v2(&handle);

    
    //测时间停止参数
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // //print("Time cost: %3.1f ms", elapsedTime);
    // std::cout<<"申请handle: "<<elapsedTime<<" ms"<<std::endl;

    // if (status != CUBLAS_STATUS_SUCCESS)
    // {
        
    //     if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
    //         std::cout << "CUBLAS 对象实例化出错" << std::endl;
    //     }
    //     getchar ();
    //     //return EXIT_FAILURE;
        
    //     return;// torch::from_blob(thrust::raw_pointer_cast(&d_output[0]),{M,N});
    // }
 
    // 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
    float a=1; float b=0;

    // 同步函数
    //cudaDeviceSynchronize();

    //cudaEventRecord(start, 0);//计时开始

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
    //cudaDeviceSynchronize();

    //测时间停止参数
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // //print("Time cost: %3.1f ms", elapsedTime);
    // std::cout<<"cublas计算时间: "<<elapsedTime<<" ms"<<std::endl;

    //cudaEventRecord(start, 0);//计时开始
    // 释放 CUBLAS 库对象
    //cublasDestroy (handle);

    //测时间停止参数
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // //print("Time cost: %3.1f ms", elapsedTime);
    // std::cout<<"释放handle时间: "<<elapsedTime<<" ms"<<std::endl;

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    
    //std::cout<<d_output[0]<<std::endl;
    //torch::Tensor put.data_ptr<float>() = torch::from_blob(d_output_ptr,{M,N});
    //auto put = torch::from_blob(d_output_ptr,{M,N},d_input.options()).clone();
    //std::cout <<".cu中put的设备: "<<put.device().type() << std::endl;
    //std::cout<<put<<std::endl;
    //return put;
}

void calculate_dataincpu(float* h_input, float* h_weight, float* h_output, int64_t size, int64_t M, int64_t N, int64_t K) 
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
    //在 显存 中为将要存放运算结果的矩阵开辟空间
    cudaMalloc (
       (void**)&d_output,        // 指向开辟的空间的指针
       M*N * sizeof(float)     //　需要开辟空间的字节数
    );

    
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

    
    // 从 显存 中取出运算结果至 内存中去
    
    cublasGetVector (
        M*N,    //  要取出元素的个数
        sizeof(float),    // 每个元素大小
        d_output,    // GPU 端起始地址
        1,    // 连续元素之间的存储间隔
        h_output,    // 主机端起始地址
        1    // 连续元素之间的存储间隔
    );
    
    cudaFree (d_input);
    cudaFree (d_weight);
    cudaFree (d_output);

    
    // 释放 CUBLAS 库对象
    cublasDestroy (handle);
}










