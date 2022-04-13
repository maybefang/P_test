#include<torch/extension.h>
#include "cuda_runtime.h"
#include <typeinfo> 

//void calculate(float*, float*, float*, int64_t, int64_t, int64_t, int64_t);
//torch::Tensor calculate_dataingpu(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);
void calculate_dataingpu(float*, float*, float*, int64_t, int64_t, int64_t, int64_t);
void calculate_dataincpu(float*, float*, float*, int64_t, int64_t, int64_t, int64_t);
void getret_ongpu(float*,int64_t, int64_t);
/*
//正常乘法，返回矩阵
torch::Tensor mymultiply(torch::Tensor input, torch::Tensor weight, torch::Tensor bias){//a:input b:kernel
    int64_t m = input.size(0);
    int64_t k = input.size(1);
    int64_t n = weight.size(1);
    //auto ret = torch::zeros({m, n});
    //int64_t size = ret.numel();
    if(input.device().type()==torch::kCPU){
        auto ret = torch::zeros({m, n});
        int64_t size = ret.numel();
        calculate_dataincpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
        return ret;
    }
    else if(input.device().type()==torch::kCUDA){
        auto ret = torch::zeros({m, n}).cuda();
        int64_t size = ret.numel();
        //ret = ret.cuda();
        //gpu上执行任务
        calculate_dataingpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
        return ret;
    }
    //return ret;
}*/

//torch::Tensor mymultiply(torch::Tensor input, torch::Tensor weight, torch::Tensor bias){//a:input b:kernel
/*
//正常乘法，返回时间
float mymultiply(torch::Tensor input, torch::Tensor weight, torch::Tensor bias){//a:input b:kernel
    int64_t m = input.size(0);
    int64_t k = input.size(1);
    int64_t n = weight.size(1);
    //auto    ret = torch::zeros({m, n});
    //int64_t size = ret.numel();
    float elapsedTime;
    if(input.device().type()==torch::kCPU){
        auto    ret = torch::zeros({m, n});
        int64_t size = ret.numel();
        calculate_dataincpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
    }
    else if(input.device().type()==torch::kCUDA){
        //ret = ret.cuda();

        //测时间参数初始化
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        torch::Tensor  ret = torch::zeros({m, n}).cuda();
        int64_t size = ret.numel();

        //gpu上执行任务
        calculate_dataingpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
        
        //测时间停止参数
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        //print("Time cost: %3.1f ms", elapsedTime);
        //std::cout<<"in cc文件 time:"<<elapsedTime<<" ms"<<std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    //return ret;
    return elapsedTime;
}*/


//尝试直接从cuda上申请ret
torch::Tensor mymultiply(torch::Tensor input, torch::Tensor weight, torch::Tensor bias){//a:input b:kernel
    int64_t m = input.size(0);
    int64_t k = input.size(1);
    int64_t n = weight.size(1);
    
    //float elapsedTime,elapsedTime2;
    
    //auto    ret = torch::zeros({m, n});
    //int64_t size = ret.numel();
    
    if(input.device().type()==torch::kCPU){
        auto    ret = torch::zeros({m, n});
        int64_t size = ret.numel();
        calculate_dataincpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
        return ret;
    }
    else if(input.device().type()==torch::kCUDA){
        //ret = ret.cuda();
        //float* ret_float;
        int64_t size = m*n;
        torch::Device device(torch::kCUDA);
        
        //测时间参数初始化
        // cudaEvent_t start,start2,stop,stop2;
        // cudaEventCreate(&start);
        // cudaEventCreate(&start2);
        // cudaEventCreate(&stop);
        // cudaEventCreate(&stop2);

        //gpu上执行任务
        //auto ret = calculate_dataingpu(input, weight, size, m, n, k);
        //std::cout<<".cc中ret的数据类型: "<<typeid(ret).name()<<std::endl;

        //cudaEventRecord(start, 0);
        
        torch::Tensor ret = torch::zeros({m,n},device);
        
        //测时间停止参数
        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&elapsedTime, start, stop);
        // //print("Time cost: %3.1f ms", elapsedTime);
        // std::cout<<"初始化ret: "<<elapsedTime<<" ms"<<std::endl;
        // cudaEventDestroy(start);
        // cudaEventDestroy(stop);
        
        //std::cout<<"ret设备: "<<ret.device().type()<<std::endl;
        //float* ret_gpu;
        
        //cudaEventRecord(start2, 0);
        calculate_dataingpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
        
        // //测时间停止参数
        // cudaEventRecord(stop2, 0);
        // cudaEventSynchronize(stop2);
        // cudaEventElapsedTime(&elapsedTime2, start2, stop2);
        // //print("Time cost: %3.1f ms", elapsedTime);
        // std::cout<<"cc文件里总时间: "<<elapsedTime2<<" ms"<<std::endl;
        // cudaEventDestroy(start2);
        // cudaEventDestroy(stop2);
        
        //calculate_dataingpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret_gpu, size, m, n, k);
        //torch::Tensor ret_gpu_tensor=ret_gpu.toTensor({M,N});
        
        return ret;
    }
}

//torch::Tensor mymultiply_nobias(torch::Tensor input, torch::Tensor weight){//a:input b:kernel
/*
int mymultiply_nobias(torch::Tensor input, torch::Tensor weight){
    int64_t m = input.size(0);
    int64_t k = input.size(1);
    int64_t n = weight.size(1);
    //auto ret = torch::zeros({m, n});
    //int64_t size = ret.numel();
    float elapsedTime;
    if(input.device().type()==torch::kCPU){
        auto ret = torch::zeros({m, n});
        int64_t size = ret.numel();
        calculate_dataincpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
    }
    else if(input.device().type()==torch::kCUDA){
        //ret = ret.cuda();
        auto ret = torch::zeros({m, n}).cuda();
        int64_t size = m*n;
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        //gpu上执行任务
        calculate_dataingpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
 
        
        cudaEventElapsedTime(&elapsedTime, start, stop);
        //print("Time cost: %3.1f ms", elapsedTime);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    //calculate(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
    //calculate(a, b, ret, size, m, n);
    //return ret;
    return elapsedTime;
}*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("mymultiply", &mymultiply, "a test");
}
