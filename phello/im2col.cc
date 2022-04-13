#include<torch/extension.h>
#include <chrono>
void phello(int a,int b,int *c){
    auto begin = std::chrono::high_resolution_clock::now();
    //int c;
    *c=a*b;
    std::cout<<"c:"<<c<<std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    std::cout<<"time:"<<elapsed.count()*(1e-9)<<std::endl;
    //return c
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("phello", &phello, "print hello");
}