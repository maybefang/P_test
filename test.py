import torch
import gemm
import numpy as np
import phello
import time
'''
# 测乘法时间
#a = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0],[10.,11.,12.]]).cuda()
#b = torch.tensor([[1.],[2.],[3.]]).cuda()
a = torch.randn(64,3*32*32,dtype=torch.float).cuda()
b = torch.randn(3*32*32,1,dtype=torch.float).cuda()
bias = torch.zeros(1)
loops=1000
c=0
start, end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)

for i in range(10):
    warmupc=gemm.mymultiply(a,b,bias)
start.record()
for i in range(loops):
    c += gemm.mymultiply(a,b,bias)
end.record()
torch.cuda.synchronize()
mytimepython = start.elapsed_time(end)/loops
mytime = c/loops

b = b.t()
for i in range(10):
    warmupd=torch.nn.functional.linear(b,a)
start.record()
for i in range(loops):
    d = torch.nn.functional.linear(b,a)
end.record()
torch.cuda.synchronize()
pytorchtime = start.elapsed_time(end)/loops

print("mytime in c++:",mytime)
print("mytime in python:",mytimepython)
print("pytorch time:",pytorchtime)
'''

'''
#测试返回矩阵直接在gpu中初始化
a = torch.randn(64,3*32*32,dtype=torch.float).cuda()
b = torch.randn(3*32*32,1,dtype=torch.float).cuda()
#a = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0],[10.0,11.0,12.0]]).t().cuda()
#b = torch.tensor([[1.],[2.],[3.]]).t().cuda()
bias = torch.zeros(1)
c = gemm.mymultiply(a,b,bias)
print()
print("c size:",c.size())
print()
'''
#将im2col返回直接变为原来的转置
'''
a_3 = torch.tensor([[[[1.0,2.0,3.0],
                    [4.0,5.0,6.0],
                    [7.0,8.0,9.0]],
                    
                   [[2.0,2.0,3.0],
                    [4.0,5.0,6.0],
                    [7.0,8.0,9.0]]],

                  [[[3.0,12.0,13.0],
                    [14.0,15.0,16.0],
                    [17.0,18.0,19.0]],
                      
                   [[4.0,12.0,13.0],
                    [14.0,15.0,16.0],
                    [17.0,18.0,19.0]]]]).cuda()#[2,2,3,3]
'''
'''
a = torch.tensor([[[[1.0,2.0,3.0],
                    [4.0,5.0,6.0],
                    [7.0,8.0,9.0],
                    [1.0,1.0,1.0]],
                    
                   [[2.0,2.0,3.0],
                    [4.0,5.0,6.0],
                    [7.0,8.0,9.0],
                    [2.0,2.0,2.0]]],

                  [[[3.0,12.0,13.0],
                    [14.0,15.0,16.0],
                    [17.0,18.0,19.0],
                    [3.0,3.0,3.0]],
                      
                   [[4.0,12.0,13.0],
                    [14.0,15.0,16.0],
                    [17.0,18.0,19.0],
                    [4.0,4.0,4.0]]]]).cuda()#[2,2,4,3]
'''

a = torch.randn(64,3,32,32,dtype=torch.float).cuda()

bias = torch.zeros(1)
N, C, H, W = a.shape
stride = 1
ksize = 3
pad=0
out_h = (H + 2 * pad - ksize) // stride + 1
out_w = (W + 2 * pad - ksize) // stride + 1
strides = (C*H*W,H*W,W*stride,stride,W*stride,stride)

print("N:",N,"  out_h:",out_h,"  out_w:",out_w,"  C:",C,"  ksize:",ksize)

A = torch.as_strided(a, size=(N,C,out_h,out_w,ksize,ksize), stride=strides)
col = A.permute(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w, -1)
col_t = A.permute(1,4,5,0,2,3).reshape(-1, N*out_h*out_w)

print(col.t().equal(col_t))#True
