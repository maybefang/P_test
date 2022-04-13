import torch
import numpy as np
#import minpy.numpy as np
from numpy.lib.stride_tricks import as_strided
#from minpy.numpy.lib.stride_tricks import as_strided
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, datasets
#pytorch load的data.shape：[batchsize, inchannel,h,w]
import gemm
import time


def im2col(input_data, ksize, out_h, out_w, input_shape, stride=1, pad=0):
    N, C, H, W = input_shape
    #out_h = (H + 2 * pad - ksize) // stride + 1
    #out_w = (W + 2 * pad - ksize) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    #print(img)
    #col = np.zeros((N, C, ksize, ksize, out_h, out_w))

    input_data = input_data.numpy()
    strides = (*input_data.strides[:-2], input_data.strides[-2]*stride, input_data.strides[-1]*stride, *input_data.strides[-2:])
    A = as_strided(input_data, shape=(N,C,out_h,out_w,ksize,ksize), strides=strides)
    col = A.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    #col.astype(float32)
    #print("col.shape:",col.shape)
    #print("col:",col)
    return col

#tensor版
def im2col_tensor(input_data, ksize, out_h, out_w, input_shape, stride=1, pad=0):
    N, C, H, W = input_shape
    #out_h = (H + 2 * pad - ksize) // stride + 1
    #out_w = (W + 2 * pad - ksize) // stride + 1

    img = torch.nn.functional.pad(input_data, (pad, pad, pad, pad, 0, 0), "constant", value=0)
    #print(img)
    #col = np.zeros((N, C, ksize, ksize, out_h, out_w))

    #strides = (*input_data.strides[:-2], input_data.strides[-2]*stride, input_data.strides[-1]*stride, *input_data.strides[-2:])
    strides = (C*H*W,H*W,W*stride,stride,W*stride,stride)
    
    A = torch.as_strided(img, size=(N,C,out_h,out_w,ksize,ksize), stride=strides)
    col = A.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


inputdata = torch.randn(64,3,32,32,dtype=torch.float).cuda()
weight = torch.randn(64,3,3,3,dtype=torch.float).cuda()
bias = nn.Parameter(torch.zeros(1))
lmask = torch.tensor([0.,1.,0.,1.,1.,1.,0.,1.,0.,1.,1.,1.,1.,1.,0.,0.,1.,1.,0.,0.,0.,1.,1.,1.,0.,1.,0.]).cuda()
mask = torch.tensor([[[0.,1.,1.],[1.,1.,1.],[1.,0.,0.]],[[1.,0.,1.],[1.,1.,1.],[1.,0.,1.]],[[1.,1.,1.],[0.,1.,1.],[0.,1.,1.]]]).cuda()

linput = torch.randn(64,128,dtype=torch.float).cuda()
lweight = torch.randn(10,128,dtype=torch.float).cuda()

'''
start, end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)

inputdata = torch.randn(64,3,32,32,dtype=torch.float)
weight = torch.randn(64,3,3,3,dtype=torch.float)
bias = nn.Parameter(torch.zeros(1))
lmask = torch.tensor([0.,1.,0.,1.,1.,1.,0.,1.,0.,1.,1.,1.,1.,1.,0.,0.,1.,1.,0.,0.,0.,1.,1.,1.,0.,1.,0.])
mask = torch.tensor([[[0.,1.,1.],[1.,1.,1.],[1.,0.,0.]],[[1.,0.,1.],[1.,1.,1.],[1.,0.,1.]],[[1.,1.,1.],[0.,1.,1.],[0.,1.,1.]]])


linput = torch.randn(64,128,dtype=torch.float)
lweight = torch.randn(10,128,dtype=torch.float)

loops=1000
'''
#im2col一些参数初始化
input_shape = inputdata.shape
N, C, H, W = input_shape
k_h,k_w =3,3
pad=0
stride=1
out_h = (H + 2 * pad - k_h) // stride + 1
out_w = (W + 2 * pad - k_w) // stride + 1
'''
#测im2col时间
cpu_start = time.time()#cpu
for i in range(loops):
    im2col_out=im2col(inputdata,k_h,out_h, out_w, input_shape)
cpu_end = time.time()-cpu_start


im2col_time = cpu_end/loops  #cpu
#im2col_time=sum(im2col_times)/loops  #gpu

print()
print("my im2col time in cpu:",im2col_time)

print()

inputdata = inputdata.cuda()

start.record()
for i in range(loops):
    im2col_tensor_out=im2col_tensor(inputdata,k_h,out_h, out_w, input_shape)
end.record()
torch.cuda.synchronize()
im2col_time=start.elapsed_time(end)/loops

print()
print("my im2col time in gpu:",im2col_time)

print()
'''

start, end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
mask = mask.reshape(-1)#mask变为1维
idx = [i.item() for i in mask.nonzero()]
input_tensor=im2col_tensor(inputdata,k_h,out_h, out_w, input_shape)

start.record()
input_tile = torch.index_select(input_tensor,1,torch.tensor(idx).cuda())#输入重新拼接成密集矩阵
end.record()
torch.cuda.synchronize()
time=start.elapsed_time(end)
print("列跳过:",time)

start.record()
input_tensor = input_tensor.t()
end.record()
torch.cuda.synchronize()
time=start.elapsed_time(end)
print("转置:",time)

start.record()
input_tile = torch.index_select(input_tensor,0,torch.tensor(idx).cuda())#输入重新拼接成密集矩阵
end.record()
torch.cuda.synchronize()
time=start.elapsed_time(end)
print("行跳过:",time)
