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

#正常的im2col且在cpu中
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
    col = A.transpose(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w, -1)
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
    #col = A.permute(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w, -1)#之后进行列跳过
    #col = A.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)#之后进行列跳过,网上顺序
    col = A.permute(1,4,5,0,2,3).reshape(-1, N*out_h*out_w)#之后进行行跳过
    return col


def myconv2d(input, kernel, stride=1, pad=0, bias=0):
    input_shape = input.shape
    N, C, H, W = input_shape
    k_h,k_w = kernel.shape[-2],kernel.shape[-1]
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1
    #input_tile = torch.from_numpy(im2col(a,k_h,out_h,out_w,input_shape)).float()
    input_tensor = im2col_tensor(input,k_h,out_h,out_w,input_shape)
    #print(input_tile)
    #print()
    #print(input_tensor)
    
    kernel_shape = kernel.shape
    kernel_tile = kernel.reshape(-1,kernel_shape[1]*kernel_shape[2]*kernel_shape[3]).t()
    #print(kernel_tile.type)
    output = gemm.mymultiply(input_tile,kernel_tile,bias)
    output = output.reshape(N,kernel_shape[0],out_h,-1) #[batch_size, output_channel(kernel_size[0]), out_h, out_w]
    return output
    
#有mask的卷积层算法
def sparse_myconv2d(input, kernel, mask, stride=1, pad=0, bias=0):

    #初始化计算处理数据时间和卷积时间event（测试时间包括im2col、剪枝（重新拼接）和gemm）
    im2col_start, im2col_end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    processing_start, processing_end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    processing_weight_start, processing_weight_end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    gemm_start, gemm_end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)

    #计算im2col时间
    im2col_start.record()
    input_shape = input.shape
    N, C, H, W = input_shape
    k_h,k_w = kernel.shape[-2],kernel.shape[-1]
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1
    
    #input_tile = torch.from_numpy(im2col(input,k_h,out_h,out_w,input_shape)).float()
    input_tensor = im2col_tensor(input,k_h,out_h,out_w,input_shape)#将输入平铺
    kernel_shape = kernel.shape
    kernel_tile = kernel.reshape(-1,kernel_shape[1]*kernel_shape[2]*kernel_shape[3]).t()#权重平铺
    im2col_end.record()
    torch.cuda.synchronize()
    im2col_time=im2col_start.elapsed_time(im2col_end)

    #计算剪枝时间
    mask = mask.reshape(-1)#mask变为1维
    idx = [i.item() for i in mask.nonzero()]
    idx = torch.tensor(idx).cuda()
    
    processing_start.record()
    #input_tile = torch.index_select(input_tensor,1,idx)#输入重新拼接成密集矩阵
    input_tile = torch.index_select(input_tensor,0,idx).t()#输入重新拼接成密集矩阵,行跳过并转置
    processing_end.record()
    torch.cuda.synchronize()
    processing_time=processing_start.elapsed_time(processing_end)

    processing_weight_start.record()
    kernel_tile = torch.index_select(kernel_tile,0,idx)#权重重新拼接
    processing_weight_end.record()
    torch.cuda.synchronize()
    processing_weight_time=processing_weight_start.elapsed_time(processing_weight_end)
    
    #计算gemm时间
    gemm_start.record()
    output = gemm.mymultiply(input_tile,kernel_tile,bias) #返回值可以为c++中测的时间
    gemm_end.record()
    torch.cuda.synchronize()
    gemm_time=gemm_start.elapsed_time(gemm_end)
    
    #output = output.reshape(N,kernel_shape[0],out_h,-1) #[batch_size, output_channel(kernel_size[0]), out_h, out_w]
    #output = output.reshape(N,-1,kernel_shape[0]).permute(0, 2, 1).reshape(N,kernel_shape[0],out_h,-1)
    return output,im2col_time,processing_time,processing_weight_time,gemm_time
    


#a = torch.tensor([[[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]],[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]],[[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]],[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]]])
#print(a.shape)
def testpytorch(inputdata,weight,mask_pytorch):
    #处理数据和进行卷积的event
    #processing_start, processing_end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    #conv2d_start, conv2d_end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)

    #计算处理数据时间
    #processing_start.record()
    masked_weight = torch.einsum('ijkl,jkl->ijkl', weight, mask_pytorch)
    #processing_end.record()
    #torch.cuda.synchronize()
    #processing_time=processing_start.elapsed_time(processing_end)

    #计算卷积时间（包括im2col和gemm）
    #conv2d_start.record()
    conv_out_pytorch = torch.nn.functional.conv2d(inputdata, masked_weight, bias=None, stride=1,
            padding=0, dilation=1, groups=1)
    #conv2d_end.record()
    #torch.cuda.synchronize()
    #conv2d_time=conv2d_start.elapsed_time(conv2d_end)
    #return processing_time,conv2d_time
            

def testmyconv2d(inputdata,weight,mask,bias):
    conv_out,im2col_time,processingtime,processingweighttime,gimmtime = sparse_myconv2d(inputdata, weight, mask, bias=bias)
    #conv_out = sparse_myconv2d(inputdata, weight, mask, bias=bias)
    #sparse_myconv2d(inputdata, weight, mask, bias=bias)
    return conv_out,im2col_time,processingtime,processingweighttime,gimmtime
    
def testpytorchlinear(inputdata,weight,mask):
    input_shape=inputdata.shape
    N, C, H, W = input_shape
    k_h,k_w = 3,3
    pad=0
    stride=1
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1
    inputdata = im2col_tensor(inputdata, 3, out_h, out_w, input_shape, stride=1, pad=0)
    weight=weight.reshape(-1,27).t()
    #测试pytorch的矩阵乘
    start, end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    masked_weight = torch.einsum('ij,i->ij', weight, mask).t()#[27,-1]

    start.record()
    output = torch.nn.functional.linear(inputdata, masked_weight, None)
    end.record()
    torch.cuda.synchronize()
    time_linear=start.elapsed_time(end)

    masked_weight = masked_weight.t()
    start.record()
    output = torch.mm(inputdata, masked_weight)
    end.record()
    torch.cuda.synchronize()
    time_mm=start.elapsed_time(end)

    return time_linear,time_mm
    
def testmylinear(inputdata,weight,mask,bias):
    weight = weight.t()
    idx = [i.item() for i in mask.nonzero()]
    input_tile = torch.index_select(input_tensor,1,torch.tensor(idx).cuda())
    kernel_tile = torch.index_select(kernel_tile,0,torch.tensor(idx).cuda())
    output = gemm.mymultiply(input_tile,kernel_tile,bias)


#test myconv2d and pytorch time or mymultiply and pytorch linear
inputdata = torch.randn(64,3,32,32,dtype=torch.float).cuda()
weight = torch.randn(64,3,3,3,dtype=torch.float).cuda()
bias = nn.Parameter(torch.zeros(1))
lmask = torch.tensor([0.,1.,0.,1.,1.,1.,0.,1.,0.,1.,1.,1.,1.,1.,0.,0.,1.,1.,0.,0.,0.,1.,1.,1.,0.,1.,0.]).cuda()
mask = torch.tensor([[[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]],[[1.,0.,0.],[1.,0.,0.],[0.,0.,0.]],[[0.,0.,1.],[0.,0.,0.],[0.,0.,1.]]]).cuda()

linput = torch.randn(64,128,dtype=torch.float).cuda()
lweight = torch.randn(10,128,dtype=torch.float).cuda()


start, end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)

loops=1000
#测pytorch数据处理和conv2d
'''
for i in range(10):
    testpytorch(inputdata,weight,mask)
    

pytorch_processing_times,pytorch_conv2d_times = [0 for i in range(loops)],[0 for i in range(loops)]
start.record()
for i in range(loops):
    #pytorch_processing_times[i],pytorch_conv2d_times[i] =testpytorch(inputdata,weight,mask)
    testpytorch(inputdata,weight,mask)
end.record()
torch.cuda.synchronize()
pytorchtime = start.elapsed_time(end)/loops
#pytorch_processing_time = sum(pytorch_processing_times)/loops
#pytorch_conv2d_time = sum(pytorch_conv2d_times)/loops
'''
#测pytorch矩阵乘
'''
for i in range(10):
    testpytorchlinear(inputdata,weight,lmask)

pytorch_linear_times,pytorch_mm_times=[0 for i in range(loops)],[0 for i in range(loops)]
for i in range(loops):
    pytorch_linear_times[i],pytorch_mm_times[i]=testpytorchlinear(inputdata,weight,lmask)
pytorch_linear_time = sum(pytorch_linear_times)/loops
pytorch_mm_time = sum(pytorch_mm_times)/loops
'''
#测ours

for i in range(10):
    testmyconv2d(inputdata,weight,mask,bias)

gemminc_times,im2col_times,processing_times,gemm_times=[0 for i in range(loops)],[0 for i in range(loops)],[0 for i in range(loops)],[0 for i in range(loops)]
processing_times_weights = [0 for i in range(loops)]

#start.record()
for i in range(loops):
    #testmyconv2d(inputdata,weight,mask,bias)
    gemminc_times,im2col_times[i],processing_times[i],processing_times_weights[i],gemm_times[i]=testmyconv2d(inputdata,weight,mask,bias)
#end.record()
#torch.cuda.synchronize()
#myconv2dtime = start.elapsed_time(end)/loops

#gemminc_time=sum(gemminc_times)/loops
im2col_time=sum(im2col_times)/loops
processing_time=sum(processing_times)/loops
processing_time_weight=sum(processing_times_weights)/loops
gemm_time=sum(gemm_times)/loops




#print("pytorch total time:",pytorchtime)
#print("pytorch processing time:",pytorch_processing_time)
#print("pytorch conv2c time:",pytorch_conv2d_time)
#print("pytorch linear time:",pytorch_linear_time)
#print("pytorch mm time:",pytorch_mm_time)
print()
#print("my toal time:",myconv2dtime)
#print("my im2col time:",im2col_time)
print("my processing time:",processing_time)
print("my processing weight time:",processing_time_weight)
#print("my gemm time in python:",gemm_time)
#print("my gemm time in c++:",gemminc_time)
#print()
'''
inputdata = torch.randn(64,3,32,32,dtype=torch.float).cuda()
weight = torch.randn(64,3,3,3,dtype=torch.float).cuda()
'''
'''
inputdata = torch.tensor([[[[1.0,2.0,3.0],
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

weight = torch.tensor([[[[1.0,2.0,3.0],
                    [4.0,5.0,6.0],
                    [1.0,1.0,1.0]],
                    
                   [[2.0,2.0,3.0],
                    [4.0,5.0,6.0],
                    [2.0,2.0,2.0]]]]).cuda()#[2,2,3,3]
'''
'''
bias = nn.Parameter(torch.zeros(1))

mask = torch.tensor([[[0.,1.,1.],[1.,1.,1.],[1.,0.,0.]],[[1.,0.,1.],[1.,1.,1.],[1.,0.,1.]],[[1.,1.,1.],[0.,1.,1.],[0.,1.,1.]]]).cuda()
#mask = torch.tensor([[[0.,1.,1.],[1.,1.,1.],[1.,0.,0.]],[[1.,0.,1.],[1.,1.,1.],[1.,0.,1.]]]).cuda()

my_out,a,b,c = sparse_myconv2d(inputdata, weight, mask, bias=bias)

masked_weight = torch.einsum('ijkl,jkl->ijkl', weight, mask)
pytorch_out = torch.nn.functional.conv2d(inputdata, masked_weight, bias=None, stride=1,
            padding=0, dilation=1, groups=1)
pytorch_out2 = torch.nn.functional.conv2d(inputdata, masked_weight, bias=None, stride=1,
            padding=0, dilation=1, groups=1)
#print(pytorch_out)
#print(my_out)
#my_out = my_out.reshape(1,-1)
#pytorch_out = pytorch_out.reshape(1,-1)
#chazhi = pytorch_out-my_out
#print(sum(chazhi))
print(my_out.equal(pytorch_out))
'''