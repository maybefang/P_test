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

#测自己的结果和pytorch是不是一样
'''
a = torch.tensor([[[[1111.0,1112.0,1113.0,1114.0,1115.0],
                    [1121.0,1122.0,1123.0,1124.0,1125.0],
                    [1131.0,1132.0,1133.0,1134.0,1135.0],
                    [1141.0,1142.0,1143.0,1144.0,1145.0]],
                    
                   [[1211.0,1212.0,1213.0,1214.0,1215.0],
                    [1221.0,1222.0,1223.0,1224.0,1225.0],
                    [1231.0,1232.0,1233.0,1234.0,1235.0],
                    [1241.0,1242.0,1243.0,1244.0,1245.0]],
                    
                   [[1311.0,1312.0,1313.0,1314.0,1315.0],
                    [1321.0,1322.0,1323.0,1324.0,1325.0],
                    [1331.0,1332.0,1333.0,1334.0,1335.0],
                    [1341.0,1342.0,1343.0,1344.0,1345.0]]],

                  [[[2111.0,2112.0,2113.0,2114.0,2115.0],
                    [2121.0,2122.0,2123.0,2124.0,2125.0],
                    [2131.0,2132.0,2133.0,2134.0,2135.0],
                    [2141.0,2142.0,2143.0,2144.0,2145.0]],
                      
                   [[2211.0,2212.0,2213.0,2214.0,2215.0],
                    [2221.0,2222.0,2223.0,2224.0,2225.0],
                    [2231.0,2232.0,2233.0,2234.0,2235.0],
                    [2241.0,2242.0,2243.0,2244.0,2245.0]],

                   [[2311.0,2312.0,2313.0,2314.0,2315.0],
                    [2321.0,2322.0,2323.0,2324.0,2325.0],
                    [2331.0,2332.0,2333.0,2334.0,2335.0],
                    [2341.0,2342.0,2343.0,2344.0,2345.0]]]]).cuda()#[2,3,4,5]

#atest = torch.from_numpy(np.arange(648).reshape(24,27)).float().cuda()

atest = torch.randn(12,27).cuda()
#a = torch.randn(64,3,32,32,dtype=torch.float).cuda()

#b = torch.tensor([[[[0.00001,0.00001,0.00001],
#                    [0.00001,0.00001,0.00001],
#                    [0.00001,0.00001,0.00001]],
#                    
#                   [[0.00001,0.00001,0.00001],
#                    [0.00001,0.00001,0.00001],
#                    [0.00001,0.00001,0.00001]],
#
#                   [[0.00001,0.00001,0.00001],
#                    [0.00001,0.00001,0.00001],
#                    [0.00001,0.00001,0.00001]]],
#                    
#                  [[[0.0001,0.0001,0.0001],
#                    [0.0001,0.0001,0.0001],
#                    [0.0001,0.0001,0.0001]],
#                    
#                   [[0.001,0.001,0.001],
#                    [0.001,0.001,0.001],
#                    [0.001,0.001,0.001]],
#                    
#                   [[0.01,0.01,0.01],
#                    [0.01,0.01,0.01],
#                    [0.01,0.01,0.01]]]]).cuda()#[2,3,3,3]

b = torch.tensor([[[[0.01,0.01,0.01],
                    [0.01,0.01,0.01],
                    [0.01,0.01,0.01]],
                    
                   [[0.01,0.01,0.01],
                    [0.01,0.01,0.01],
                    [0.01,0.01,0.01]],
                    
                   [[0.01,0.01,0.01],
                    [0.01,0.01,0.01],
                    [0.01,0.01,0.01]]],
                    
                  [[[0.01,0.01,0.01],
                    [0.01,0.01,0.01],
                    [0.01,0.01,0.01]],
                    
                   [[0.01,0.01,0.01],
                    [0.01,0.01,0.01],
                    [0.01,0.01,0.01]],
                    
                   [[0.01,0.01,0.01],
                    [0.01,0.01,0.01],
                    [0.01,0.01,0.01]]]]).cuda()#[2,3,3,3]

#btest = torch.from_numpy(np.arange(648,702).reshape(27,2)).float().cuda()
btest = torch.randn(2,3,3,3).cuda()
bias = torch.zeros(1).cuda()

bshape = b.shape
b_tile_t = b.reshape(bshape[0],-1)
b_tile = b_tile_t.t()#[-1,out_c]

# test if matrixs are same
N, C, H, W = a.shape
stride = 1
ksize = 3
pad=0
out_h = (H + 2 * pad - ksize) // stride + 1
out_w = (W + 2 * pad - ksize) // stride + 1
strides = (C*H*W,H*W,W*stride,stride,W*stride,stride)

A = torch.as_strided(a, size=(N,C,out_h,out_w,ksize,ksize), stride=strides)
col = A.permute(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w, -1)#正常卷积
col_t = A.permute(1,4,5,0,2,3).reshape(-1, N*out_h*out_w)#加速用这个

#my_out = gemm.mymultiply(col,b_tile,bias)
#pytorch_linear_out = torch.nn.functional.linear(col,b_tile.t())
#pytorch_mm_out = torch.mm(col,b_tile)

#print("col shape:",col.shape," b_tile shape:",b_tile.shape)
#testmy_out = gemm.mymultiply(atest,btest,bias)
#testpytorch_linear_out = torch.nn.functional.linear(atest,btest.t())
#testpytorch_mm_out = torch.mm(atest,btest)
#pytorch_conv2d_out = torch.nn.functional.conv2d(a,b,bias=None)
#my_conv2d = my_out.reshape(N,-1,2).permute(0, 2, 1).reshape(N,2,out_h,-1)
#print(col.shape)
#print(b_tile.shape)
#print(col.t().equal(col_t))#True
#print("A:")
#print(col)
#print("B:")
#print(b_tile)
print(pytorch_mm_out.equal(my_out))
#print(pytorch_conv2d_out.equal(my_conv2d))
#print(my_conv2d)
#print(torch.mm(atest,b_tile).equal(my_out))#pytorch_linear_out)
#print(my_out)
#print(pytorch_conv2d_out)
#print(testmy_out)
#print(pytorch_mm_out)
#print(pytorch_linear_out.equal(pytorch_mm_out))
'''
'''
start, end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
loops = 100
start.record()
for i in range(loops):
    my_out = gemm.mymultiply(col,b_tile,bias)
end.record()
torch.cuda.synchronize()
mytime = start.elapsed_time(end)/loops

start.record()
for i in range(loops):
    pytorch_linear_out = torch.nn.functional.linear(col,b_tile.t())
end.record()
torch.cuda.synchronize()
pytorch_linear_time = start.elapsed_time(end)/loops

start.record()
for i in range(loops):
    pytorch_mm_out = torch.mm(col,b_tile)
end.record()
torch.cuda.synchronize()
pytorchtime = start.elapsed_time(end)/loops


print("mytime:",mytime)
print("pytorch.mm time:",pytorchtime)
print("pytorch linear time:",pytorch_linear_time)
'''

''''
#测试cublasSgemm中参数影响
atest = torch.tensor([[1.,2.,3.],[0.1,0.2,0.3]]).cuda()
#btest = torch.tensor([[0.4,7],[0.5,8],[0.6,9]]).cuda()
btest = torch.tensor([[0.4,7,1.1,14],[0.5,8,1.2,15],[0.6,9,1.3,16]]).cuda()
bias = torch.zeros(1)
c = gemm.mymultiply(atest,btest,bias)
print(atest)
print()
print(btest)
print()
print(c)
'''

#test ours pruning time
mask = torch.tensor([[[0,1,0],
                      [1,1,0],
                      [0,0,0]],

                     [[1,0,0],
                      [1,0,0],
                      [0,1,1]],
                     
                     [[0,0,1],
                      [1,0,1],
                      [0,0,0]]]).cuda()


start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
loops=100

start.record()
mask_tile = mask.reshape(-1)
idx = [i.item() for i in mask_tile.nonzero()]
idx = torch.tensor(idx).cuda()
end.record()
torch.cuda.synchronize()
time_mask=start.elapsed_time(end)
    
a = torch.randn(27,12).cuda()

for i in range(10):
    pruning_a = torch.index_select(a,0,idx)
    
start.record()
for i in range(loops):
    pruning_a = torch.index_select(a,0,idx)
end.record()
torch.cuda.synchronize()
time0=start.elapsed_time(end)/loops

a = a.t()

for i in range(10):
    pruning_a = torch.index_select(a,1,idx)
    
start.record()
for i in range(loops):
    pruning_a = torch.index_select(a,1,idx)
end.record()
torch.cuda.synchronize()
time1=start.elapsed_time(end)/loops

print("mask:",time_mask)
print("time0:",time0)
print("time1:",time1)









