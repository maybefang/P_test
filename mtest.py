import torch
import gemm
import numpy as np
import phello
import time

#test if results are same
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
'''
b = torch.tensor([[[[0.00001,0.00001,0.00001],
                    [0.00001,0.00001,0.00001],
                    [0.00001,0.00001,0.00001]],
                    
                   [[0.00001,0.00001,0.00001],
                    [0.00001,0.00001,0.00001],
                    [0.00001,0.00001,0.00001]],

                   [[0.00001,0.00001,0.00001],
                    [0.00001,0.00001,0.00001],
                    [0.00001,0.00001,0.00001]]],
                    
                  [[[0.0001,0.0001,0.0001],
                    [0.0001,0.0001,0.0001],
                    [0.0001,0.0001,0.0001]],
                    
                   [[0.001,0.001,0.001],
                    [0.001,0.001,0.001],
                    [0.001,0.001,0.001]],
                    
                   [[0.01,0.01,0.01],
                    [0.01,0.01,0.01],
                    [0.01,0.01,0.01]]]]).cuda()#[2,3,3,3]
'''
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

#btest = torch.randn(27,2).cuda()
btest = torch.range(1,6.35,0.1).reshape(27,2).cuda()
#btest = torch.from_numpy(np.arange(648,702).reshape(27,2)).float().cuda()
bias = torch.zeros(1).cuda()

bshape = b.shape
b_tile_t = b.reshape(bshape[0],-1)
b_tile = b_tile_t.t()#[-1,out_c]

N, C, H, W = a.shape
stride = 1
ksize = 3
pad=0
out_h = (H + 2 * pad - ksize) // stride + 1
out_w = (W + 2 * pad - ksize) // stride + 1
strides = (C*H*W,H*W,W*stride,stride,W*stride,stride)


A = torch.as_strided(a, size=(N,C,out_h,out_w,ksize,ksize), stride=strides)
col = A.permute(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w, -1)

#col,b_tile pytorch_linear_out.equal(my_out)=False
#my_out = gemm.mymultiply(col,b_tile,bias)
#pytorch_linear_out = torch.nn.functional.linear(col,b_tile.t())

#col,btest pytorch_linear_out.equal(my_out)=True
#my_out = gemm.mymultiply(col,btest,bias)
#pytorch_linear_out = torch.nn.functional.linear(col,btest.t())

#atest,b_tile pytorch_linear_out.equal(my_out)=False
#my_out = gemm.mymultiply(atest,b_tile,bias)
#pytorch_linear_out = torch.nn.functional.linear(atest,b_tile.t())

#atest,btest pytorch_linear_out.equal(my_out)=True
my_out = gemm.mymultiply(atest,btest,bias)
pytorch_linear_out = torch.nn.functional.linear(atest,btest.t())

#print(my_out)
#print()
#print(pytorch_linear_out)
#print()
print(pytorch_linear_out.equal(my_out))






























