import torch
import numpy as np

anumpy = np.arange(1,37)
a = torch.from_numpy(anumpy).cuda().reshape(-1,2)
a_N = a.reshape(2, -1, a.shape[-1])#N=2
a_N_t = a_N.permute(0, 2, 1)
a_N_t_reshape = a_N_t.reshape(2,2,3,3)#[batch_size,channel,out_g,out_w]
print(a_N_t_reshape)