import torch
import torch.nn as nn
from codebook import init_grid_256
from codebook import q_vectors_256

from nncf.quantization.algorithms.weight_compression import weight_lowering


# 2.5625 bpw
class GroupParams:
    def __init__(self, super_group_size=256, group_size=8):
        assert(super_group_size % group_size == 0)
        self._super_group_size = super_group_size
        self._group_size = group_size

        self._super_scale: float = 1.0
        # must be in range [0, 15] - 4bit per group
        self._group_scale = torch.ones(self._super_group_size / self._group_size, dtype=torch.uint8)
        
        # index in table with size 256
        self._indexes = torch.ones(self._super_group_size / self._group_size, dtype=torch.uint8)
        
        # index in table with size 256
        self._signs = torch.ones(self._super_group_size / self._group_size, dtype=torch.uint8)

# 2.5625 bpw
# [M, N] -> [M, N / 256, 256] -> [M * N / 256, 256]
class TensorParams:
    def __init__(self, sz, super_group_size=256, group_size=8):
        assert(super_group_size % group_size == 0)
        self._super_group_size = super_group_size
        self._group_size = group_size

        self._super_scale = torch.ones(sz, dtype=torch.float16)
        # must be in range [0, 15] - 4bit per group
        self._group_scale = torch.ones((sz, self._super_group_size / self._group_size), dtype=torch.uint8)
        
        # index in table with size 256
        self._indexes = torch.ones((sz, self._super_group_size / self._group_size), dtype=torch.uint8)
        
        # index in table with size 256
        self._signs = torch.ones((sz, self._super_group_size / self._group_size), dtype=torch.uint8)



def compress_decompress_int8(weight):
    config = weight_lowering.WeightCompressionConfig()
    compressed_weights = weight_lowering.compress_weight(weight, 1, config)
    dweight = weight_lowering.do_int_dequantization(compressed_weights.tensor, compressed_weights.scale,
                                                    compressed_weights.zero_point, -1)
    #weight_lowering.do_int_quantization()

    return dweight.data


def pairwise_dist(xyz1, xyz2):
    xyz1 = xyz1 / torch.norm(xyz1, dim=-1, keepdim=True)
    xyz2 = xyz2 / torch.norm(xyz2, dim=-1, keepdim=True)
    
    r_xyz1 = torch.sum(xyz1 * xyz1, dim=1, keepdim=True)  # (B,N,1)
    r_xyz2 = torch.sum(xyz2 * xyz2, dim=1, keepdim=True)  # (B,M,1)
    mul = torch.matmul(xyz2, xyz1.permute(1, 0))         # (B,M,N)
    dist = r_xyz2 - 2 * mul + r_xyz1.permute(1, 0)       # (B,M,N)
    return torch.argmin(dist, dim=-1)


def table_rectification(table : torch.Tensor, gt, indexes):
    max_t = torch.max(table)
    max_gt = torch.max(gt)
    
    gt = gt / max_gt * max_t
    
    new_table = table.clone()

    # iteration over number of vectors
    for i in range(table.shape[0]):
        mask = (i == indexes).float()
        denom = torch.sum(mask)
        if denom < 1.0:
            continue
        new_vec = torch.sum(gt * mask.unsqueeze(-1), dim=(0,1)) / denom
        new_vec = new_vec.squeeze()
        new_vec = new_vec / torch.max(new_vec) * max_t
        new_vec = torch.round(torch.clamp(new_vec, min=0.0, max=255.0))
        new_table[i, :] = new_vec

    return new_table


def compress_decompress_fixed(weight: torch.Tensor):
    out_ch, out_ch = weight.shape
    super_group_size = 256
    group_size = 8
    n_sub_groups = super_group_size // group_size
    assert out_ch % super_group_size == 0
    
    gweight = weight.reshape(out_ch, -1, super_group_size)
    
    signum = gweight.sign()
    gweight = gweight.abs()
    
    super_scale = gweight.max(dim=-1, keepdim=True)[0]
    gweight = gweight / super_scale

    k_max = 3
    q_max = 43 #255 #43
    sub_scale_max = 15

    group_idxs = []
    group_scales = []

    for i in range(n_sub_groups):
        cur_w = gweight[..., i*group_size:(i+1)*group_size]
        cur_s = cur_w.max(dim=-1, keepdim=True)[0]
        mask = cur_s < torch.finfo(torch.float32).eps
        mask = mask.float()
        cur_i_s = (2 * k_max  - 1) / (cur_s + mask)
        
        #q = torch.round(torch.clamp(0.5 * (cur_i_s * cur_w - 1), min=0.0, max=2.0)).to(dtype=torch.uint8)
        q = torch.clamp(0.5 * (cur_i_s * cur_w - 1), min=0.0, max=2.0)
        # u = q[..., 0] << 0
        
        # for i in range(1, 8):
        #     u |= (q[..., i] << 2*i)
        
        # idxs = grid_map[u.long()]

        origin_shape = q.shape
        idxs_dist = pairwise_dist(q_vectors_256.float(), q.reshape(q.shape[0]*q.shape[1],-1).float())
        idxs_dist = idxs_dist.reshape(origin_shape[0], origin_shape[1])
        # if torch.any(idxs) < -1:
        #     print("ERROR")

        idxs = idxs_dist #torch.where(idxs > -1, idxs, idxs_dist)
        
        group_idxs.append(idxs)
 
        # cur_s = cur_s #(1.0 - mask) / (cur_s + mask)
        # cur_s = cur_s / super_scale  # < 1.0

        cur_s = torch.round(torch.clamp(sub_scale_max * cur_s, min=0, max=sub_scale_max)) # 4bit scale

        group_scales.append(cur_s)

    group_idxs = torch.stack(group_idxs, dim=-1)    
    group_scales = torch.stack(group_scales, dim=2)

    super_scale = super_scale / q_max
    super_scale = super_scale / sub_scale_max
    
    qw = q_vectors_256[group_idxs, :]
    qw = qw * group_scales
    qw = qw.reshape(qw.shape[0], qw.shape[1], -1)
    qw = qw * super_scale
    qw = qw * signum
    qw = qw.reshape(weight.shape)
    
    
    abs_err = torch.mean((weight-qw)**2)
    rel_err = abs_err / torch.mean(weight**2)
    
    print(f"Abs err {abs_err}, rel_err: {rel_err}")

    return qw 

def compress_decompress_vq(weight: torch.Tensor, rectify=True):
    if not rectify:
        return compress_decompress_fixed(weight)

    out_ch, out_ch = weight.shape
    super_group_size = 256
    group_size = 8
    n_sub_groups = super_group_size // group_size
    assert out_ch % super_group_size == 0
    
    gweight = weight.reshape(out_ch, -1, super_group_size)
    
    signum = gweight.sign()
    gweight = gweight.abs()
    
    super_scale = gweight.max(dim=-1, keepdim=True)[0]
    gweight = gweight / super_scale
    
    #print(torch.unique(super_scale.flatten()))

    k_max = 3
    q_max = 43 #255 #43
    sub_scale_max = 15
    
    # grid = q_vectors_256
    # grid_map = init_grid_256()

    n_iters = 3
    qv = q_vectors_256.clone()
    
    for n_i in range(n_iters):
        group_idxs = []
        group_scales = []
        rectified_tables = []

        for i in range(n_sub_groups):
            cur_w = gweight[..., i*group_size:(i+1)*group_size]
            cur_s = cur_w.max(dim=-1, keepdim=True)[0]
            mask = cur_s < torch.finfo(torch.float32).eps
            mask = mask.float()
            cur_i_s = (2 * k_max  - 1) / (cur_s + mask)
            
            #q = torch.round(torch.clamp(0.5 * (cur_i_s * cur_w - 1), min=0.0, max=2.0)).to(dtype=torch.uint8)
            q = torch.clamp(0.5 * (cur_i_s * cur_w - 1), min=0.0, max=2.0)
            # u = q[..., 0] << 0
            
            # for i in range(1, 8):
            #     u |= (q[..., i] << 2*i)
            
            # idxs = grid_map[u.long()]

            origin_shape = q.shape
            idxs_dist = pairwise_dist(qv.float(), q.reshape(q.shape[0]*q.shape[1],-1).float())
            idxs_dist = idxs_dist.reshape(origin_shape[0], origin_shape[1])
            # if torch.any(idxs) < -1:
            #     print("ERROR")

            idxs = idxs_dist #torch.where(idxs > -1, idxs, idxs_dist)
            
            group_idxs.append(idxs)
    
            # cur_s = cur_s #(1.0 - mask) / (cur_s + mask)
            # cur_s = cur_s / super_scale  # < 1.0

            cur_s = torch.round(torch.clamp(sub_scale_max * cur_s, min=0, max=sub_scale_max)) # 4bit scale

            group_scales.append(cur_s)
            
            rectified_tables.append(table_rectification(q_vectors_256, q, idxs))
        

        qv_new = torch.round(torch.mean(torch.stack(rectified_tables).float(), dim=0))
        qv = torch.round((3 * qv + qv_new) / 4)
        q_max = torch.max(qv)

        group_idxs = []
        group_scales = []

        for i in range(n_sub_groups):
            cur_w = gweight[..., i*group_size:(i+1)*group_size]
            cur_s = cur_w.max(dim=-1, keepdim=True)[0]
            mask = cur_s < torch.finfo(torch.float32).eps
            mask = mask.float()
            cur_i_s = (2 * k_max  - 1) / (cur_s + mask)
            q = torch.clamp(0.5 * (cur_i_s * cur_w - 1), min=0.0, max=2.0)

            origin_shape = q.shape
            idxs_dist = pairwise_dist(qv.float(), q.reshape(q.shape[0]*q.shape[1],-1).float())
            idxs_dist = idxs_dist.reshape(origin_shape[0], origin_shape[1])
            idxs = idxs_dist
            
            group_idxs.append(idxs)
            cur_s = torch.round(torch.clamp(sub_scale_max * cur_s, min=0, max=sub_scale_max)) # 4bit scale

            group_scales.append(cur_s)

    group_idxs = torch.stack(group_idxs, dim=-1)    
    group_scales = torch.stack(group_scales, dim=2)

    super_scale = super_scale / q_max
    super_scale = super_scale / sub_scale_max
    
    #qw = q_vectors_256[group_idxs, :]
    qw = qv[group_idxs, :]
    qw = qw * group_scales
    qw = qw.reshape(qw.shape[0], qw.shape[1], -1)
    qw = qw * super_scale
    qw = qw * signum
    qw = qw.reshape(weight.shape)
    
    
    abs_err = torch.mean((weight-qw)**2)
    rel_err = abs_err / torch.mean(weight**2)
    
    print(f"Abs err {abs_err}, rel_err: {rel_err}")
    
    # w = s * wq
    # w8 = s8 * wq8 => w = s * (s8 * wq8), wq8 {8, 25, 43}

    return qw    



def compress_phi(model):
    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                weight = layer.weight
                if 'o_proj' in name:
                    print(name)
                    layer.weight.data[:] = compress_decompress_vq(weight)
                else:
                    layer.weight.data[:] = compress_decompress_int8(weight)


def compress_llama(model):
    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                weight = layer.weight
                if 'k_proj' in name or 'o_proj' in name or 
                # if 'v_proj' in name:
                #     import numpy as np
                #     np_name = "/home/aanuf/proj/int4_with_data/fp4/v_proj.npy"
                #     npw = layer.weight.data.cpu().numpy()
                #     np.save(np_name, npw)
                #     print(name)
                #     print('VQ')
                    layer.weight.data[:] = compress_decompress_vq(weight)
                else:
                    layer.weight.data[:] = compress_decompress_int8(weight)
