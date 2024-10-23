import sys
import torch
import torch.nn as nn
from codebook import q_vectors_256
from codebook import get_packed_abs_grid_4096
from scheme_signed_codebook import compress_by_signed_notebook, compress_by_signed_notebook_group_wise, pairwise_attn
from utils import pairwise_dist, table_rectification_fast

from nncf.quantization.algorithms.weight_compression import weight_lowering

q_vectors_4096 = get_packed_abs_grid_4096()
prev = None

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


def compress_decompress_int4(weight):
    config = weight_lowering.WeightCompressionConfig()
    config.group_size = 64
    config.mode = weight_lowering.CompressWeightsMode.INT4_ASYM
    compressed_weights = weight_lowering.compress_weight(weight, 1, config)
    dweight = weight_lowering.do_int_dequantization(compressed_weights.tensor, compressed_weights.scale,
                                                    compressed_weights.zero_point, 1)
    #weight_lowering.do_int_quantization()

    return dweight.data

# def table_rectification(table : torch.Tensor, gt, indexes):
#     max_t = torch.max(table)
#     max_gt = torch.max(gt)
    
#     gt = gt / max_gt * max_t
    
#     qw = table[indexes, :]
#     dists = torch.mean((qw - gt)**2, dim=-1)
    
#     print("Before: ", dists.mean())

#     new_table = table.clone()

#     # iteration over number of vectors
#     for i in range(table.shape[0]):
#         # if i == 0:
#         #     continue
#         mask = (i == indexes).float()
#         dist_i = dists * mask
#         dist_i = (torch.max(dist_i) - dist_i) * mask
#         dist_denum = torch.sum(dist_i)
        
#         if dist_denum < 0.00001:
#             continue
        
#         dist_i = dist_i / dist_denum

#         denom = torch.sum(mask)
#         if denom < 1.0:
#             continue
#         #new_vec = torch.sum(gt * mask.unsqueeze(-1), dim=(0,1,2)) / denom
#         new_vec = torch.sum(gt * dist_i.unsqueeze(-1), dim=(0,1,2))
#         new_vec = new_vec.squeeze()
#         #max_t = torch.max(table[i, :])
#         #new_vec = new_vec / torch.max(new_vec) * max_t
#         new_vec = torch.round(torch.clamp(new_vec, min=0.0, max=255.0))
#         #new_vec[torch.where(table[i, :] == 43)] = 43
#         new_table[i, :] = new_vec

#     qw = new_table[indexes, :]
#     dists = torch.mean((qw - gt)**2, dim=-1)
#     print("After: ", dists.mean())
 
#     return new_table


def table_rectification(table : torch.Tensor, gt, indexes):    
    qw = table[indexes, :]
    dists = torch.mean((qw - gt)**2, dim=-1)
    
    print("Before: ", dists.mean())

    new_table = table.clone()
    sum_dim = [0]
    for i in range(1, len(gt.shape) - 1):
        sum_dim.append(i)
    sum_dim = tuple(sum_dim)

    # iteration over number of vectors
    for i in range(table.shape[0]):
        # if i == 0:
        #     continue
        mask = (i == indexes).float()
        dist_i = dists * mask
        dist_i = (torch.max(dist_i) - dist_i) * mask
        dist_denum = torch.sum(dist_i)
        
        if dist_denum < 0.00001:
            continue
        
        dist_i = dist_i / dist_denum

        denom = torch.sum(mask)
        if denom < 1.0:
            continue
        new_vec = torch.sum(gt * dist_i.unsqueeze(-1), dim=sum_dim)
        new_vec = new_vec.squeeze()
        #max_t = torch.max(table[i, :])
        new_vec = new_vec / torch.max(new_vec) * table[i, :].max()
        new_vec = torch.round(torch.clamp(new_vec, min=0.0, max=255.0))
        #new_vec[torch.where(table[i, :] == 43)] = 43
        new_table[i, :] = new_vec

    qw = new_table[indexes, :]
    dists = torch.mean((qw - gt)**2, dim=-1)
    print("After: ", dists.mean())
 
    return new_table



def table_rectification_group(table : torch.Tensor, gt, indexes):
    max_t = torch.max(table)
    max_gt = torch.max(gt)
    
    gt = gt / max_gt * max_t
    
    qw = table[indexes, :]
    dists = torch.mean((qw - gt)**2, dim=-1)
    
    #print("Before: ", dists.mean())

    new_table = table.clone()

    # iteration over number of vectors
    for i in range(table.shape[0]):
        if i == -1:
            continue
        mask = (i == indexes).float()
        dist_i = dists * mask
        dist_i = (torch.max(dist_i) - dist_i) * mask
        dist_denum = torch.sum(dist_i)
        
        if dist_denum < 0.00001:
            continue
        
        dist_i = dist_i / dist_denum

        denom = torch.sum(mask)
        if denom < 1.0:
            continue
        #new_vec = torch.sum(gt * mask.unsqueeze(-1), dim=(0,1,2)) / denom
        new_vec = torch.sum(gt * dist_i.unsqueeze(-1), dim=(0,1))
        new_vec = new_vec.squeeze()
        #max_t = torch.max(table[i, :])
        new_vec = new_vec / torch.max(new_vec) * max_t
        new_vec = torch.round(torch.clamp(new_vec, min=0.0, max=255.0))
        #new_vec[torch.where(table[i, :] == 43)] = 43
        new_table[i, :] = new_vec

    # qw = new_table[indexes, :]
    # dists = torch.mean((qw - gt)**2, dim=-1)
    
    # print("After: ", dists.mean())
 
    return new_table



def normalize(w, scale, max_q):
    q = torch.clamp(scale * w, min=0.0, max=max_q)
    return q


def compress_decompress_fixed(weight: torch.Tensor):
    out_ch, in_ch = weight.shape
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

    qv = q_vectors_256.clone()
    for i in range(qv.shape[0]):
        v = qv[i, :]
        if torch.max(v) != 43:
            v = torch.round(v / torch.max(v) * 43)
            qv[i, :] = v

    for i in range(n_sub_groups):
        cur_w = gweight[..., i*group_size:(i+1)*group_size]
        cur_s = cur_w.max(dim=-1, keepdim=True)[0]
        mask = cur_s < torch.finfo(torch.float32).eps
        mask = mask.float()
        tmp = torch.count_nonzero(mask)
        #print(tmp)
        
        cur_i_s = q_max / (cur_s + mask)#(2 * k_max  - 1) / (cur_s + mask)
        q = normalize(cur_w, cur_i_s, q_max)

        origin_shape = q.shape
        idxs = pairwise_dist(qv.float(), q.reshape(q.shape[0]*q.shape[1],-1).float())
        idxs = idxs.reshape(origin_shape[0], origin_shape[1])
        group_idxs.append(idxs)

        cur_s = torch.round(torch.clamp(sub_scale_max * cur_s, min=0, max=sub_scale_max)) # 4bit scale

        group_scales.append(cur_s)

    group_idxs = torch.stack(group_idxs, dim=-1)    
    group_scales = torch.stack(group_scales, dim=2)

    super_scale = super_scale / q_max
    super_scale = super_scale / sub_scale_max
    
    qw = qv[group_idxs, :]
    qw = qw * group_scales
    qw = qw.reshape(qw.shape[0], qw.shape[1], -1)
    qw = qw * super_scale
    qw = qw * signum
    qw = qw.reshape(weight.shape)
    
    
    abs_err = torch.mean((weight-qw)**2)
    rel_err = abs_err / torch.mean(weight**2)
    
    print(f"Abs err {abs_err}, rel_err: {rel_err}")

    return qw 


def compress_decompress_fixed_4096(weight: torch.Tensor):
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

    group_idxs = []

    qv = q_vectors_4096.clone()
    q_max = qv.max()

    for i in range(n_sub_groups):
        cur_w = gweight[..., i*group_size:(i+1)*group_size]
        
        q = normalize(cur_w, q_max, q_max)
        origin_shape = q.shape
        idxs = pairwise_dist(qv.float(), q.reshape(q.shape[0]*q.shape[1],-1).float(), normalize=False)
        idxs = idxs.reshape(origin_shape[0], origin_shape[1])
        group_idxs.append(idxs)


    group_idxs = torch.stack(group_idxs, dim=-1)    
    super_scale = super_scale / q_max
    
    qw = qv[group_idxs, :]
    qw = qw.reshape(qw.shape[0], qw.shape[1], -1)
    qw = qw * super_scale
    qw = qw * signum
    qw = qw.reshape(weight.shape)


    abs_err = torch.mean((weight-qw)**2)
    rel_err = abs_err / torch.mean(weight**2)
    
    print(f"Abs err {abs_err}, rel_err: {rel_err}")

    return qw


def compress_decompress_vq_codebook_per_tensor_4096(weight: torch.Tensor, super_group_size=256,
                                                    group_size=8, table_sz=2**12):    
    out_ch, in_ch = weight.shape
    n_sub_groups = super_group_size // group_size
    assert out_ch % super_group_size == 0
    
    gweight = weight.reshape(out_ch, -1, super_group_size)
    
    signum = gweight.sign()
    gweight = gweight.abs()
    
    super_scale = gweight.max(dim=-1, keepdim=True)[0]
    gweight = gweight / super_scale

    n_iters = 20

    qv = q_vectors_4096.clone().to(weight.device)[:, :group_size]
    q_max = 127 #qv.max()

    qs = []

    for n_i in range(n_iters):
        group_idxs = []
        for i in range(n_sub_groups):
            if n_i == 0:
                cur_w = gweight[..., i*group_size:(i+1)*group_size]
                q = normalize(cur_w, q_max, q_max)
                qs.append(q)
            else:
                q = qs[i]
            origin_shape = q.shape
            idxs = pairwise_dist(qv.float(), q.reshape(q.shape[0]*q.shape[1],-1).float(), normalize=False)
            idxs = idxs.reshape(origin_shape[0], origin_shape[1])
            group_idxs.append(idxs)

        if n_i == 0:
            best_alpha = 0.01
            scores = 0
            for i in range(n_sub_groups):
                qi = qs[i].reshape(q.shape[0]*q.shape[1],-1)
                attn = pairwise_attn(qv.float(), qi[:qi.shape[0] // 2, :], normalize=False, alpha=best_alpha)
                scores += attn
                attn = pairwise_attn(qv.float(), qi[qi.shape[0] // 2:, :], normalize=False, alpha=best_alpha)
                scores += attn
            scores = torch.sum(attn, dim=0)
            top_idxs = torch.topk(torch.tensor(scores), table_sz)[1]
            qv = qv[top_idxs, :]
            group_idxs = []
            for i in range(n_sub_groups):
                idxs = pairwise_dist(qv.float(), qs[i].reshape(q.shape[0]*q.shape[1],-1).float(), normalize=False)    
                idxs = idxs.reshape(origin_shape[0], origin_shape[1])
                group_idxs.append(idxs)

        if n_i + 1 != n_iters:
            groups_idxs = torch.stack(group_idxs, dim=-1) # [32][4096, 16]
            qss = torch.stack(qs, dim=-2) # [32][4096, 16, 8]
            qv_new = table_rectification_fast(qv, qss, groups_idxs)
            qv = torch.round(torch.clamp(qv_new, 0, q_max))
            #q_max = torch.max(qv)
            #group_idxs = []

            # for i in range(n_sub_groups):
            #     q = qs[i]
            #     origin_shape = q.shape
            #     idxs_dist = pairwise_dist(qv.float(), q.reshape(q.shape[0]*q.shape[1],-1).float(), normalize=False)
            #     idxs_dist = idxs_dist.reshape(origin_shape[0], origin_shape[1])
            #     idxs = idxs_dist
            #     group_idxs.append(idxs)

    #print(torch.unique(group_idxs))
    group_idxs = torch.stack(group_idxs, dim=-1)
    qw = qv[group_idxs, :]
    qw = qw.reshape(qw.shape[0], qw.shape[1], -1).float()

    # num = torch.sum(torch.mul(qw, gweight * super_scale), dim=-1, keepdim=True)
    # denum = torch.sum(torch.mul(qw, qw), dim=-1, keepdim=True)
    # super_scale_ = num / denum
    
    super_scale = super_scale / q_max
    qw = qw * super_scale
    qw = qw * signum
    qw = qw.reshape(weight.shape)

    abs_err = torch.mean((weight-qw)**2)
    rel_err = abs_err / torch.mean(weight**2)
    
    print(f"Abs err {abs_err}, rel_err: {rel_err}")
    sys.stdout.flush()
    # w = s * wq
    # w8 = s8 * wq8 => w = s * (s8 * wq8), wq8 {8, 25, 43}

    return qw


def compress_decompress_vq_codebook_per_tensor_256(weight: torch.Tensor):    
    out_ch, in_ch = weight.shape
    super_group_size = 256
    group_size = 8
    n_sub_groups = super_group_size // group_size
    assert out_ch % super_group_size == 0
    
    gweight = weight.reshape(out_ch, -1, super_group_size)
    
    signum = gweight.sign()
    gweight = gweight.abs()
    
    super_scale = gweight.max(dim=-1, keepdim=True)[0]
    gweight = gweight / super_scale

    q_max = 18
    sub_scale_max = 7

    n_iters = 20

    qv = q_vectors_256.clone()
    for i in range(qv.shape[0]):
        v = qv[i, :]
        if torch.max(v) != 43 or 1:
            v = torch.round(v / torch.max(v) * q_max)
            qv[i, :] = v
    
    qs = []
    group_scales = []

    for n_i in range(n_iters):
        group_idxs = []

        for i in range(n_sub_groups):
            if n_i == 0:
                cur_w = gweight[..., i*group_size:(i+1)*group_size]
                cur_s = cur_w.max(dim=-1, keepdim=True)[0]
                mask = cur_s < torch.finfo(torch.float32).eps
                mask = mask.float()
                cur_i_s = q_max / (cur_s + mask)#(2 * k_max  - 1) / (cur_s + mask)
                q = normalize(cur_w, cur_i_s, q_max)

                origin_shape = q.shape
                qs.append(q)
                cur_s = torch.round(torch.clamp(sub_scale_max * cur_s, min=0, max=sub_scale_max)) # 4bit scale
                group_scales.append(cur_s)
            else:
                q = qs[i]
                origin_shape = q.shape
            idxs = pairwise_dist(qv.float(), q.reshape(q.shape[0]*q.shape[1],-1).float())
            idxs = idxs.reshape(origin_shape[0], origin_shape[1])
            group_idxs.append(idxs)


        if n_i + 1 != n_iters:
            groups_idxs = torch.stack(group_idxs, dim=-1)
            qss = torch.stack(qs, dim=-2)
            #qv_new = table_rectification(qv, qss, groups_idxs)
            qv_new = table_rectification_fast(qv, qss, groups_idxs)
            qv = torch.round(torch.clamp(qv_new, 0, q_max))
            q_max = torch.max(qv)
            group_idxs = []

            for i in range(n_sub_groups):
                q = qs[i]
                origin_shape = q.shape
                idxs_dist = pairwise_dist(qv.float(), q.reshape(q.shape[0]*q.shape[1],-1).float())
                idxs_dist = idxs_dist.reshape(origin_shape[0], origin_shape[1])
                idxs = idxs_dist
                group_idxs.append(idxs)

    if True:
        group_scales = []
        for i in range(n_sub_groups):
            cur_w = gweight[..., i*group_size:(i+1)*group_size]
            cur_s = cur_w.max(dim=-1, keepdim=True)[0]
            
            cur_qw = qv[group_idxs[i], :]
            cur_qs = cur_qw.max(dim=-1, keepdim=True)[0]
            
            cur_s = cur_s * q_max / cur_qs.float()
            cur_s = torch.round(torch.clamp(sub_scale_max * cur_s, min=0, max=sub_scale_max)) # 4bit scale
            group_scales.append(cur_s)
            
                
    group_idxs = torch.stack(group_idxs, dim=-1)    
    group_scales = torch.stack(group_scales, dim=2)

    super_scale = super_scale / q_max
    super_scale = super_scale / sub_scale_max
    
    #qw = q_vectors_256[group_idxs, :]
    qw = qv[group_idxs, :]
    qw = qw.long() * group_scales.long()
    qw = qw.reshape(qw.shape[0], qw.shape[1], -1).float()
    qw = qw * super_scale
    qw = qw * signum
    qw = qw.reshape(weight.shape)
    

    abs_err = torch.mean((weight-qw)**2)
    rel_err = abs_err / torch.mean(weight**2)
    
    print(f"Abs err {abs_err}, rel_err: {rel_err}")

    return qw


def compress_decompress_vq(weight: torch.Tensor, rectify=True):
    #return compress_by_signed_notebook(weight)
    #return compress_by_signed_notebook_group_wise(weight, 128, 8, 2**10)
    #return compress_by_signed_notebook(weight, 64, 4, 2**8)
    return compress_by_signed_notebook_group_wise(weight, 64, 4, 2**8)
    if not rectify:
        #return compress_decompress_fixed_4096(weight)
        return compress_decompress_fixed(weight)

    #return compress_decompress_vq_codebook_per_tensor_256(weight)
    #return compress_decompress_vq_codebook_per_tensor_4096(weight, 64, 4, 16)
    return compress_decompress_vq_tables(weight)
    
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

    q_max = 18
    sub_scale_max = 7

    n_iters = 20

    qv = q_vectors_256.clone()
    for i in range(qv.shape[0]):
        v = qv[i, :]
        if torch.max(v) != 43 or 1:
            v = torch.round(v / torch.max(v) * q_max)
            qv[i, :] = v
    
    qs = []
    group_scales = []

    for n_i in range(n_iters):
        group_idxs = []


        for i in range(n_sub_groups):
            if n_i == 0:
                cur_w = gweight[..., i*group_size:(i+1)*group_size]
                cur_s = cur_w.max(dim=-1, keepdim=True)[0]
                mask = cur_s < torch.finfo(torch.float32).eps
                mask = mask.float()
                cur_i_s = q_max / (cur_s + mask)#(2 * k_max  - 1) / (cur_s + mask)
                q = normalize(cur_w, cur_i_s, q_max)

                origin_shape = q.shape
                idxs = pairwise_dist(qv.float(), q.reshape(q.shape[0]*q.shape[1],-1).float())
                idxs = idxs.reshape(origin_shape[0], origin_shape[1])
                group_idxs.append(idxs)
                qs.append(q)
                cur_s = torch.round(torch.clamp(sub_scale_max * cur_s, min=0, max=sub_scale_max)) # 4bit scale
                group_scales.append(cur_s)
            else:
                q = qs[i]
                origin_shape = q.shape
                idxs_dist = pairwise_dist(qv.float(), q.reshape(q.shape[0]*q.shape[1],-1).float())
                #idxs_dist = pairwise_dist(group_tables[i].float(), q.reshape(q.shape[0]*q.shape[1],-1).float())
                idxs_dist = idxs_dist.reshape(origin_shape[0], origin_shape[1])
                idxs = idxs_dist #torch.where(idxs > -1, idxs, idxs_dist)
                group_idxs.append(idxs)


        if n_i + 1 != n_iters:
            groups_idxs = torch.stack(group_idxs, dim=-1)
            qss = torch.stack(qs, dim=-2)
            qv_new = table_rectification(qv, qss, groups_idxs)
            qv = torch.round(torch.clamp(qv_new, 0, q_max))
            q_max = torch.max(qv)
            group_idxs = []

            for i in range(n_sub_groups):
                q = qs[i]
                origin_shape = q.shape
                idxs_dist = pairwise_dist(qv.float(), q.reshape(q.shape[0]*q.shape[1],-1).float())
                idxs_dist = idxs_dist.reshape(origin_shape[0], origin_shape[1])
                idxs = idxs_dist
                group_idxs.append(idxs)

    group_idxs = torch.stack(group_idxs, dim=-1)    
    group_scales = torch.stack(group_scales, dim=2)

    super_scale = super_scale / q_max
    super_scale = super_scale / sub_scale_max
    
    #qw = q_vectors_256[group_idxs, :]
    qw = qv[group_idxs, :]
    qw = qw.long() * group_scales.long()
    qw = qw.reshape(qw.shape[0], qw.shape[1], -1).float()
    qw = qw * super_scale
    qw = qw * signum
    qw = qw.reshape(weight.shape)
    

    abs_err = torch.mean((weight-qw)**2)
    rel_err = abs_err / torch.mean(weight**2)
    
    print(f"Abs err {abs_err}, rel_err: {rel_err}")
    
    # w = s * wq
    # w8 = s8 * wq8 => w = s * (s8 * wq8), wq8 {8, 25, 43}

    return qw


def compress_decompress_vq_tables(weight: torch.Tensor):
    out_ch, out_in = weight.shape
    super_group_size = 256
    group_size = 8
    n_sub_groups = super_group_size // group_size
    assert out_ch % super_group_size == 0
    
    gweight = weight.reshape(out_ch, -1, super_group_size)
    
    n_codebooks = gweight.shape[1]
    
    signum = gweight.sign()
    gweight = gweight.abs()
    
    super_scale = gweight.max(dim=-1, keepdim=True)[0]
    gweight = gweight / super_scale

    k_max = 3
    q_max = 18 # 43 #8 #255 #43
    sub_scale_max = 7

    n_iters = 20 #20

    qv = q_vectors_256.clone()
    for i in range(qv.shape[0]):
        v = qv[i, :]
        if torch.max(v) > q_max or 1:
            v = torch.round(v / torch.max(v) * q_max)
            qv[i, :] = v
    codebooks = [qv.clone() for _ in range(n_codebooks)]

    group_scales = []
    res_idxs = []

    for n_c in  range(n_codebooks):
        qs = []
        for n_i in range(n_iters):
            group_idxs = []
            for i in range(n_sub_groups):
                if n_i == 0:
                    cur_w = gweight[..., n_c, i*group_size:(i+1)*group_size]
                    cur_s = cur_w.max(dim=-1, keepdim=True)[0]
                    mask = cur_s < torch.finfo(torch.float32).eps
                    mask = mask.float()
                    cur_i_s = q_max / (cur_s + mask)#(2 * k_max  - 1) / (cur_s + mask)
                    q = normalize(cur_w, cur_i_s, q_max)

                    origin_shape = q.shape
                    idxs = pairwise_dist(codebooks[n_c].float(), q.float(), normalize=False)
                    #idxs = idxs.reshape(origin_shape[0], 1)
                    group_idxs.append(idxs)
                    qs.append(q)
                    cur_s = torch.round(torch.clamp(sub_scale_max * cur_s, min=0, max=sub_scale_max)) # 4bit scale
                    # if True:
                    #     cur_s = torch.round(15 / cur_s)
                    #     # tmp = cur_s.clone()
                    #     # for i in range(16):
                    #     #     cur_s[tmp == i] = 16 - i
                    #print("cur_s ", cur_s.min(), cur_s.max())
                    group_scales.append(cur_s)
                else:
                    q = qs[i]
                    origin_shape = q.shape
                    idxs = pairwise_dist(codebooks[n_c].float(), q.float(), normalize=False)
                    #idxs = idxs.reshape(origin_shape[0], origin_shape[1])
                    group_idxs.append(idxs)


            if n_i + 1 != n_iters:
                groups_idxs = torch.stack(group_idxs, dim=-1)
                qss = torch.stack(qs, dim=-2)
                qv_new = table_rectification_group(codebooks[n_c], qss, groups_idxs)
                codebooks[n_c] = torch.round(torch.clamp(qv_new, 0, q_max))
                group_idxs = []

                for i in range(n_sub_groups):
                    q = qs[i]
                    origin_shape = q.shape
                    idxs = pairwise_dist(codebooks[n_c].float(), q.float(), normalize=False)
                    group_idxs.append(idxs)
            else:
                res_idxs.append(torch.stack(group_idxs, dim=-1))
    
    super_scale = super_scale / q_max
    super_scale = super_scale / sub_scale_max

    qw = dequantize(super_scale, group_scales, signum, res_idxs, codebooks, weight)
    
    
    abs_err = torch.mean((weight-qw)**2)
    rel_err = abs_err / torch.mean(weight**2)
    
    print(f"Abs err {abs_err}, rel_err: {rel_err}")
    
    # w = s * wq
    # w8 = s8 * wq8 => w = s * (s8 * wq8), wq8 {8, 25, 43}

    return qw    


def dequantize(super_scale, group_scales, signum, idxs, codebooks, weight):
    qw = []
    for gi in range(len(idxs)):
        qw.append(codebooks[gi][idxs[gi], :])

    qw = torch.stack(qw, dim=1)
    group_scales = torch.stack(group_scales, dim=1)
    group_scales = group_scales.reshape(qw.shape[0], qw.shape[1], -1).unsqueeze(-1)
    
    #qw = qw.long() // group_scales.long()
    qw = qw.long() * group_scales.long()
    if qw.max() > 127:
        print("Error in MAX: ", qw.max())
    qw = qw.reshape(qw.shape[0], qw.shape[1], -1).float()
    qw = qw * super_scale
    qw = qw * signum
    qw = qw.reshape(weight.shape)
    
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
    from llama_config import get_llama_3_8b_instruct_config
    bit_config = get_llama_3_8b_instruct_config(model)
    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                weight = layer.weight
                bits = bit_config[name]
                print(name, weight.shape, bits)
                if bits == 2:
                    layer.weight.data[:] = compress_decompress_vq(weight.to("cuda:2")).to('cpu')
                elif bits == 4:
                    layer.weight.data[:] = compress_decompress_int4(weight)
                else:
                    layer.weight.data[:] = compress_decompress_int8(weight)
                # weight = layer.weight
                # if 'k_proj' in name or 'o_proj' in name:
                # # if 'v_proj' in name:
                # #     import numpy as np
                # #     np_name = "/home/aanuf/proj/int4_with_data/fp4/v_proj.npy"
                # #     npw = layer.weight.data.cpu().numpy()
                # #     np.save(np_name, npw)
                #     print(name, layer.weight.shape)
                #     print('VQ')
                #     layer.weight.data[:] = compress_decompress_vq(weight)
                # else:
                #     layer.weight.data[:] = compress_decompress_int8(weight)
                #     #print(name, layer.weight.shape)
                #     #print('int8')
