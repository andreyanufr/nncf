import torch
import sys
import torch
import torch.nn as nn
from codebook import q_vectors_256, get_initial_array
from codebook import get_packed_abs_grid_4096
from utils import pairwise_dist, pairwise_attn, table_rectification_fast, table_rectification

q_vectors_4096 = get_packed_abs_grid_4096()
q_vectors_init = get_initial_array()

def get_signed_groups(weight: torch.Tensor):
    assert len(weight.shape) == 2
    assert weight.shape[1] == 8
    
    res = torch.zeros(weight.shape[0]).long().to(weight.device)
    
    
    for i in range(8):
        s = weight[:, i] < 0
        res[:] = res[:] | (s.long() << i)
    
    return res


def to_sign(val):
    res = torch.ones(8).long()
    for i in range(8):
        res[i] -= ((val & (1 << i)) > 0) * 2
    
    return res 



def test_signed_groups():
    x = torch.tensor([[0, -1, 2, -3, 4, -5, 6, -7], [-0, 1, -2, 3, -4, 5, -6, 7], [1, 2, 3, 4, 5, 6, 7, 8], [-1, -2, -3, -4, -5, -6, -7, -8]])
    
    target = torch.tensor([0b10101010, 0b01010100, 0b00000000, 0b11111111])
    
    sg = get_signed_groups(x)
    
    print(sg)
    print(target)

def test_to_sign():
    vals = torch.tensor([0b10101010, 0b01010100, 0b00000000, 0b11111111])
    x = torch.tensor([[0, -1, 2, -3, 4, -5, 6, -7], [-0, 1, -2, 3, -4, 5, -6, 7], [1, 2, 3, 4, 5, 6, 7, 8], [-1, -2, -3, -4, -5, -6, -7, -8]])
    
    target = torch.sign(x)
    
    sq = [to_sign(val) for val in vals]
    
    print(target)
    print(sq)


def get_codebook_attn(weight: torch.Tensor, sign, targe_sz, rectification_iterations=10):
    qv = q_vectors_4096.clone().to(weight.device)
    qv = torch.cat([qv, qv // 2, qv // 3])
    q_max = qv.max()

    sign = to_sign(sign).to(weight.device)
    weight = q_max * weight.abs().float()
    
    best_alpha = 0.01
    # min_dist = 200000000
    # for alpha in [0.9, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
    #     attn = pairwise_attn(qv.float(), weight, normalize=False, alpha=alpha)
    #     scores = torch.zeros(qv.shape[0])
    
    #     for i in range(qv.shape[0]):
    #         scores[i] = torch.sum(attn[:, i])
        
    #     top_idxs = torch.topk(scores, targe_sz)[1]    
    #     qv_cur = qv[top_idxs, :]
        
    #     idxs = pairwise_dist(qv_cur.float(), weight, normalize=False)
        
    #     qw = qv_cur[idxs, :]
    #     dists = torch.mean((qw - weight)**2)
    #     if dists < min_dist:
    #         min_dist = dists
    #         best_alpha = alpha
    #     print(alpha, dists.mean())

    attn = pairwise_attn(qv.float(), weight, normalize=False, alpha=best_alpha)

    scores = torch.zeros(qv.shape[0])
    
    for i in range(qv.shape[0]):
        scores[i] = torch.sum(attn[:, i])
    
    top_idxs = torch.topk(scores, targe_sz)[1]    
    qv = qv[top_idxs, :]

    for r_i in range(rectification_iterations):
        idxs = pairwise_dist(qv.float(), weight, normalize=False)
        qv_new = table_rectification_fast(qv, weight, idxs)
        #qv_new = table_rectification(qv, weight, idxs)
        qv = torch.round(torch.clamp(qv_new, 0, q_max))
    qv = qv * sign.unsqueeze(0)
    return idxs, qv


def get_codebook(weight: torch.Tensor, sign, targe_sz, rectification_iterations=10):
    qv = q_vectors_4096.clone()[:targe_sz, :].to(weight.device)
    q_max = qv.max()
    
    sign = to_sign(sign).to(weight.device)
    weight = q_max * weight.abs().float()

    for r_i in range(rectification_iterations):
        idxs = pairwise_dist(qv.float(), weight, normalize=False)
        qv_new = table_rectification_fast(qv, weight, idxs)
        qv = torch.round(torch.clamp(qv_new, 0, q_max))
    qv = qv * sign.unsqueeze(0)
    return idxs, qv
    
def compress_by_signed_notebook(weight: torch.Tensor, target_sz=2**14):
    out_ch, in_ch = weight.shape
    super_group_size = 256
    group_size = 8
    n_sub_groups = super_group_size // group_size
    assert out_ch % super_group_size == 0
    
    gweight = weight.reshape(out_ch, -1, super_group_size)
    
    super_scale = gweight.abs().max(dim=-1, keepdim=True)[0]
    mask = (gweight >= 0).long()
    mask = torch.sum(mask, dim=-1, keepdim=True)
    super_scale[torch.where(mask < super_group_size // 2)] *= -1
    gweight = gweight / super_scale
    
    super_group_shape = gweight.shape

    gweight = gweight.reshape(-1, group_size)
    
    sg = get_signed_groups(gweight)
    sg_weight = torch.zeros(256)

    for i in range(256):
        sg_weight[i] = torch.count_nonzero(sg == i)
        #print(i, sg_weight[i])
        
    sg_weight /= torch.sum(sg_weight)
    n_per_sg_group = torch.round(target_sz * sg_weight)
    
    codebook = []
    idxs = torch.zeros(gweight.shape[0]).long().to(weight.device)
    offset = 0
    for i in range(256):
        i_idxs = torch.where(sg == i)[0]
        gr_weights = gweight[i_idxs, :]
        gr_idxs, gr_codebook = get_codebook_attn(gr_weights, i, int(n_per_sg_group[i].item()))
        idxs[i_idxs] = gr_idxs + offset
        codebook.append(gr_codebook)
        offset += gr_codebook.shape[0]
    
    codebook = torch.cat(codebook)

    super_scale = super_scale / 127
    qweights = codebook[idxs, :]
    qweights = qweights.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1)
    qweights = qweights.reshape(super_group_shape[0], super_group_shape[1], -1)
    qweights = qweights * super_scale
    qweights = qweights.reshape(weight.shape)

    abs_err = torch.mean((weight - qweights)**2)
    rel_err = abs_err / torch.mean(weight**2)
    
    print(f"Abs err {abs_err}, rel_err: {rel_err}")
    #print(torch.sum(n_per_sg_group))
    
    return weight

if __name__ == "__main__":
    test_signed_groups()
    test_to_sign()
    
