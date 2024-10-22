import torch
import sys
import torch
import torch.nn as nn
from codebook import q_vectors_256, get_initial_array
from codebook import get_packed_abs_grid_4096
from utils import pairwise_dist, pairwise_attn, table_rectification_fast, table_rectification
from faiss.contrib import clustering

q_vectors_4096 = get_packed_abs_grid_4096()
q_vectors_init = get_initial_array()

def get_signed_groups(weight: torch.Tensor):
    assert len(weight.shape) == 2
    #assert weight.shape[1] == 8
    
    sz = weight.shape[1]
    assert sz <= 32
    
    res = torch.zeros(weight.shape[0]).long().to(weight.device)
    
    
    for i in range(sz):
        s = weight[:, i] < 0
        res[:] = res[:] | (s.long() << i)
    
    return res


def to_sign(val, bits=8):
    sz = 32
    max_sz = 8
    res = torch.ones(sz).long()
    for i in range(sz):
        if val & (1 << i):
            res[i] -= ((val & (1 << i)) > 0) * 2
            max_sz = max(i, max_sz)
    res = res[:bits]
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


def get_codebook_attn(weight: torch.Tensor, sign, targe_sz, rectification_iterations=20):
    qv = q_vectors_4096.clone().to(weight.device)
    #qv = torch.round(qv / qv.max() * 127)
    #qv = torch.cat([qv, qv // 2, qv // 4, qv // 8])
    qv = qv[:, :weight.shape[-1]]
    q_max = qv.max()

    sign = to_sign(sign, weight.shape[-1]).to(weight.device)
    weight = q_max * weight.abs().float()
    
    best_alpha = 0.01

    attn = pairwise_attn(qv.float(), weight, normalize=False, alpha=best_alpha)
    scores = torch.sum(attn, dim=0)
    while 2 * targe_sz < scores.shape[0] and False:
        top_idxs = torch.topk(scores, scores.shape[0] // 2)[1]
        qv = qv[top_idxs, :]
        attn = pairwise_attn(qv.float(), weight, normalize=False, alpha=best_alpha)
        scores = torch.sum(attn, dim=0) 
    top_idxs = torch.topk(scores, targe_sz)[1]
    qv = qv[top_idxs, :]

    idxs = pairwise_dist(qv.float(), weight, normalize=False)
    for r_i in range(rectification_iterations):
        qv_new = table_rectification_fast(qv, weight, idxs)
        #qv_new = table_rectification(qv, weight, idxs)
        qv = torch.round(torch.clamp(qv_new, 0, q_max))
        if r_i + 1 < rectification_iterations:
            idxs = pairwise_dist(qv.float(), weight, normalize=False)
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


def get_kmeans_codebook(weight: torch.Tensor, target_sz):
    x = weight * 127
    # cluster_ids_x, cluster_centers = kmeans(
    #     X=x, num_clusters=target_sz, distance='euclidean', device=torch.device('cuda:2')
    # )
    
    # kmeans = faiss.Kmeans(x.cpu().numpy(), k=target_sz, niter=500, nredo=10)
    # kmeans.train(x)
    # distances, cluster_ids = kmeans.index.search(x, 1)
    da = clustering.DatasetAssignGPU(x.cpu(), 1)
    centroids = clustering.kmeans(
            target_sz, da, niter=50, verbose=False)
    qv = torch.from_numpy(centroids).to(weight.device)

    idxs = pairwise_dist(qv, x, normalize=False)
    
    return idxs, torch.round(qv)
    
    
def compress_by_signed_notebook(weight: torch.Tensor, super_group_size = 256,
                                group_size = 8, target_sz=2**14):
    out_ch, in_ch = weight.shape
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
    
    n_signs = 2**group_size
    sg = get_signed_groups(gweight)
    sg_weight = torch.zeros(n_signs)

    for i in range(n_signs):
        sg_weight[i] = torch.count_nonzero(sg == i)
        #print(i, sg_weight[i])
        
    sg_weight /= torch.sum(sg_weight)
    n_per_sg_group = torch.round(target_sz * sg_weight)
    
    codebook = []
    idxs = torch.zeros(gweight.shape[0]).long().to(weight.device)
    offset = 0
    for i in range(n_signs):
        i_idxs = torch.where(sg == i)[0]
        gr_weights = gweight[i_idxs, :]
        #gr_idxs, gr_codebook = get_codebook_attn(gr_weights, i, int(n_per_sg_group[i].item()))
        gr_idxs, gr_codebook = get_kmeans_codebook(gr_weights, int(n_per_sg_group[i].item()))
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
    return qweights


def compress_by_signed_notebook_group_wise(weight: torch.Tensor, super_group_size = 256,
                                group_size = 8, target_sz=2**14):
    out_ch, in_ch = weight.shape
    n_sub_groups = super_group_size // group_size
    assert out_ch % super_group_size == 0
    
    gweight = weight.reshape(out_ch, -1, super_group_size)
    
    super_scale = gweight.abs().max(dim=-1, keepdim=True)[0]
    mask = (gweight >= 0).long()
    mask = torch.sum(mask, dim=-1, keepdim=True)
    super_scale[torch.where(mask < super_group_size // 2)] *= -1
    gweight = gweight / super_scale
    
    super_group_shape = gweight.shape
    #gweight = gweight.reshape(super_group_shape[0], super_group_shape[1], -1, group_size)
    
    n_signs = 2**group_size
    group_codebook = []
    group_idxs = []

    for i_g in range(gweight.shape[1]):
        i_gweights = gweight[:, i_g, :]
        i_gweights = i_gweights.reshape(-1, group_size)
        sg = get_signed_groups(i_gweights)
        sg_weight = torch.zeros(n_signs)

        for i in range(n_signs):
            sg_weight[i] = torch.count_nonzero(sg == i)
            #print(i, sg_weight[i])
            
        sg_weight /= torch.sum(sg_weight)
        n_per_sg_group = torch.round(target_sz * sg_weight)
        
        codebook = []
        idxs = torch.zeros(i_gweights.shape[0]).long().to(weight.device)
        offset = 0
        for i in range(n_signs):
            i_idxs = torch.where(sg == i)[0]
            gr_weights = i_gweights[i_idxs, :]
            gr_idxs, gr_codebook = get_codebook_attn(gr_weights, i, int(n_per_sg_group[i].item()))
            idxs[i_idxs] = gr_idxs + offset
            codebook.append(gr_codebook)
            offset += gr_codebook.shape[0]
        
        codebook = torch.cat(codebook)
        
        group_codebook.append(codebook)
        group_idxs.append(idxs)
    
    qweights = []
    for i_g in range(gweight.shape[1]):
        i_qweights = group_codebook[i_g][group_idxs[i_g], :]
        i_qweights = i_qweights.reshape(gweight.shape[0], 1, gweight.shape[2])
        qweights.append(i_qweights)
    qweights = torch.cat(qweights, dim=1)

    super_scale = super_scale / 127
    #qweights = qweights.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1)
    #qweights = qweights.reshape(super_group_shape[0], super_group_shape[1], -1)
    qweights = qweights * super_scale
    qweights = qweights.reshape(weight.shape)

    abs_err = torch.mean((weight - qweights)**2)
    rel_err = abs_err / torch.mean(weight**2)

    print(f"Abs err {abs_err}, rel_err: {rel_err}")
    #print(torch.sum(n_per_sg_group))
    return qweights


if __name__ == "__main__":
    test_signed_groups()
    test_to_sign()
    
