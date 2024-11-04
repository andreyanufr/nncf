import torch
import sys
import torch
import torch.nn as nn
from codebook import q_vectors_256, get_initial_array
from codebook import get_packed_abs_grid_4096, get_packed_abs_grid_d4
from utils import pairwise_dist, pairwise_attn, table_rectification_fast, table_rectification_fast_weighted
from utils import table_rectification, pairwise_dist_torch, pairwise_attn_torch
from tqdm import tqdm

import numpy as np
from sklearn.cluster import KMeans, MeanShift, Birch
from sklearn.mixture import GaussianMixture

q_vectors_4096 = get_packed_abs_grid_4096()
q_vectors_4096_4_1 = torch.unique(q_vectors_4096[:, :4].clone(), dim=0)
q_vectors_4096_4_2 = torch.unique(q_vectors_4096[:, 4:].clone(), dim=0)
q_vectors_4096_4 = torch.unique(torch.cat([q_vectors_4096_4_1, q_vectors_4096_4_2], dim=0), dim=0)
q_vectors_4096_4 = torch.cat([q_vectors_4096_4, q_vectors_4096_4 // 2, q_vectors_4096_4 // 4, q_vectors_4096_4 // 8])
q_vectors_4096_4 = torch.unique(q_vectors_4096_4, dim=0)

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


def get_codebook_kmeans(weight: torch.Tensor, sign, targe_sz, normalize=True,
                        sample_weight=None):
    q_max = 127 #qv.max()

    if sign is not None:
        sign = to_sign(sign, weight.shape[-1]).to(weight.device)
        weight = weight.abs().float()
    else:
        sign = torch.Tensor([1.0]).to(weight.device)

    if normalize:
        weight = q_max * weight
    weight = weight.round()

    kmeans = KMeans(n_clusters=targe_sz, random_state=0, init='k-means++', n_init="auto", max_iter=1000).fit(weight.cpu().numpy(),
                                                                                               sample_weight=sample_weight)
    #kmeans = GaussianMixture(n_components=targe_sz).fit(weight.cpu().numpy())
    #kmeans = MeanShift().fit(weight.cpu().numpy())
    #kmeans = Birch(n_clusters=targe_sz).fit(weight.cpu().numpy())

    qv = torch.tensor(kmeans.cluster_centers_).to(weight.device)
    qv = torch.round(torch.clamp(qv, 0, q_max))
    qv = qv * sign.unsqueeze(0)

    idxs = torch.tensor(kmeans.labels_).to(weight.device).long()

    return idxs, qv


def get_codebook_attn(weight: torch.Tensor, sign, targe_sz, rectification_iterations=30, normalize=True):
    if weight.shape[-1] == 8:
        qv = q_vectors_4096.clone().to(weight.device)
    else:
        qv = get_packed_abs_grid_d4().to(weight.device)
        qv = torch.round(qv / qv.max() * 127)
        #qv = q_vectors_4096_4.clone().to(weight.device)
        # qv = torch.cat([qv, qv // 2, qv // 4, qv // 8])
        # qv = torch.unique(qv, dim=0)
    #qv = torch.round(qv / qv.max() * 127)
    #qv = torch.cat([qv, qv // 2, qv // 4, qv // 8])
    #qv = qv[:, :weight.shape[-1]]
    q_max = 127 #qv.max()

    sign = to_sign(sign, weight.shape[-1]).to(weight.device)
    weight = weight.abs().float()
    if normalize:
        weight = q_max * weight
    
    best_alpha = 0.01
    iter_shape = weight.shape[0]#qv.shape[0]
    step = iter_shape

    # attn = pairwise_attn(qv.float(), weight[:iter_shape//2, :], normalize=False, alpha=best_alpha)
    # scores = torch.sum(attn, dim=0)

    # attn = pairwise_attn(qv.float(), weight[iter_shape//2:, :], normalize=False, alpha=best_alpha)
    # scores += torch.sum(attn, dim=0)
    
    if True:
        attn = pairwise_attn(qv.float(), weight[:iter_shape//2, :], alpha=best_alpha)
        scores = torch.sum(attn, dim=0)
        attn = pairwise_attn(qv.float(), weight[iter_shape//2:, :], alpha=best_alpha)
        scores += torch.sum(attn, dim=0)
    else:
        scores = torch.zeros(qv.shape[0]).to(weight.device)
        idxs = pairwise_dist(qv.float(), weight[:iter_shape//2, :])
        for i in range(qv.shape[0]):
            scores[i] += torch.count_nonzero(idxs == i)
        idxs = pairwise_dist(qv.float(), weight[iter_shape//2:, :])
        for i in range(qv.shape[0]):
            scores[i] += torch.count_nonzero(idxs == i)

    # while iter_shape < weight.shape[0]:
    #     if iter_shape + 2 * step < weight.shape[0]:
    #         attn = pairwise_attn(qv.float(), weight[iter_shape:iter_shape+step, :], normalize=False, alpha=best_alpha)
    #     else:
    #         attn = pairwise_attn(qv.float(), weight[iter_shape:, :], normalize=False, alpha=best_alpha)
    #         iter_shape += step
    #     scores += torch.sum(attn, dim=0)
    #     iter_shape += step
    
    # while 2 * targe_sz < scores.shape[0] and False:
    #     top_idxs = torch.topk(scores, scores.shape[0] // 2)[1]
    #     qv = qv[top_idxs, :]
    #     attn = pairwise_attn(qv.float(), weight, normalize=False, alpha=best_alpha)
    #     scores = torch.sum(attn, dim=0) 
    top_idxs = torch.topk(scores, targe_sz)[1]
    qv = qv[top_idxs, :]

    idxs = pairwise_dist(qv.float(), weight)
    prev_dist = 0.0
    for r_i in range(rectification_iterations):
        qv_new, cur_dist = table_rectification_fast(qv, weight, idxs, return_dist=True)
        #qv_new = table_rectification(qv, weight, idxs)
        qv = torch.round(torch.clamp(qv_new, 0, q_max))
        #if r_i + 1 < rectification_iterations:
        idxs = pairwise_dist(qv.float(), weight)
        if torch.abs(cur_dist - prev_dist) < 1.0:
            break
        prev_dist = cur_dist
    qv = qv * sign.unsqueeze(0)
    return idxs, qv


def get_codebook_attn_weighted(weight: torch.Tensor, sign, targe_sz, 
                               importance, rectification_iterations=30, normalize=True):
    if weight.shape[-1] == 8:
        qv = q_vectors_4096.clone().to(weight.device)
    else:
        qv = get_packed_abs_grid_d4().to(weight.device)
        qv = torch.round(qv / qv.max() * 127)
        #qv = q_vectors_4096_4.clone().to(weight.device)
        # qv = torch.cat([qv, qv // 2, qv // 4, qv // 8])
        # qv = torch.unique(qv, dim=0)
    #qv = torch.round(qv / qv.max() * 127)
    #qv = torch.cat([qv, qv // 2, qv // 4, qv // 8])
    #qv = qv[:, :weight.shape[-1]]
    q_max = 127 #qv.max()

    sign = to_sign(sign, weight.shape[-1]).to(weight.device)
    weight = weight.abs().float()
    if normalize:
        weight = q_max * weight
    
    best_alpha = 0.01
    iter_shape = weight.shape[0]

    attn = pairwise_attn(qv.float(), weight[:iter_shape//2, :], alpha=best_alpha)
    scale = importance[:iter_shape//2, :].mean(dim=1, keepdim=True)
    attn = attn * scale
    scores = torch.sum(attn, dim=0)

    attn = pairwise_attn(qv.float(), weight[iter_shape//2:, :], alpha=best_alpha)
    scale = importance[iter_shape//2:, :].mean(dim=1, keepdim=True)
    attn = attn * scale
    scores += torch.sum(attn, dim=0)

    top_idxs = torch.topk(scores, targe_sz)[1]
    qv = qv[top_idxs, :]

    idxs = pairwise_dist(qv.float(), weight)
    for r_i in range(rectification_iterations):
        qv_new = table_rectification_fast_weighted(qv, weight, idxs, importance)
        #qv_new = table_rectification(qv, weight, idxs)
        qv = torch.round(torch.clamp(qv_new, 0, q_max))
        #if r_i + 1 < rectification_iterations:
        idxs = pairwise_dist(qv.float(), weight)
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
    centroids, stats = clustering.kmeans(
            target_sz, da, niter=50, verbose=False, return_stats=True)
    qv = torch.from_numpy(centroids).to(weight.device)

    idxs = pairwise_dist(qv, x, normalize=False)
    
    return idxs, torch.round(qv)
    
    
def compress_by_signed_notebook(weight: torch.Tensor, super_group_size = 256,
                                group_size = 8, target_sz=2**14, verbose=True, stat=None):
    out_ch, in_ch = weight.shape
    importance = None
    if stat is not None:
        importance = torch.ones_like(weight) * stat.unsqueeze(0).to(weight.device)

    n_sub_groups = super_group_size // group_size
    assert out_ch % super_group_size == 0
    
    gweight = weight.reshape(out_ch, -1, super_group_size)
    if importance is not None:
        importance = importance.reshape(out_ch, -1, super_group_size)
        denum = importance.sum(dim=-1, keepdim=True)
        importance = importance / denum
    
    super_scale = gweight.abs().max(dim=-1, keepdim=True)[0]
    mask = (gweight >= 0).long()
    mask = torch.sum(mask, dim=-1, keepdim=True)
    super_scale[torch.where(mask < super_group_size // 2)] *= -1
    gweight = gweight / super_scale
    
    super_group_shape = gweight.shape
    gweight = gweight.reshape(-1, group_size)
    if importance is not None:
        importance = importance.reshape(-1, group_size)
    
    n_signs = 2**group_size
    sg = get_signed_groups(gweight)
    sg_weight = torch.zeros(n_signs)

    for i in range(n_signs):
        sg_weight[i] = torch.count_nonzero(sg == i)
        #print(i, sg_weight[i])
        
    sg_weight /= torch.sum(sg_weight)
    n_per_sg_group = torch.round(target_sz * sg_weight)

    diff = target_sz - torch.sum(n_per_sg_group)
    if diff != 0:
        n_per_sg_group[torch.argmax(n_per_sg_group)] += diff
    
    codebook = []
    idxs = torch.zeros(gweight.shape[0]).long().to(weight.device)
    offset = 0
    for i in range(n_signs):#tqdm(range(n_signs)):
        i_idxs = torch.where(sg == i)[0]
        gr_weights = gweight[i_idxs, :]
        if importance is not None:
            gr_importance = importance[i_idxs, :].mean(dim=1).cpu().numpy()
            #gr_idxs, gr_codebook = get_codebook_attn_weighted(gr_weights, i, int(n_per_sg_group[i].item()), gr_importance)
            gr_idxs, gr_codebook = get_codebook_kmeans(gr_weights, i, int(n_per_sg_group[i].item()),
                                                       sample_weight=gr_importance)
        else:
            gr_idxs, gr_codebook = get_codebook_kmeans(gr_weights, i, int(n_per_sg_group[i].item()))
            #gr_idxs, gr_codebook = get_codebook_attn(gr_weights, i, int(n_per_sg_group[i].item()))
        #gr_idxs, gr_codebook = get_kmeans_codebook(gr_weights, int(n_per_sg_group[i].item()))
        idxs[i_idxs] = gr_idxs + offset
        codebook.append(gr_codebook)
        offset += gr_codebook.shape[0]
    
    codebook = torch.cat(codebook)
    
    #idxs, codebook = get_codebook_kmeans(gweight, None, target_sz, True, importance.mean(dim=1).cpu().numpy())

    super_scale = super_scale / 127
    qweights = codebook[idxs, :]
    qweights = qweights.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1)
    qweights = qweights.reshape(super_group_shape[0], super_group_shape[1], -1)

    # gweight = gweight.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1)
    # gweight = gweight.reshape(super_group_shape[0], super_group_shape[1], -1)
    
    # num = torch.sum(torch.mul(qweights, gweight * super_scale), dim=-1, keepdim=True)
    # denum = torch.sum(torch.mul(qweights, qweights), dim=-1, keepdim=True)
    # super_scale_ = num / denum

    qweights = qweights * super_scale
    qweights = qweights.reshape(weight.shape)

    if verbose:
        abs_err = torch.mean((weight - qweights)**2)
        rel_err = abs_err / torch.mean(weight**2)
        
        print(f"Abs err {abs_err}, rel_err: {rel_err}")
        sys.stdout.flush()
    #print(torch.sum(n_per_sg_group))
    return qweights



def compress_by_signed_notebook_mit_residual(weight: torch.Tensor, super_group_size = 256,
                                group_size = 8, target_sz=2**14, stat=None):
    out_ch, in_ch = weight.shape
    n_sub_groups = super_group_size // group_size
    assert out_ch % super_group_size == 0
    
    gweight = weight.reshape(out_ch, -1, super_group_size)

    importance = None
    if stat is not None:
        importance = torch.ones_like(weight) * stat.unsqueeze(0).to(weight.device)
        importance = importance.reshape(out_ch, -1, super_group_size)
    
    super_scale = gweight.abs().max(dim=-1, keepdim=True)[0]
    mask = (gweight >= 0).long()
    mask = torch.sum(mask, dim=-1, keepdim=True)
    super_scale[torch.where(mask < super_group_size // 2)] *= -1
    gweight = gweight / super_scale
    
    super_group_shape = gweight.shape
    gweight = gweight.reshape(-1, group_size)
    
    if importance is not None:
        importance = importance.reshape(-1, group_size)
    
    n_signs = 2**group_size
    sg = get_signed_groups(gweight)
    sg_weight = torch.zeros(n_signs)

    for i in range(n_signs):
        sg_weight[i] = torch.count_nonzero(sg == i)
        #print(i, sg_weight[i])
        
    sg_weight /= torch.sum(sg_weight)
    n_per_sg_group = torch.round(target_sz * sg_weight)

    diff = target_sz - torch.sum(n_per_sg_group)
    if diff != 0:
        n_per_sg_group[torch.argmax(n_per_sg_group)] += diff
    
    codebook = []
    idxs = torch.zeros(gweight.shape[0]).long().to(weight.device)
    
    offset = 0
    for i in range(n_signs):
        i_idxs = torch.where(sg == i)[0]
        gr_weights = gweight[i_idxs, :]
        #gr_idxs, gr_codebook = get_codebook_attn(gr_weights, i, int(n_per_sg_group[i].item()))
        
        gr_importance = None
        if importance is not None:
            gr_importance = importance[i_idxs, :].mean(dim=1).cpu().numpy()
            #gr_importance = importance[i_idxs, :].cpu().numpy()      
        gr_idxs, gr_codebook = get_codebook_kmeans(gr_weights, i, int(n_per_sg_group[i].item()), sample_weight=gr_importance)
        #gr_idxs, gr_codebook = get_kmeans_codebook(gr_weights, int(n_per_sg_group[i].item()))
        idxs[i_idxs] = gr_idxs + offset
        codebook.append(gr_codebook)
        offset += gr_codebook.shape[0]
    
    codebook = torch.cat(codebook)

    super_scale = super_scale / 127
    qweights = codebook[idxs, :]

    if True:
        n_signs = 2**8
        residual = gweight * 127 - qweights
        if residual.shape[-1] != 8:
            residual = residual.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1)
            residual = residual.reshape(super_group_shape[0], super_group_shape[1], -1)
            residual = residual.reshape(-1, 8)
            
            importance = importance.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1)
            importance = importance.reshape(super_group_shape[0], super_group_shape[1], -1)
            importance = importance.reshape(-1, 8)

        sg = get_signed_groups(residual)
        n_per_sg_group = torch.ones(n_signs)
        
        residual_codebook = []
        residual_idxs = torch.zeros(residual.shape[0]).long().to(weight.device)
        
        offset = 0
        for i in range(n_signs):
            i_idxs = torch.where(sg == i)[0]
            gr_weights = residual[i_idxs, :]
            gr_importance = None
            if importance is not None:
                gr_importance = importance[i_idxs, :].mean(dim=1).cpu().numpy()
                #gr_importance = importance[i_idxs, :].cpu().numpy()
            gr_idxs, gr_codebook = get_codebook_kmeans(gr_weights, i, int(n_per_sg_group[i].item()), normalize=False,
                                                       sample_weight=gr_importance)
            #gr_idxs, gr_codebook = get_codebook_attn(gr_weights, i, int(n_per_sg_group[i].item()), normalize=False)
            #gr_idxs, gr_codebook = get_kmeans_codebook(gr_weights, int(n_per_sg_group[i].item()))
            residual_idxs[i_idxs] = gr_idxs + offset
            residual_codebook.append(gr_codebook)
            offset += gr_codebook.shape[0]
        residual_codebook = torch.cat(residual_codebook)
        
        q_residual = residual_codebook[residual_idxs, :]

        qweights = qweights.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1)
        qweights = qweights.reshape(super_group_shape[0], super_group_shape[1], -1)
        q_residual = q_residual.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // 8, -1)
        q_residual = q_residual.reshape(super_group_shape[0], super_group_shape[1], -1)
        
        qweights = qweights + q_residual
    else:
        qweights = qweights.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1)
        qweights = qweights.reshape(super_group_shape[0], super_group_shape[1], -1)

    qweights = qweights * super_scale
    qweights = qweights.reshape(weight.shape)
    
    

    abs_err = torch.mean((weight - qweights)**2)
    rel_err = abs_err / torch.mean(weight**2)
    
    print(f"Abs err {abs_err}, rel_err: {rel_err}")
    sys.stdout.flush()
    #print(torch.sum(n_per_sg_group))
    return qweights


def compress_by_signed_notebook_group_wise(weight: torch.Tensor, super_group_size = 256,
                                group_size = 8, target_sz=2**14, stat=None):
    out_ch, in_ch = weight.shape
    n_sub_groups = super_group_size // group_size
    
    res = []

    ratio = max(in_ch // out_ch, 1)
    
    group_size_2 = 8 * super_group_size * ratio
    n_iters = in_ch // group_size_2

    for i in range(n_iters):
        cur_stat = None
        if stat is not None:
            cur_stat = stat[i*group_size_2:(i+1)*group_size_2]
        gr_w = weight[:, i*group_size_2:(i+1)*group_size_2]
        qgr_w = compress_by_signed_notebook(gr_w, super_group_size, group_size, target_sz, verbose=False, stat=cur_stat)
        res.append(qgr_w)
    
    res = torch.cat(res, dim=1)
    
    abs_err = torch.mean((weight - res)**2)
    rel_err = abs_err / torch.mean(weight**2)
    print(f"Abs err {abs_err}, rel_err: {rel_err}")
    sys.stdout.flush()
    
    return res
    
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

    assert gweight.shape[1] % 8 == 0

    n_steps = gweight.shape[1] // 4
    codebook_group_sz = gweight.shape[1] // n_steps

    for i_g in range(n_steps):#gweight.shape[1]):
        i_gweights = gweight[:, i_g * codebook_group_sz: (i_g + 1) * codebook_group_sz, :]
        i_gweights = i_gweights.reshape(-1, group_size)
        sg = get_signed_groups(i_gweights)
        sg_weight = torch.zeros(n_signs)

        for i in range(n_signs):
            sg_weight[i] = torch.count_nonzero(sg == i)
            
        sg_weight /= torch.sum(sg_weight)
        n_per_sg_group = torch.round(target_sz * sg_weight)

        diff = target_sz - torch.sum(n_per_sg_group)
        if diff != 0:
            n_per_sg_group[torch.argmax(n_per_sg_group)] += diff

        codebook = []
        idxs = torch.zeros(i_gweights.shape[0]).long().to(weight.device)
        offset = 0
        for i in range(n_signs):
            i_idxs = torch.where(sg == i)[0]
            gr_weights = i_gweights[i_idxs, :]
            gr_idxs, gr_codebook = get_codebook_attn(gr_weights, i, int(n_per_sg_group[i].item()))
            #gr_idxs, gr_codebook = get_kmeans_codebook(gr_weights, int(n_per_sg_group[i].item()))
            idxs[i_idxs] = gr_idxs + offset
            codebook.append(gr_codebook)
            offset += gr_codebook.shape[0]
        
        codebook = torch.cat(codebook)
        
        group_codebook.append(codebook)
        group_idxs.append(idxs)
    
    qweights = []
    q_max = 127
    for i_g in range(n_steps):
        i_qweights = group_codebook[i_g][group_idxs[i_g], :]
        q_max = max(q_max, group_codebook[i_g].max())
        i_qweights = i_qweights.reshape(gweight.shape[0], codebook_group_sz, gweight.shape[2])
        qweights.append(i_qweights)
    qweights = torch.cat(qweights, dim=1)

    super_scale = super_scale / q_max
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
    
