import torch
import sys
import torch
import torch.nn as nn
import numpy as np
from nncf.tensor import Tensor
from sklearn.cluster import KMeans


def get_signed_groups(weight: Tensor):
    assert len(weight.shape) == 2
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


def get_codebook_kmeans(weight: torch.Tensor, sign, targe_sz, normalize=True,
                        sample_weight=None, init='k-means++', q_max = 127):
    if sign is not None:
        sign = to_sign(sign, weight.shape[-1]).to(weight.device)
        weight = weight.abs().float()

    if normalize:
        weight = q_max * weight
    weight = weight.round()

    kmeans = KMeans(n_clusters=targe_sz, random_state=0, init=init, n_init="auto", max_iter=1000).fit(weight.cpu().numpy(),
                                                                                               sample_weight=sample_weight)
    qv = torch.tensor(kmeans.cluster_centers_).to(weight.device)
    if sign is not None:
        qv = torch.round(torch.clamp(qv, 0, q_max))
        qv = qv * sign.unsqueeze(0)
    else:
        qv = torch.round(torch.clamp(qv, -q_max, q_max))

    idxs = torch.tensor(kmeans.labels_).to(weight.device).long()

    return idxs, qv



def compress_by_signed_notebook_with_residual(weight: torch.Tensor, super_group_size = 256,
                                group_size = 8, target_sz=2**14, stat=None, verbose=True, residual_sz=8):
    out_ch, _ = weight.shape
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
        
        gr_importance = None
        if importance is not None:
            gr_importance = importance[i_idxs, :].mean(dim=1).cpu().numpy()     
        gr_idxs, gr_codebook = get_codebook_kmeans(gr_weights, i, int(n_per_sg_group[i].item()), sample_weight=gr_importance)
        idxs[i_idxs] = gr_idxs + offset
        codebook.append(gr_codebook)
        offset += gr_codebook.shape[0]
    
    codebook = torch.cat(codebook)

    idxs, codebook = get_codebook_kmeans(gweight, None,
                                         target_sz, sample_weight=importance, init=codebook.cpu().numpy())

    super_scale = super_scale / 127
    qweights = codebook[idxs, :]

    if residual_sz:
        n_signs = 2**8
        residual = gweight * 127 - qweights
        if residual.shape[-1] != residual_sz:
            residual = residual.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1)
            residual = residual.reshape(super_group_shape[0], super_group_shape[1], -1)
            residual = residual.reshape(-1, residual_sz)

            if importance is not None:
                importance = importance.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1)
                importance = importance.reshape(super_group_shape[0], super_group_shape[1], -1)
                importance = importance.reshape(-1, residual_sz)

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
            gr_idxs, gr_codebook = get_codebook_kmeans(gr_weights, i, int(n_per_sg_group[i].item()), normalize=False,
                                                       sample_weight=gr_importance)
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
    
    
    if verbose:
        abs_err = torch.mean((weight - qweights)**2)
        rel_err = abs_err / torch.mean(weight**2)
        
        print(f"Abs err {abs_err}, rel_err: {rel_err}")
        sys.stdout.flush()
    return qweights


def compress_by_signed_notebook_group_wise_with_residual(weight: torch.Tensor, super_group_size = 256,
                                group_size = 8, target_sz=2**14, stat=None, per_rows=True):    
    res = []

    n_iters = max(1, weight.shape[0] * weight.shape[1] // (4096*1024)) # minimal size of llama-3b

    if per_rows:
        step = weight.shape[0] // n_iters
        for i in range(n_iters):
            cur_stat = None
            if stat is not None:
                cur_stat = stat[0]
            gr_w = weight[i*step:(i+1)*step, :]
            qgr_w = compress_by_signed_notebook_with_residual(gr_w, super_group_size, group_size, target_sz, stat=cur_stat, verbose=False)
            res.append(qgr_w)
        res = torch.cat(res, dim=0)
    else:
        step = weight.shape[1] // n_iters
        while step % super_group_size != 0:
            n_iters -= 1
            step = weight.shape[1] // n_iters
        for i in range(n_iters):
            cur_stat = None
            if stat is not None:
                cur_stat = [stat[0][i*step:(i+1)*step], stat[1][:, i*step:(i+1)*step]]
            gr_w = weight[:, i*step:(i+1)*step]
            qgr_w = compress_by_signed_notebook_with_residual(gr_w, super_group_size, group_size, target_sz, stat=cur_stat, verbose=False)
            res.append(qgr_w)
        res = torch.cat(res, dim=1)

    abs_err = torch.mean((weight - res)**2)
    rel_err = abs_err / torch.mean(weight**2)
    print(f"Abs err {abs_err}, rel_err: {rel_err}")
    sys.stdout.flush()
    return res
