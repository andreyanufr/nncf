import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from nncf.tensor import Tensor


class WeightVQ:
    def __init__(self, codebook, idx_codebook, residual, idx_residual, scale, super_group_shape, group_size):
        self.codebook = codebook
        self.idx_codebook = idx_codebook
        self.residual = residual
        self.idx_residual = idx_residual
        self.scale = scale
        self.super_group_shape = super_group_shape
        self.group_size = group_size

    def decompress(self):
        res = self.codebook.data[self.idx_codebook.data, :]
        res = res.reshape(
            self.super_group_shape[0], self.super_group_shape[1], self.super_group_shape[2] // self.group_size, -1
        )
        res = res.reshape(self.super_group_shape[0], self.super_group_shape[1], -1)

        if self.residual is not None:
            q_residual = self.residual.data[self.idx_residual.data, :]
            q_residual = q_residual.reshape(
                self.super_group_shape[0], self.super_group_shape[1], self.super_group_shape[2] // 8, -1
            )
            q_residual = q_residual.reshape(self.super_group_shape[0], self.super_group_shape[1], -1)
            res = res + q_residual

        res = res * self.scale.data
        res = res.reshape(res.shape[0], res.shape[1] * res.shape[2])

        return res


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


def get_codebook_kmeans(
    weight: torch.Tensor, sign, targe_sz, normalize=True, sample_weight=None, init="k-means++", q_max=127
):
    if sign is not None:
        sign = to_sign(sign, weight.shape[-1]).to(weight.device)
        weight = weight.abs().float()

    if normalize:
        weight = q_max * weight
    weight = weight.round()

    kmeans = KMeans(n_clusters=targe_sz, random_state=0, init=init, n_init="auto" if isinstance(init, str) else 1, max_iter=1000).fit(
        weight.cpu().numpy(), sample_weight=sample_weight
    )
    qv = torch.tensor(kmeans.cluster_centers_).to(weight.device)
    if sign is not None:
        qv = torch.round(torch.clamp(qv, 0, q_max))
        qv = qv * sign.unsqueeze(0)
    else:
        qv = torch.round(torch.clamp(qv, -q_max, q_max))

    idxs = torch.tensor(kmeans.labels_).to(weight.device).long()

    return idxs, qv


def compress_by_signed_notebook_with_residual(
    weight: torch.Tensor, super_group_size=64, group_size=4, target_sz=2**8, stat=None, verbose=True, residual_sz=8
):
    weight = torch.Tensor(weight.data)
    out_ch, in_ch = weight.shape
    assert in_ch % super_group_size == 0
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
        gr_idxs, gr_codebook = get_codebook_kmeans(
            gr_weights, i, int(n_per_sg_group[i].item()), sample_weight=gr_importance
        )
        idxs[i_idxs] = gr_idxs + offset
        codebook.append(gr_codebook)
        offset += gr_codebook.shape[0]

    codebook = torch.cat(codebook)

    codebook_idxs, codebook = get_codebook_kmeans(
        gweight, None, target_sz, sample_weight=importance, init=codebook.cpu().numpy()
    )

    super_scale = super_scale / 127
    qweights = codebook[codebook_idxs, :]

    if residual_sz:
        n_signs = 2**8
        residual = gweight * 127 - qweights

        tmp = qweights[:]
        if residual.shape[-1] != residual_sz:
            residual = residual.reshape(
                super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1
            )
            residual = residual.reshape(super_group_shape[0], super_group_shape[1], -1)
            residual = residual.reshape(-1, residual_sz)

            tmp = tmp.reshape(super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1)
            tmp = tmp.reshape(super_group_shape[0], super_group_shape[1], -1)
            tmp = tmp.reshape(-1, residual_sz)

            if importance is not None:
                importance = importance.reshape(
                    super_group_shape[0], super_group_shape[1], super_group_shape[2] // group_size, -1
                )
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
            gr_idxs, gr_codebook = get_codebook_kmeans(
                gr_weights, i, int(n_per_sg_group[i].item()), normalize=False, sample_weight=gr_importance
            )
            residual_idxs[i_idxs] = gr_idxs + offset
            residual_codebook.append(gr_codebook)
            offset += gr_codebook.shape[0]
        residual_codebook = torch.cat(residual_codebook)

        # residual_idxs, residual_codebook = get_codebook_kmeans(residual, None,
        #                                 n_signs, normalize=False,
        #                                 sample_weight=importance, init=residual_codebook.cpu().numpy())

        q_residual = residual_codebook[residual_idxs, :]
        idxs = torch.where(tmp + q_residual < -128)
        res_idxs = residual_idxs[idxs[0]]
        visited = {}
        for i, ridx in enumerate(res_idxs):
            y = idxs[0][i]
            x = idxs[1][i]
            c = (ridx, x)
            diff = -128 - (tmp[y, x] + residual_codebook[ridx, x])
            if c in visited and diff > visited[c]:
                visited[c] = diff
            else:
                visited[c] = diff

        for k, v in visited.items():
            residual_codebook[k[0], k[1]] = min(residual_codebook[k[0], k[1]] + v, 0)
        q_residual = residual_codebook[residual_idxs, :]

        idxs = torch.where(tmp + q_residual > 127)
        res_idxs = residual_idxs[idxs[0]]
        visited = {}
        for i, ridx in enumerate(res_idxs):
            y = idxs[0][i]
            x = idxs[1][i]
            c = (ridx, x)
            diff = 127 - (tmp[y, x] + residual_codebook[ridx, x])
            if c in visited and diff > visited[c]:
                visited[c] = diff
            else:
                visited[c] = diff

        for k, v in visited.items():
            residual_codebook[k[0], k[1]] = max(residual_codebook[k[0], k[1]] + v, 0)
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
        abs_err = torch.mean((weight - qweights) ** 2)
        rel_err = abs_err / torch.mean(weight**2)

        print(f"Abs err {abs_err}, rel_err: {rel_err}")
        sys.stdout.flush()

    return WeightVQ(
        Tensor(codebook),
        Tensor(codebook_idxs),
        Tensor(residual_codebook),
        Tensor(residual_idxs),
        Tensor(super_scale),
        super_group_shape,
        group_size,
    )


def compress_by_signed_notebook_group_wise_with_residual(
    weight: torch.Tensor, super_group_size=256, group_size=8, target_sz=2**14, stat=None, per_rows=True, verbose=False
):
    res = []

    shape_per_codebook = 4096 * 1024
    #shape_per_codebook = 614400 -- for debug in small models
    n_iters = max(1, weight.shape[0] * weight.shape[1] // shape_per_codebook)  # minimal size of llama-3b

    if per_rows:
        while weight.shape[0] % n_iters != 0:
            n_iters -= 1
        step = weight.shape[0] // n_iters
        for i in range(n_iters):
            cur_stat = None
            if stat is not None:
                cur_stat = stat[0]
            gr_w = weight[i * step : (i + 1) * step, :]
            qgr_w = compress_by_signed_notebook_with_residual(
                gr_w, super_group_size, group_size, target_sz, stat=cur_stat, verbose=False
            )
            res.append(qgr_w)
    else:
        step = weight.shape[1] // n_iters
        while step % super_group_size != 0:
            n_iters -= 1
            step = weight.shape[1] // n_iters
        for i in range(n_iters):
            cur_stat = None
            if stat is not None:
                cur_stat = [stat[0][i * step : (i + 1) * step], stat[1][:, i * step : (i + 1) * step]]
            gr_w = weight[:, i * step : (i + 1) * step]
            qgr_w = compress_by_signed_notebook_with_residual(
                gr_w, super_group_size, group_size, target_sz, stat=cur_stat, verbose=False
            )
            res.append(qgr_w)

    if verbose:
        qweights = [qw.decompress() for qw in res]
        qweights = torch.cat(qweights, dim=0 if per_rows else 1)
        tmp = torch.tensor(weight.data)
        abs_err = torch.mean((tmp - qweights) ** 2)
        rel_err = abs_err / torch.mean(tmp**2)
        print(f"Abs err {abs_err}, rel_err: {rel_err}")
        sys.stdout.flush()

    return res
