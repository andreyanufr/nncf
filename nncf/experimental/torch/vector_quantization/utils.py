import torch



def pairwise_dist(xyz1, xyz2, normalize=False):
    if normalize:
        xyz1 = xyz1 / torch.norm(xyz1, dim=-1, keepdim=True)
        xyz2 = xyz2 / torch.norm(xyz2, dim=-1, keepdim=True)
    
    r_xyz1 = torch.sum(xyz1 * xyz1, dim=1, keepdim=True)  # (B,N,1)
    r_xyz2 = torch.sum(xyz2 * xyz2, dim=1, keepdim=True)  # (B,M,1)
    mul = torch.matmul(xyz2, xyz1.permute(1, 0))         # (B,M,N)
    dist = r_xyz2 - 2 * mul + r_xyz1.permute(1, 0)       # (B,M,N)
    return torch.argmin(dist, dim=-1)

def pairwise_dist_torch(codebook, vectors, p=2):
    dist = torch.cdist(vectors, codebook, p=p)
    return torch.argmin(dist, dim=-1)


def pairwise_attn_torch(codebook, vectors, alpha=0.01, p=2):
    dist = torch.cdist(vectors, codebook, p=p)

    tmp = torch.exp(-alpha * dist)
    
    attn = tmp / torch.sum(tmp, dim=-1, keepdim=True)
    
    return attn


# xyz1 - codebook
# xyz2 - vectors
def pairwise_attn(xyz1, xyz2, normalize=False, alpha=0.01):
    if normalize:
        xyz1 = xyz1 / torch.norm(xyz1, dim=-1, keepdim=True)
        xyz2 = xyz2 / torch.norm(xyz2, dim=-1, keepdim=True)
    r_xyz1 = torch.sum(xyz1 * xyz1, dim=1, keepdim=True)  # (B,N,1)
    r_xyz2 = torch.sum(xyz2 * xyz2, dim=1, keepdim=True)  # (B,M,1)
    mul = torch.matmul(xyz2, xyz1.permute(1, 0))         # (B,M,N)
    dist = r_xyz2 - 2 * mul + r_xyz1.permute(1, 0)       # (B,M,N)
    tmp = torch.exp(-alpha * dist)
    
    attn = tmp / torch.sum(tmp, dim=-1, keepdim=True)
    
    return attn


def table_rectification_fast(table : torch.Tensor, gt, indexes, return_dist=False):    
    # qw = table[indexes, :]
    # dists = torch.mean((qw - gt)**2, dim=-1)
    # print("Before: ", dists.mean())
    new_table = table.clone()
    # iteration over number of vectors
    sum_dim = [0]
    for i in range(1, len(gt.shape) - 1):
        sum_dim.append(i)
    sum_dim = tuple(sum_dim)
    
    empty_centers = []
    hist = []    
    for i in range(table.shape[0]):
        mask = (i == indexes)
        hist.append(torch.count_nonzero(mask))
        if not torch.any(mask):
            empty_centers.append(i)
            #print("SKIP ELMENT IN CODEBOOK: ", i)
            continue
        mask = mask.float()
        denom = torch.sum(mask)
        if denom < 1.0:
            print("SKIP ELMENT IN CODEBOOK: ", i)
            continue

        new_vec = torch.sum(gt * mask.unsqueeze(-1), dim=sum_dim) / denom
        new_vec = new_vec.squeeze()
        #new_vec = new_vec / new_vec.max() * scale[i]
        new_vec = torch.round(torch.clamp(new_vec, min=0.0, max=127))
        new_table[i, :] = new_vec

    if len(empty_centers) > 0:
        sz = len(empty_centers)
        top_clusters = torch.topk(torch.tensor(hist), sz)[1]
        top_clusters = [t for t in  top_clusters if hist[t] > 10]
        sz = len(top_clusters)
        cur_top = 0
        
        for idx in empty_centers:
            if sz == 0:
                break
            cur_gt = gt[torch.where(indexes == top_clusters[cur_top])]
            dist = torch.mean((cur_gt - new_table[top_clusters[cur_top]].unsqueeze(0))**2, dim=1)
            new_idx = torch.argmax(dist)
            new_table[idx, :] = cur_gt[new_idx]           
            cur_top = (cur_top + 1) % sz

    if return_dist:
        qw = new_table[indexes, :]
        return new_table, torch.mean((qw - gt)**2)
    # qw = new_table[indexes, :]
    # dists = torch.mean((qw - gt)**2, dim=-1)
    # print("After: ", dists.mean())
 
    return new_table


def table_rectification_fast_weighted(table : torch.Tensor, gt, indexes, importance):    
    # qw = table[indexes, :]
    # dists = torch.mean((qw - gt)**2, dim=-1)
    # print("Before: ", dists.mean())
    new_table = table.clone()
    # iteration over number of vectors
    sum_dim = [0]
    for i in range(1, len(gt.shape) - 1):
        sum_dim.append(i)
    sum_dim = tuple(sum_dim)
    
    for i in range(table.shape[0]):
        mask = (i == indexes)
        if not torch.any(mask):
            print("SKIP ELMENT IN CODEBOOK: ", i)
            continue
        mask = mask.float()
        importance_i = importance * mask.unsqueeze(-1)
        importance_i = importance_i / importance_i.sum(dim=0, keepdim=True)

        new_vec = torch.sum(gt * importance_i, dim=sum_dim)
        new_vec = new_vec.squeeze()
        #new_vec = new_vec / new_vec.max() * scale[i]
        new_vec = torch.round(torch.clamp(new_vec, min=0.0, max=127))
        new_table[i, :] = new_vec

    # qw = new_table[indexes, :]
    # dists = torch.mean((qw - gt)**2, dim=-1)
    # print("After: ", dists.mean())
 
    return new_table


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
