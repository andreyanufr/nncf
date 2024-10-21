import torch



def pairwise_dist(xyz1, xyz2, normalize=True):
    if normalize:
        xyz1 = xyz1 / torch.norm(xyz1, dim=-1, keepdim=True)
        xyz2 = xyz2 / torch.norm(xyz2, dim=-1, keepdim=True)
    
    r_xyz1 = torch.sum(xyz1 * xyz1, dim=1, keepdim=True)  # (B,N,1)
    r_xyz2 = torch.sum(xyz2 * xyz2, dim=1, keepdim=True)  # (B,M,1)
    mul = torch.matmul(xyz2, xyz1.permute(1, 0))         # (B,M,N)
    dist = r_xyz2 - 2 * mul + r_xyz1.permute(1, 0)       # (B,M,N)
    return torch.argmin(dist, dim=-1)


# xyz1 - codebook
# xyz2 - vectors
def pairwise_attn(xyz1, xyz2, normalize=True, alpha=0.01):
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


def table_rectification_fast(table : torch.Tensor, gt, indexes):    
    qw = table[indexes, :]
    #dists = torch.mean((qw - gt)**2, dim=-1)
    #print("Before: ", dists.mean())
    new_table = table.clone()
    denums = []
    # iteration over number of vectors
    sum_dim = [0]
    for i in range(1, len(gt.shape) - 1):
        sum_dim.append(i)
    sum_dim = tuple(sum_dim)
    for i in range(table.shape[0]):
        mask = (i == indexes)
        if not torch.any(mask):
            continue
        mask = mask.float()
        denom = torch.sum(mask)
        if denom < 1.0:
            continue
        denums.append(denom.item())
        new_vec = torch.sum(gt * mask.unsqueeze(-1), dim=sum_dim) / denom
        new_vec = new_vec.squeeze()
        new_vec = torch.round(torch.clamp(new_vec, min=0.0, max=127))
        new_table[i, :] = new_vec

    # qw = new_table[indexes, :]
    # dists = torch.mean((qw - gt)**2, dim=-1)
    # print("After: ", dists.mean())
 
    return new_table



def table_rectification(table : torch.Tensor, gt, indexes):    
    qw = table[indexes, :]
    dists = torch.mean((qw - gt)**2, dim=-1)
    
    #print("Before: ", dists.mean())

    new_table = table.clone()
    sum_dim = [0]
    for i in range(1, len(gt.shape) - 1):
        sum_dim.append(i)
    sum_dim = tuple(sum_dim)

    # iteration over number of vectors
    for i in range(table.shape[0]):
        if i == 0:
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
        new_vec = torch.sum(gt * dist_i.unsqueeze(-1), dim=sum_dim)
        new_vec = new_vec.squeeze()
        #max_t = torch.max(table[i, :])
        new_vec = new_vec / torch.max(new_vec) * table[i, :].max()
        new_vec = torch.round(torch.clamp(new_vec, min=0.0, max=255.0))
        #new_vec[torch.where(table[i, :] == 43)] = 43
        new_table[i, :] = new_vec

    qw = new_table[indexes, :]
    dists = torch.mean((qw - gt)**2, dim=-1)
    #print("After: ", dists.mean())
 
    return new_table
