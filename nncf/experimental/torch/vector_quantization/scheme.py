import torch



class GroupParams:
    def __init__(self, super_group_size=256, group_size=8):
        assert(super_group_size % group_size == 0)
        self._super_group_size = super_group_size
        self._group_size = group_size

        self._super_scale: float = 1.0
        # must be in range [0, 15] - 4bit per group
        self._group_scale = torch.ones(self._super_group_size / self._group_size, dtype=torch.int8) 
