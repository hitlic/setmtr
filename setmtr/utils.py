import pickle
import random as rand
import torch
from collections import defaultdict


def load_dataset(file, val_ratio):
    with open(file, 'rb') as f:
        ds_lst = pickle.load(f)
    
    ds_lst_loaded = []
    for ds in ds_lst:
        train_sets = ds["train_sets"]
        train_sets_baseline = ds["train_sets_baseline"]
        test_sets = ds["test_sets"]
        ele_num = ds["ele_num"]
        max_set_size = ds["max_set_size"]

        rand.shuffle(train_sets)
        rand.shuffle(test_sets)

        # 训练、验证集划分
        val_num = int(len(train_sets) * val_ratio)
        idx = list(range(len(train_sets)))
        rand.shuffle(idx)
        train_idx, val_idx = idx[:-val_num], idx[-val_num:]
        train_sets_ = [train_sets[i] for i in train_idx]
        val_sets_ = [train_sets[i] for i in val_idx]

        # baseline数据的训练、验证集划分
        val_num = int(len(train_sets_baseline) * val_ratio)
        idx = list(range(len(train_sets_baseline)))
        rand.shuffle(idx)
        train_idx, val_idx = idx[:-val_num], idx[-val_num:]
        train_sets_baseline_ = [train_sets_baseline[i] for i in train_idx]
        val_sets_baseline_ = [train_sets_baseline[i] for i in val_idx]
        ds_lst_loaded.append((train_sets_, val_sets_, train_sets_baseline_, val_sets_baseline_, test_sets, ele_num, max_set_size))
    return ds_lst_loaded


def unpack(data, num=None):
    if num is None:
        return data
    if len(data) < num:
        data = [d for d in data]
        data.extend([None] * (num - len(data)))
    else:
        data = data[:num]
    return data


def groupby_apply(values: torch.Tensor, keys: torch.Tensor, reduction: str = "mean"):
    """
    Groupby apply for torch tensors.
    Example: 
        Code:
            x = torch.FloatTensor([[1,1], [2,2],[3,3],[4,4],[5,5]])
            g = torch.LongTensor([0,0,1,1,1])
            print(groupby_apply(x, g, 'mean'))
        Output:
            tensor([[1.5000, 1.5000],
                    [4.0000, 4.0000]])
    Args:
        values: values to aggregate - same size as keys
        keys: tensor of groups. 
        reduction: either "mean" or "sum"
    Returns:
        tensor with aggregated values
    """
    if reduction == "mean":
        reduce = torch.mean
    elif reduction == "sum":
        reduce = torch.sum
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")
    keys = keys.to(values.device)
    _, counts = keys.unique(return_counts=True)
    reduced = torch.stack([reduce(item, dim=0) for item in torch.split_with_sizes(values, tuple(counts))])
    return reduced

def merge_dict(dict_lst):
    m_dict = defaultdict(list)
    key_set = set()
    for d in dict_lst:
        for k in d.keys():
            key_set.add(k)
    for d in dict_lst:
        for key in key_set:
            m_dict[key].append(d.get(key, None))
    return m_dict

