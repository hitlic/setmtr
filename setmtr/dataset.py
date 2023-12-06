from torch.utils.data import Dataset, DataLoader
import random as rand
import torch


class SetMTrDataset(Dataset):
    def __init__(self, set_list, element_num, mask_idx, max_set_size, mask_p=0.15, mask_rand_p=0.1, mask_keep_p=0.1, repeat=1, has_element=True, has_feat=False, aux=False):
        super().__init__()
        self.set_list = set_list            # 集合数据列表
        self.element_num = element_num      # 唯一元素数量
        self.mask_idx = mask_idx            # mask符号的index
        self.max_set_size = max_set_size    # 最大集合大小
        self.mask_p = mask_p                # 集合中被mask的元素比例
        self.mask_rand_p = mask_rand_p      # 被选择为mask的元素用随机元素代替的比例
        self.mask_keep_p = mask_keep_p      # 被选择为mask的元素使用原元素保持不变的比例
        self.repeat = repeat                # 每个集合数据被重复mask多少次
        self.has_element = has_element      # 数据是否有元素id
        self.has_feat = has_feat            # 数据是否有元素特征
        self.aux = aux                      # 是否使用辅助任务

    def __len__(self):
        return len(self.set_list)

    def __getitem__(self, index):
        if not self.aux:
            set_data = self.set_list[index]
            return [self.rand_mask(set_data) for _ in range(self.repeat)]  # 重复mask一个集合数据repeat次
        else:
            set_data, cls_label = self.set_list[index]
            return [(self.rand_mask(set_data), cls_label) for _ in range(self.repeat)]

    def rand_mask(self, set_data):
        if len(set_data) > self.max_set_size:
            set_data = rand.sample(set_data, self.max_set_size)
        data = []
        target = []
        for e in set_data:
            target.append(e)
            prob = rand.random()
            if prob < self.mask_p:
                prob /= self.mask_p
                if prob < self.mask_rand_p:                            # rand
                    if self.has_element and self.has_feat:    # (元素, 特征)
                        rand_e = (rand.randrange(self.element_num),  [rand.random() for _ in range(len(e[1]))])
                    elif self.has_feat:                   # 仅特征
                        rand_e = [rand.random() for _ in range(len(e))]
                    else:                                 # 仅元素
                        rand_e = rand.randrange(self.element_num)
                    data.append(rand_e)
                elif prob < self.mask_rand_p + self.mask_keep_p:       # keep
                    data.append(e)
                else:                                                  # mask
                    if self.has_element and self.has_feat:
                        mask_e = (self.mask_idx, [0]*len(e[1]))
                    elif self.has_feat:
                        mask_e = [0] * len(e)
                    else:
                        mask_e = self.mask_idx
                    data.append(mask_e)
            else:
                data.append(e)
        return data, target


class SetMTrDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, padd_idx, max_set_size, has_element, has_feat, shuffle, aux=False, test=False):
        self.padd_idx = padd_idx
        self.max_set_size = max_set_size
        self.has_element = has_element
        self.has_feat = has_feat
        self.aux = aux
        self.test = test
        super().__init__(dataset, batch_size, shuffle, collate_fn=self._collate)

    def _collate(self, batch):
        if self.test:
            batch_data = batch
        else:
            batch_data = []
            for data in batch:
                batch_data.extend(data)

        cls_label = None
        if self.aux:
            batch_data, cls_label = zip(*batch_data)
            cls_label = torch.FloatTensor(cls_label).unsqueeze(-1)

        if self.test:
            datas = targets = batch_data
        else:
            datas, targets = zip(*batch_data)
        datas_padded = padding_batch(datas, self.padd_idx, self.max_set_size, self.has_element, self.has_feat)
        targets_padded = padding_batch(targets, self.padd_idx, self.max_set_size, self.has_element, self.has_feat)

        if self.has_element and self.has_feat:                      # 同时有集合元素id及元素特征向量
            ele_data, feat_data, masks = datas_padded      # pylint: disable=unbalanced-tuple-unpacking
            ele_targets, feat_targets, _ = targets_padded  # pylint: disable=unbalanced-tuple-unpacking
            ele_data = torch.LongTensor(ele_data)
            feat_data = torch.FloatTensor(feat_data)
            ele_targets = torch.LongTensor(ele_targets)
            feat_targets = torch.FloatTensor(feat_targets)
            masks = torch.BoolTensor(masks)
            if self.aux:
                return ele_data, feat_data, masks, (ele_targets, feat_targets, masks, cls_label)
            else:
                return ele_data, feat_data, masks, (ele_targets, feat_targets, masks)
        elif self.has_feat:                                         # 仅有元素集合特征向量
            feat_data, masks = datas_padded
            feat_targets, _ = targets_padded
            feat_data = torch.FloatTensor(feat_data)
            feat_targets = torch.FloatTensor(feat_targets)
            masks = torch.BoolTensor(masks)
            if self.aux:
                return feat_data, masks, (feat_targets, masks, cls_label)
            else:
                return feat_data, masks, (feat_targets, masks)
        else:                                                       # 仅有集合元素id
            ele_data, masks = datas_padded
            ele_targets, _ = targets_padded
            ele_data = torch.LongTensor(ele_data)
            ele_targets = torch.LongTensor(ele_targets)
            masks = torch.BoolTensor(masks)
            if self.aux:
                return ele_data, masks, (ele_targets, masks, cls_label)
            else:
                return ele_data, masks, (ele_targets, masks)


class TestDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, padd_idx, max_set_size, has_element, has_feat):
        self.padd_idx = padd_idx
        self.max_set_size = max_set_size
        self.has_element = has_element
        self.has_feat = has_feat
        super().__init__(dataset, batch_size, False, collate_fn=self._collate)

    def _collate(self, batch):
        datas, targets = zip(*batch)
        datas_padded = padding_batch(datas, self.padd_idx, self.max_set_size, self.has_element, self.has_feat)
        if self.has_element and self.has_feat:                      # 同时有集合元素id及元素特征向量
            ele_data, feat_data, masks = datas_padded  # pylint: disable=unbalanced-tuple-unpacking
            ele_data = torch.LongTensor(ele_data)
            feat_data = torch.FloatTensor(feat_data)
            masks = torch.BoolTensor(masks)
            cls_targets = torch.LongTensor(targets)
            # 在测试中，模型重构的目标是集合自身
            return ele_data, feat_data, masks, (ele_data, feat_data, masks, cls_targets)
        elif self.has_feat:                                         # 仅有元素集合特征向量
            feat_data, masks = datas_padded
            feat_data = torch.FloatTensor(feat_data)
            masks = torch.BoolTensor(masks)
            cls_targets = torch.LongTensor(targets)
            return feat_data, masks, (feat_data, masks, cls_targets)
        else:                                                       # 仅有集合元素id
            ele_data, masks = datas_padded
            ele_data = torch.LongTensor(ele_data)
            cls_targets = torch.LongTensor(targets)
            masks = torch.BoolTensor(masks)
            return ele_data, masks, (ele_data, masks, cls_targets)


def padding_batch(batch_data, padd_idx, max_size, has_element, has_feat):
    """
    padding one batch
    """
    m_size = max_size if max_size is not None else max(len(d) for d in batch_data)
    padded_data = [padding(data, padd_idx, m_size, has_element, has_feat) for data in batch_data]
    padded_data = tuple(zip(*padded_data))
    return (padded_data, m_size) if max_size is None else padded_data


def padding(data_lst, padd_idx, max_size, has_element, has_feat):
    """
    padding one instance
    """
    if len(data_lst) > max_size:
        mask = [True] * max_size
        return data_lst[:max_size], mask

    padd_len = max_size - len(data_lst)
    padds = [padd_idx] * padd_len
    mask = [True] * len(data_lst) + [False] * padd_len

    if has_element and has_feat:                     # 同时有集合元素id及元素特征向量
        eles, feats = zip(*data_lst)
        feat_dim = len(feats[0])
        feat_padds = [[0.] * feat_dim] * padd_len
        return list(eles) + padds, list(feats) + feat_padds, mask
    elif has_feat:                                   # 仅有元素集合特征向量
        feats = data_lst
        feat_dim = len(feats[0])
        feat_padds = [[0.] * feat_dim] * padd_len
        return feats + feat_padds, mask
    else:                                            # 仅有集合元素id
        return list(data_lst) + padds, mask
