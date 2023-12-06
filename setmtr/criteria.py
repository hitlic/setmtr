import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Metric
from torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryAccuracy, MulticlassAccuracy
from scipy.optimize import linear_sum_assignment
from utils import groupby_apply, unpack


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.shape[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1)
        loss = loss.mean() if self.reduction else loss
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)

        return self.epsilon*loss/n + (1-self.epsilon)*nll


class Matcher(nn.Module):
    """
    集合预测损失函数预测元素和目标元素之间的mathcer
    """
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    @torch.no_grad()
    def forward(self, preds, targets, masks=None):
        batch, num, dim = preds.shape

        # 计算costmatrix
        preds = preds.repeat(1, 1, num).reshape(-1, dim)
        if len(targets.shape) == 2:  # 当目标是元素id时
            targets = targets.repeat(1, num).reshape(-1)
            cost_matrics = self.criterion(preds, targets).reshape(batch, num, num).cpu()
        else:                        # 当目标是特征向量时
            targets = targets.repeat(1, 1, num).reshape(-1, dim)
            cost_matrics = self.criterion(preds, targets).mean(dim=1).reshape(batch, num, num).cpu()

        if masks is None:
            masks = [None] * batch
        index_preds, index_targets = [], []
        for i, cost_matrix, m in zip(range(batch), cost_matrics, masks):
            if m is not None:
                midx = torch.where(m)[0].cpu().numpy()
                cost_matrix = cost_matrix[:, midx]
            idx_p, idx_t = linear_sum_assignment(cost_matrix)  # index配对
            if m is not None:
                idx_t = midx[idx_t]  # 避免mask掉的元素不在列表最后的情况
            idx_p, idx_t = torch.LongTensor(idx_p), torch.LongTensor(idx_t)
            idx = torch.LongTensor([i] * len(idx_p))
            index_preds.append([idx, idx_p])
            index_targets.append([idx, idx_t])
        index_preds = [torch.cat(i) for i in zip(*index_preds)]
        index_targets = [torch.cat(i) for i in zip(*index_targets)]

        return index_preds, index_targets


def data_extract(model_out, label, has_element, has_feat):
    """
    输入处理
    """
    ele_preds = ele_targets = feat_preds = feat_targets = cls_label = None

    if has_element and has_feat:                    # 模型预测集合元素和特征
        ele_preds, feat_preds = model_out
        ele_targets, feat_targets, masks, cls_label = unpack(label, num=4)
    elif has_feat:                                        # 模型预测特征
        feat_preds = model_out
        feat_targets, masks, cls_label = unpack(label, num=3)
    else:                                                       # 模型预测集合元素
        ele_preds = model_out
        ele_targets, masks, cls_label = unpack(label, num=3)

    return ele_preds, ele_targets, feat_preds, feat_targets, masks, cls_label


class Loss(nn.Module):
    def __init__(self, has_element=True, has_feat=False, mu=0.5, epsilon=0.1):
        """
        has_element: 模型是否预测元素
        has_feat:    模型是否预测元素特征
        mu:           元素预测损失的重要性比例
        epsilon:      标签平滑参数
        """
        super().__init__()
        assert has_element or has_feat
        self.has_element = has_element
        self.has_feat = has_feat
        self.mu = mu
        if has_element:
            self.matcher = Matcher(LabelSmoothingCrossEntropy(epsilon, reduction='none'))
        else:
            self.matcher = Matcher(nn.MSELoss(reduction='none'))
        self.element_loss = LabelSmoothingCrossEntropy(epsilon)
        self.feat_loss = nn.MSELoss()

    def forward(self, model_out, label):
        ele_preds, ele_targets, feat_preds, feat_targets, masks, _ = data_extract(model_out, label, self.has_element, self.has_feat)
        if self.has_element and self.has_feat:                    # 模型预测集合元素和特征
            idx_p, idx_t = self.matcher(ele_preds, ele_targets, masks)
            return self.mu * self.element_loss(ele_preds[idx_p], ele_targets[idx_t]) \
                   + (1 - self.mu) * self.feat_loss(feat_preds[idx_p], feat_targets[idx_t])
        elif self.has_element:                                     # 模型预测集合元素
            idx_p, idx_t = self.matcher(ele_preds, ele_targets, masks)
            return self.element_loss(ele_preds[idx_p], ele_targets[idx_t])
        else:                                                       # 模型预测特征
            idx_p, idx_t = self.matcher(feat_preds, feat_targets, masks)
            return self.feat_loss(feat_preds[idx_p], feat_targets[idx_t])


class LossAux(nn.Module):
    def __init__(self, rec_loss, theta=0.5):
        super().__init__()
        self.rec_loss = rec_loss
        self.theta = theta

    def forward(self, model_out, labels):
        cls_out, rec_out = model_out
        *labels, cls_labels = labels
        return self.theta * self.rec_loss(rec_out, labels) + (1-self.theta) * F.binary_cross_entropy_with_logits(cls_out, cls_labels)


class ElementReconstructAcc(MulticlassAccuracy):
    def __init__(self, ele_num, has_feat=False, epsilon=0.1, name=None, aux=False):
        super().__init__(top_k=1, num_classes=ele_num+1)
        self.has_feat = has_feat
        self.name = name
        self.aux = aux
        self.matcher = Matcher(LabelSmoothingCrossEntropy(epsilon, reduction='none'))

    def forward(self, model_out, label):
        if self.aux:
            model_out = model_out[1]
            label = label[:-1]
        ele_preds, ele_targets, _, _, masks, _ = data_extract(model_out, label, True, self.has_feat)
        idx_p, idx_t = self.matcher(ele_preds, ele_targets, masks)
        return super().forward(ele_preds[idx_p], ele_targets[idx_t])


class CeAUC(BinaryAUROC):
    def __init__(self, has_feat=False, epsilon=0.1, name='ceauc'):
        super().__init__(pos_label=1)
        self.has_feat = has_feat
        self.name = name
        self.matcher = Matcher(LabelSmoothingCrossEntropy(epsilon, reduction='none'))
        self.ce_loss = LabelSmoothingCrossEntropy(epsilon, reduction='none')

    def forward(self,  model_out, label):
        ele_preds, ele_targets, _, _, masks, cls_label = data_extract(model_out, label, True, self.has_feat)
        idx_p, idx_t = self.matcher(ele_preds, ele_targets, masks)

        losses = self.ce_loss(ele_preds[idx_p], ele_targets[idx_t])
        loss_mat = torch.zeros_like(masks).float()
        loss_mat[idx_p] = losses
        mean_loss_rev = masks.sum(1)/loss_mat.sum(1)       # 集合各元素平均损失的倒数
        # mean_loss_rev = masks.sum(1)/loss_mat.max(1)[0]  # 集合各元素最大损失除以元素数量的倒数

        return super().forward(F.sigmoid(mean_loss_rev), cls_label.int())


class MseAUC(BinaryAUROC):
    def __init__(self, name=None):
        super().__init__(pos_label=1)
        self.name = name
        self.matcher = Matcher(nn.MSELoss(reduction='none'))
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, model_out, label):
        _, _, feat_preds, feat_targets, masks, cls_label = data_extract(model_out, label, False, True)
        idx_p, idx_t = self.matcher(feat_preds, feat_targets, masks)

        losses = self.mse_loss(feat_preds[idx_p], feat_targets[idx_t]).mean(dim=1)
        loss_mat = torch.zeros_like(masks).float()
        loss_mat[idx_p] = losses
        mean_loss_rev = masks.sum(1)/loss_mat.sum(1)       # 集合各元素平均损失的倒数
        # mean_loss_rev = masks.sum(1)/loss_mat.max(1)[0]  # 集合各元素最大损失除以元素数量的倒数

        return super().forward(F.sigmoid(mean_loss_rev), cls_label.int())


def jeccard(preds, targets, ele_groups):
    preds = F.one_hot(preds.argmax(1), preds.shape[1]).bool()
    targets = F.one_hot(targets, preds.shape[1]).bool()

    ele_and = preds & targets
    ele_or = preds | targets
    jcd = groupby_apply(ele_and.sum(1).float(), ele_groups, 'sum')/groupby_apply(ele_or.sum(1).float(), ele_groups, 'sum')
    return jcd


class JaccardAUC(BinaryAUROC):
    def __init__(self, has_feat=False, epsilon=0.1, name='jcdauc'):
        super().__init__(pos_label=1)
        self.has_feat = has_feat
        self.name = name
        self.matcher = Matcher(LabelSmoothingCrossEntropy(epsilon, reduction='none'))

    def forward(self, model_out, label):
        ele_preds, ele_targets, _, _, masks, cls_label = data_extract(model_out, label, True, self.has_feat)
        idx_p, idx_t = self.matcher(ele_preds, ele_targets, masks)
        preds, targets = ele_preds[idx_p], ele_targets[idx_t]
        jcds = jeccard(preds, targets, idx_p[0])
        return super().forward(jcds, cls_label.int())


class Jaccard(Metric):
    def __init__(self, has_feat=False, epsilon=0.1, name='jcd', aux=False):
        super().__init__()
        self.add_state("jaccards", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.has_element = True
        self.has_feat = has_feat
        self.name = name
        self.aux = aux
        self.matcher = Matcher(LabelSmoothingCrossEntropy(epsilon, reduction='none'))

    def update(self, model_out, label):
        if self.aux:
            model_out = model_out[1]
            label = label[:-1]
        ele_preds, ele_targets, _, _, masks, _ = data_extract(model_out, label, self.has_element, self.has_feat)
        idx_p, idx_t = self.matcher(ele_preds, ele_targets, masks)
        preds, targets = ele_preds[idx_p], ele_targets[idx_t]
        jcds = jeccard(preds, targets, idx_p[0])
        self.jaccards += jcds.sum()
        self.total += jcds.numel()

    def compute(self):
        return self.jaccards.float() / self.total


class CeAcc(BinaryAccuracy):
    def __init__(self, threshold,has_feat=False, epsilon=0.1, name='ceacc'):
        super().__init__(threshold=threshold)
        self.has_feat = has_feat
        self.name = name
        self.matcher = Matcher(LabelSmoothingCrossEntropy(epsilon, reduction='none'))
        self.ce_loss = LabelSmoothingCrossEntropy(epsilon, reduction='none')


    def forward(self, model_out, label):
        ele_preds, ele_targets, _, _, masks, cls_label = data_extract(model_out, label, True, self.has_feat)
        idx_p, idx_t = self.matcher(ele_preds, ele_targets, masks)

        losses = self.ce_loss(ele_preds[idx_p], ele_targets[idx_t])
        loss_mat = torch.zeros_like(masks).float()
        loss_mat[idx_p] = losses
        mean_loss_rev = masks.sum(1)/loss_mat.sum(1)       # 集合各元素平均损失的倒数
        # mean_loss_rev = masks.sum(1)/loss_mat.max(1)[0]  # 集合各元素最大损失除以元素数量的倒数

        return super().forward(F.sigmoid(mean_loss_rev), cls_label)


class CeF1(BinaryF1Score):
    def __init__(self, threshold, has_feat=False, epsilon=0.1, name='cef1'):
        super().__init__(threshold=threshold)
        self.has_feat = has_feat
        self.name = name
        self.matcher = Matcher(LabelSmoothingCrossEntropy(epsilon, reduction='none'))
        self.ce_loss = LabelSmoothingCrossEntropy(epsilon, reduction='none')


    def forward(self, model_out, label):
        ele_preds, ele_targets, _, _, masks, cls_label = data_extract(model_out, label, True, self.has_feat)
        idx_p, idx_t = self.matcher(ele_preds, ele_targets, masks)

        losses = self.ce_loss(ele_preds[idx_p], ele_targets[idx_t])
        loss_mat = torch.zeros_like(masks).float()
        loss_mat[idx_p] = losses
        mean_loss_rev = masks.sum(1)/loss_mat.sum(1)       # 集合各元素平均损失的倒数
        # mean_loss_rev = masks.sum(1)/loss_mat.max(1)[0]  # 集合各元素最大损失除以元素数量的倒数

        return super().forward(F.sigmoid(mean_loss_rev), cls_label)


class JeccardAcc(BinaryAccuracy):
    def __init__(self, threshold, has_feat=False, epsilon=0.1, name='jcdacc'):
        super().__init__(threshold=threshold)
        self.has_feat = has_feat
        self.name = name
        self.matcher = Matcher(LabelSmoothingCrossEntropy(epsilon, reduction='none'))

    def forward(self, model_out, label):
        ele_preds, ele_targets, _, _, masks, cls_label = data_extract(model_out, label, True, self.has_feat)
        idx_p, idx_t = self.matcher(ele_preds, ele_targets, masks)
        preds, targets = ele_preds[idx_p], ele_targets[idx_t]
        jcds = jeccard(preds, targets, idx_p[0])

        return super().forward(jcds, cls_label)


class JeccardF1(BinaryF1Score):
    def __init__(self, threshold, has_feat=False, epsilon=0.1, name='jcdf1'):
        super().__init__(threshold=threshold)
        self.has_feat = has_feat
        self.name = name
        self.matcher = Matcher(LabelSmoothingCrossEntropy(epsilon, reduction='none'))

    def forward(self, model_out, label):
        ele_preds, ele_targets, _, _, masks, cls_label = data_extract(model_out, label, True, self.has_feat)
        idx_p, idx_t = self.matcher(ele_preds, ele_targets, masks)
        preds, targets = ele_preds[idx_p], ele_targets[idx_t]
        jcds = jeccard(preds, targets, idx_p[0])

        return super().forward(jcds, cls_label)


class MseAcc(BinaryAccuracy):
    def __init__(self, threshold, name='mseacc'):
        super().__init__(threshold=threshold)
        self.name = name
        self.matcher = Matcher(nn.MSELoss(reduction='none'))
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, model_out, label):
        _, _, feat_preds, feat_targets, masks, cls_label = data_extract(model_out, label, False, True)
        idx_p, idx_t = self.matcher(feat_preds, feat_targets, masks)

        losses = self.mse_loss(feat_preds[idx_p], feat_targets[idx_t]).mean(dim=1)
        loss_mat = torch.zeros_like(masks).float()
        loss_mat[idx_p] = losses
        mean_loss_rev = masks.sum(1)/loss_mat.sum(1)       # 集合各元素平均损失的倒数
        # mean_loss_rev = masks.sum(1)/loss_mat.max(1)[0]  # 集合各元素最大损失除以元素数量的倒数

        return super().forward(F.sigmoid(mean_loss_rev), cls_label)


class MseF1(BinaryF1Score):
    def __init__(self, threshold, name='msef1'):
        super().__init__(threshold=threshold)
        self.name = name
        self.matcher = Matcher(nn.MSELoss(reduction='none'))
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, model_out, label):
        _, _, feat_preds, feat_targets, masks, cls_label = data_extract(model_out, label, False, True)
        idx_p, idx_t = self.matcher(feat_preds, feat_targets, masks)

        losses = self.mse_loss(feat_preds[idx_p], feat_targets[idx_t]).mean(dim=1)
        loss_mat = torch.zeros_like(masks).float()
        loss_mat[idx_p] = losses
        mean_loss_rev = masks.sum(1)/loss_mat.sum(1)       # 集合各元素平均损失的倒数
        # mean_loss_rev = masks.sum(1)/loss_mat.max(1)[0]  # 集合各元素最大损失除以元素数量的倒数

        return super().forward(F.sigmoid(mean_loss_rev), cls_label)

class AuxAcc(BinaryAccuracy):
    def __init__(self, name='aacc', test=False):
        super().__init__()
        self.name = name
        self.test = test
    
    def forward(self, model_out, label):
        cls_pred = model_out[0]
        cls_target = label[-1]
        if self.test:
            cls_target = cls_target.unsqueeze(-1)
        return super().forward(F.sigmoid(cls_pred), cls_target)


class AuxF1(BinaryF1Score):
    def __init__(self, name='af1', test=False):
        super().__init__()
        self.name = name
        self.test = test
    
    def forward(self, model_out, label):
        cls_pred = model_out[0]
        cls_target = label[-1]
        if self.test:
            cls_target = cls_target.unsqueeze(-1)
        return super().forward(F.sigmoid(cls_pred), cls_target)


class AuxAUC(BinaryAUROC):
    def __init__(self, name='aauc', test=False):
        super().__init__()
        self.name = name
        self.test = test
    
    def forward(self, model_out, label):
        cls_pred = model_out[0]
        cls_target = label[-1]
        if self.test:
            cls_target = cls_target.unsqueeze(-1)
        return super().forward(F.sigmoid(cls_pred), cls_target)