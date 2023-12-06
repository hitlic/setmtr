from torch import nn
from modules import SetMTransformer

class Model(nn.Module):
    def __init__(self, element_num, max_set_size, feat_dim, element_embedding, padding_idx=None, has_element=True, has_feat=False,
        encoder_layer=2, decoder_layer=2, model_dim=128, num_heads=8, dropout=.0, layer_norm=True, element_query_method='embed'):
        super().__init__()

        assert feat_dim>0 or element_embedding, "the model need inputs of element or feat or both."
        assert has_element or has_feat, "the model need outputs of element or feat or both."

        self.element_embedding = element_embedding
        self.feat_dim = feat_dim
        self.embed = None
        self.feat_trans = None
        self.element_pred = None
        self.feat_pred = None

        if element_embedding:
            assert padding_idx is not None, "need padding index!"
            # 倒数第二个id用于表示padding符号，最后一个id用于表示mask符号
            self.embed = nn.Embedding(element_num+2, model_dim, padding_idx=padding_idx)

        if feat_dim > 0:
            self.feat_trans = nn.Sequential(nn.Linear(feat_dim, model_dim), nn.ReLU())

        self.setmtr = SetMTransformer(max_set_size, encoder_layer, decoder_layer, model_dim, num_heads, dropout, layer_norm, element_query_method)

        if has_element:
            self.element_pred = nn.Sequential(nn.ReLU(), nn.Linear(model_dim, element_num+1), nn.Dropout(dropout))

        if has_feat:
            self.feat_pred = nn.Sequential(nn.ReLU(), nn.Linear(model_dim, feat_dim), nn.Dropout(dropout))

    def forward(self, *inputs):
        outputs, padd_mask = self.pre_feature_transform(*inputs)
        outputs = self.setmtr(outputs, padd_mask)
        return self.reconstruct(outputs)

    def aux_forward(self, *inputs):
        """
        同时输出编码器结果和重构结果
        """
        outputs, padd_mask = self.pre_feature_transform(*inputs)
        padd_mask_ = padd_mask.unsqueeze(1).unsqueeze(1)
        enc_out = self.setmtr.encode(outputs, padd_mask_)
        dec_out = self.setmtr.decode(enc_out, padd_mask_)
        return enc_out, self.reconstruct(dec_out), padd_mask

    def reconstruct(self, dec_out):
        if self.element_pred and self.feat_pred:
            return self.element_pred(dec_out), self.feat_pred(dec_out)
        elif self.element_pred:
            return self.element_pred(dec_out)
        else:
            return self.feat_pred(dec_out)

    def pre_feature_transform(self, *inputs):
        if self.element_embedding and self.feat_dim >0:
            elements, ele_feats, padd_mask = inputs
            outputs = self.embed(elements) + self.feat_trans(ele_feats)
        elif self.element_embedding:
            elements, padd_mask = inputs
            outputs = self.embed(elements)
        else:
            ele_feats, padd_mask = inputs
            outputs = self.feat_trans(ele_feats)
        return outputs, padd_mask


class ModelAux(nn.Module):
    def __init__(self, reconstruct_model, model_dim):
        super().__init__()
        self.rec_model = reconstruct_model
        self.cls_model = nn.Sequential(nn.ReLU(), nn.Linear(model_dim, 1))

    def forward(self, *inputs):
        enc_out, rec_out, padd_mask = self.rec_model.aux_forward(*inputs)
        enc_feats = enc_out * padd_mask.float().unsqueeze(-1)
        enc_feats = enc_feats.sum(1)
        return self.cls_model(enc_feats), rec_out
