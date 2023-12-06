import torch
from torch import nn
from torch.nn import functional as F
import math


class SetMTransformer(nn.Module):
    def __init__(self, max_set_size, encoder_layer=2, decoder_layer=2, model_dim=128, num_heads=8,
                 dropout=.0, layer_norm=True, ele_query_method='embed'):
        super().__init__()

        self.encoder = nn.ModuleList([EncoderLayer(model_dim, num_heads, layer_norm, dropout) for _ in range(encoder_layer)])

        self.dec_query = ElementQuery(model_dim, max_set_size, ele_query_method)
        self.decoder = nn.ModuleList([DecoderLayer(model_dim, num_heads, layer_norm, dropout, query_att=True)
                                      for _ in range(decoder_layer)])

    def forward(self, input_set_feats, padd_mask):
        padd_mask = padd_mask.unsqueeze(1).unsqueeze(1)
        outputs = self.encode(input_set_feats, padd_mask)
        outputs = self.decode(outputs, padd_mask)
        return outputs

    def encode(self, input_set_feats, padd_mask):
        enc_out = input_set_feats
        for layer in self.encoder:
            enc_out = layer(enc_out, padd_mask)
        return enc_out

    def decode(self, enc_out, padd_mask):
        batch_size = padd_mask.shape[0]
        dec_out = self.dec_query(batch_size)
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, padd_mask)

        return dec_out

class Attention(nn.Module):
    def __init__(self, dim_QK, dim_V, num_heads, dropout=.0):
        super().__init__()
        assert dim_QK % num_heads == 0, "Mismatching Dimensions!"
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.head_dim_qk = dim_QK // num_heads
        self.head_dim_v = dim_V // num_heads

        self.fc_q = nn.Linear(dim_QK, dim_V)
        self.fc_k = nn.Linear(dim_QK, dim_V)
        self.fc_v = nn.Linear(dim_QK, dim_V)
        self.fc_out = nn.Linear(dim_V, dim_V)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        batch_size = Q.shape[0]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim_qk).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim_qk).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim_v).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_V)
        if mask is not None:
            scores = scores.masked_fill(mask == False, -1e9)
        atten = scores.softmax(dim=-1)

        atten = self.dropout(atten)

        Z = torch.matmul(atten, V)
        Z = Z.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_V)
        del Q
        del K
        del V
        return self.fc_out(Z)


class SkipConnection(nn.Module):
    def __init__(self, norm_dim=None, dropout=.0):
        super().__init__()
        self.layer_norm = None if norm_dim is None else nn.LayerNorm(norm_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, new_X):
        Z = self.dropout(new_X)
        Z = X + Z
        Z = Z if self.layer_norm is None else self.layer_norm(Z)
        return Z


class FeedForward(nn.Module):
    def __init__(self, model_dim, dropout=.0):
        super().__init__()
        self.ff_1 = nn.Linear(model_dim, 4*model_dim)
        self.ff_2 = nn.Linear(4*model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        Z = F.relu(self.ff_1(X))
        Z = self.dropout(Z)
        Z = F.relu(self.ff_2(Z))
        return Z


class EncoderLayer(nn.Module):  # SAB
    def __init__(self, model_dim, num_heads, layer_norm=False, dropout=.0):
        super().__init__()
        self.attention = Attention(model_dim, model_dim, num_heads, dropout)
        self.skip_att = SkipConnection(model_dim if layer_norm else None, dropout)
        self.ff = FeedForward(model_dim, dropout)
        self.skip_ff = SkipConnection(model_dim if layer_norm else None, dropout)

    def forward(self, X, mask=None):
        Z = self.attention(X, X, X, mask)
        Z = self.skip_att(X, Z)
        Z_ = self.ff(Z)
        Z = self.skip_ff(Z, Z_)
        return Z


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, layer_norm=False, dropout=None, query_att=True):
        super().__init__()
        self.query_att = query_att  # 解码器输入是否要做自注意力
        if query_att:
            self.attention_q = Attention(model_dim, model_dim, num_heads, dropout)
            self.skip_att_q = SkipConnection(model_dim if layer_norm else None, dropout)
        self.attention = Attention(model_dim, model_dim, num_heads, dropout)
        self.skip_att = SkipConnection(model_dim if layer_norm else None, dropout)
        self.ff = FeedForward(model_dim, dropout)
        self.skip_ff = SkipConnection(model_dim if layer_norm else None, dropout)

    def forward(self, X, memory, mask=None):
        if self.query_att:
            X_ = self.attention_q(X, X, X, mask)
            X = self.skip_att_q(X, X_)
        Z = self.attention(X, memory, memory, mask)
        Z = self.skip_att(X, Z)
        Z_ = self.ff(Z)
        Z = self.skip_ff(Z, Z_)
        return Z


class ElementQuery(nn.Module):
    def __init__(self, model_dim, max_set_size, query_method):
        super().__init__()
        assert query_method in ["l_embed", "embed_t", "p_embed", "p_embed_t"]
        self.query_method = query_method
        if query_method == "l_embed":
            self.query = nn.Parameter(torch.randn(max_set_size, model_dim))
            # nn.init.xavier_uniform_(self.dec_query)
            # nn.init.xavier_normal_(self.dec_query)
        elif query_method == "embed_t":
            self.query = nn.Parameter(torch.randn(max_set_size, model_dim), requires_grad=False)
        elif query_method in ["p_embed", "p_embed_t"]:
            query = torch.zeros(max_set_size, model_dim)
            position = torch.arange(0, max_set_size).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim))
            query[:, 0::2] = torch.sin(position * div_term)
            query[:, 1::2] = torch.cos(position * div_term)
            query = query.unsqueeze(0)
            self.register_buffer("query", query)

        if query_method in ["embed_t", "p_embed_t"]:
            self.ff = nn.Linear(model_dim, model_dim)
        else:
            self.ff = None

    def forward(self, batch_size):
        if self.query_method in ["l_embed", "embed_t"]:
            Z = self.query.expand(batch_size, -1, -1)
        elif self.query_method in ["p_embed", "p_embed_t"]:
            Z = self.query.requires_grad_(False).expand(batch_size, -1, -1)
        if self.ff is not None:
            Z = self.ff(Z)
        return Z
