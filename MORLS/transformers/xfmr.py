# Transformer Components Implementation Adapted from Annotated Transformer:
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
import math

import torch
from torch import nn
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # small change here -- we use "1" for masked element
        scores = scores.masked_fill(mask > 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.d_model = d_model
        self.output_linear = output_linear

        if output_linear:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)] + [nn.Linear(d_model, d_model), ])
        else:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)])
        #for i in range(len(self.linears)):
            #nn.init.xavier_uniform_(self.linears[i].weight)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            l(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            return self.linears[-1](x)
        else:
            return x


class SublayerConnection(nn.Module):
    # used for residual connnection
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward=None, use_residual=False, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual
        if use_residual:
            self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask):
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            return self.self_attn(x, x, x, mask)

class XFMREncoder(nn.Module):
    def __init__(self, d_model, num_layers, self_attn, feed_forward, use_residual=False, dropout=0.1):
        super(XFMREncoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, self_attn, feed_forward, use_residual, dropout)
             for _ in range(num_layers)
             ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

"""
 modifying the ANHP transformer to make it suitable for RL dynamics model learning
 The structure of the CTxfmr is: input=[embedded time, embedded type, x] where type is either dosage(d) or measurements(m), and x is the actual value
 xfmr outputs prediction x. Note that although the data set is usually [m,d,m,d...], that is, a series of sets of (m,d)_t, it doesn't have to be the case
 since the types are embedded and the xfmr doesn't really care the order of events. This fact makes our model learning more flexible.
"""

class XFMRCT(nn.Module):
    def __init__(self, d_model, n_layers, n_head, dropout, d_time,d_measure=1,num_types=3,pad_index=2, hidden_num=256, d_inner=128, use_norm=False,
                 sharing_param_layer=False):
        # d_inner only used if we want to add feedforward
        # num_types and pad_index (for temporal embeddings) are added in for MOBO; used to be dataset fields
        # d_measure is the dimension of the measurement/dosage values. default to 1
        # values are not embedded and used rather directly, though embedding them might also be viable


        #questions: 1. why is mask in the original code flipped? i.e., unmasked elements are 0 here (not that it matters for us but that's strange)
        # 2. why do we need padding here?
        super(XFMRCT, self).__init__()
        self.d_model = d_model
        self.d_time = d_time
        self.d_measure=d_measure
        self.pad_index=pad_index
        self.div_term = torch.exp(torch.arange(0, d_time, 2) * -(math.log(10000.0) / d_time)).reshape(1, 1, -1)
        # here num_types already includes [PAD], [BOS], [EOS]
        self.Emb = nn.Embedding(num_types, d_model, padding_idx=pad_index)
        self.n_layers = n_layers
        self.n_head = n_head
        self.sharing_param_layer = sharing_param_layer
        # try two layers for now
        self.outnn1=nn.Linear(d_model * n_head, hidden_num)
        self.outnn2=nn.Linear(hidden_num, d_measure)
        self.swish = Swish()
        if not sharing_param_layer:
            self.heads = []
            for i in range(n_head):
                self.heads.append(
                    nn.ModuleList(
                        [EncoderLayer(
                            d_model + d_time + d_measure,
                            MultiHeadAttention(1, d_model + d_time + d_measure, d_model, dropout, output_linear=False),
                            # PositionwiseFeedForward(d_model + d_time, d_inner, dropout),
                            use_residual=False,
                            dropout=dropout
                        )
                            for _ in range(n_layers)
                        ]
                    )
                )
            self.heads = nn.ModuleList(self.heads)
        else:
            self.heads = []
            for i in range(n_head):
                self.heads.append(
                    nn.ModuleList(
                        [EncoderLayer(
                            d_model + d_time + d_measure,
                            MultiHeadAttention(1, d_model + d_time + d_measure, d_model, dropout, output_linear=False),
                            # PositionwiseFeedForward(d_model + d_time, d_inner, dropout),
                            use_residual=False,
                            dropout=dropout
                        )
                            for _ in range(0)
                        ]
                    )
                )
            self.heads = nn.ModuleList(self.heads)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(d_model)
        # self.inten_linear = nn.Linear(d_model * n_head, dataset.event_num)
        # self.softplus = nn.Softplus()
        # self.eps = torch.finfo(torch.float32).eps
        # # self.add_bos = dataset.add_bos
        # self.add_bos = True

    def compute_temporal_embedding(self, time):
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.d_time).to(time.device)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time.device)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)
        # pe = pe * non_pad_mask.unsqueeze(-1)
        return pe

    def forward_pass(self, init_cur_layer_, tem_enc, x_seqs, tem_enc_layer, enc_input, combined_mask, batch_non_pad_mask=None):
        cur_layers = []
        seq_len = enc_input.size(1)
        for head_i in range(self.n_head):
            cur_layer_ = init_cur_layer_
            for layer_i in range(self.n_layers):
                # print('before concat curlayer:{}'.format(cur_layer_.shape))
                # print('before concat temenclayer:{}'.format(tem_enc_layer.shape))
                layer_ = torch.cat([cur_layer_, tem_enc_layer,x_seqs], dim=-1)
                # print('concated layer:{}'.format(layer_.shape))
                # print('encinputbeforecombine:{}'.format(enc_input.shape))
                _combined_input = torch.cat([enc_input, layer_], dim=1)
                #print('concated combinein:{}'.format(_combined_input.shape))
                if self.sharing_param_layer:
                    enc_layer = self.heads[head_i][0]
                else:
                    enc_layer = self.heads[head_i][layer_i]
                enc_output = enc_layer(
                    _combined_input,
                    combined_mask
                )
                if batch_non_pad_mask is not None:
                    _cur_layer_ = enc_output[:, seq_len:, :] * (batch_non_pad_mask.unsqueeze(-1))
                else:
                    _cur_layer_ = enc_output[:, seq_len:, :]

                # add residual connection
                cur_layer_ = torch.tanh(_cur_layer_) + cur_layer_
                # print('cur_layer_residual:{}'.format(cur_layer_.shape))
                # print('encout:{}'.format(enc_output.shape))
                # print('temenc:{}'.format(tem_enc.shape))
                # print('xseqs:{}'.format(x_seqs.shape))
                enc_input = torch.cat([enc_output[:, :seq_len, :], tem_enc, x_seqs], dim=-1)
                # non-residual connection
                # cur_layer_ = torch.tanh(_cur_layer_)

                # enc_output *= _combined_non_pad_mask.unsqueeze(-1)
                # layer_ = torch.tanh(enc_output[:, enc_input.size(1):, :])
                if self.use_norm:
                    cur_layer_ = self.norm(cur_layer_)
            cur_layers.append(cur_layer_)
        cur_layer_ = torch.cat(cur_layers, dim=-1)

        return cur_layer_

    # putting the mask creator in this class for now
    # and modifying it so that it works for single samples (instead of batches)
    def createPadAttnMask(self, event_seq, concurrent_mask=None):
        # 1 -- pad, 0 -- non-pad
        batch_size, seq_len = event_seq.size(0), event_seq.size(1)
        batch_seq_pad_mask = event_seq.eq(self.pad_index)
        attention_key_pad_mask = batch_seq_pad_mask.unsqueeze(1).expand(batch_size, seq_len, -1)
        subsequent_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=event_seq.device, dtype=torch.uint8), diagonal=0
        ).unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = subsequent_mask | attention_key_pad_mask.bool()
        if concurrent_mask is None:
            # no way to judge concurrent events, simply believe there is no concurrent events
            pass
        else:
            attention_mask |= concurrent_mask.bool()
        return ~batch_seq_pad_mask, attention_mask


    def forward(self, event_seqs, time_seqs, x_seqs, batch_non_pad_mask, attention_mask, extra_times=None):
        #trivial embedding for x_seqs
        #[batch size, len, 1]
        x_seqs=x_seqs.unsqueeze(-1)
        tem_enc = self.compute_temporal_embedding(time_seqs)
        tem_enc *= batch_non_pad_mask.unsqueeze(-1)
        enc_input = torch.tanh(self.Emb(event_seqs))
        init_cur_layer_ = torch.zeros_like(enc_input)
        layer_mask = (torch.eye(attention_mask.size(1)) < 1).unsqueeze(0).expand_as(attention_mask).to(
            attention_mask.device)
        if extra_times is None:
            tem_enc_layer = tem_enc
        else:
            tem_enc_layer = self.compute_temporal_embedding(extra_times)
            tem_enc_layer *= batch_non_pad_mask.unsqueeze(-1)
        # batch_size * (seq_len) * (2 * seq_len)
        _combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)
        # batch_size * (2 * seq_len) * (2 * seq_len)
        contextual_mask = torch.cat([attention_mask, torch.ones_like(layer_mask)], dim=-1)
        _combined_mask = torch.cat([contextual_mask, _combined_mask], dim=1)
        #print(enc_input.shape,tem_enc.shape,x_seqs.shape)
        enc_input = torch.cat([enc_input, tem_enc,x_seqs], dim=-1)
        #print('concated enc input:{}'.format(enc_input.shape))
        #[batch size, seq_len, d_model]
        cur_layer_ = self.forward_pass(init_cur_layer_, tem_enc, x_seqs, tem_enc_layer, enc_input, _combined_mask, batch_non_pad_mask)
        #get all hidden states
        xout=self.swish(self.outnn1(cur_layer_))
        xout=self.outnn2(xout)
        #[batch seq_len]
        return xout
