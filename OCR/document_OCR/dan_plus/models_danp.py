#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import relu, softmax
from torch.nn import Conv1d, Dropout,  Linear, LayerNorm
from torch.nn import LSTM, Embedding
from torch.nn import ModuleList, Module
import math
from torch.nn.init import xavier_uniform_


class PositionalEncoding1D(Module):

    def __init__(self, dim, len_max, device):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, len_max), device=device, requires_grad=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)
        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)

    def forward(self, x, start):
        """
        Add 1D positional encoding to x
        x: (B, C, L)
        start: index for x[:,:, 0]
        """
        if isinstance(start, int):
            x[:, :, start:] = x[:, :, start:] + self.pe[0, :, :x.size(2) - start]
            return x
        else:
            for i in range(x.size(0)):
                x[i, :, start[i]:] = x[i, :, start[i]:] + self.pe[0, :, :x.size(2)-start[i]]
            return x


class PositionalEncoding2D(Module):

    def __init__(self, dim, h_max, w_max, device):
        super(PositionalEncoding2D, self).__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, h_max, w_max), device=device, requires_grad=False)

        div = torch.exp(-torch.arange(0., dim // 2, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        w_pos = torch.arange(0., w_max)
        h_pos = torch.arange(0., h_max)
        self.pe[:, :dim // 2:2, :, :] = torch.sin(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, 1:dim // 2:2, :, :] = torch.cos(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, dim // 2::2, :, :] = torch.sin(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        self.pe[:, dim // 2 + 1::2, :, :] = torch.cos(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

    def get_pe_by_size(self, h, w, device):
        return self.pe[:, :, :h, :w].to(device)


class CustomMultiHeadAttention(Module):
    """
    Re-implementation of Multi-head Attention
    """
    def __init__(self, embed_dim, num_heads, dropout=0, proj_value=True):
        super().__init__()

        self.proj_value = proj_value

        self.in_proj_q = Linear(embed_dim, embed_dim)
        self.in_proj_k = Linear(embed_dim, embed_dim)
        if self.proj_value:
            self.in_proj_v = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale_factor = float(self.head_dim) ** -0.5
        self.dropout = Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, output_weights=True):
        target_len, b, c = query.size()
        source_len = key.size(0)
        q = self.in_proj_q(query)
        k = self.in_proj_k(key)
        v = self.in_proj_v(value) if self.proj_value else value
        q = q * self.scale_factor

        q = torch.reshape(q, (target_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        k = torch.reshape(k, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        v = torch.reshape(v, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_output_weights = attn_output_weights.view(b, self.num_heads, target_len, source_len)
            attn_mask = attn_mask.unsqueeze(1)
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask
            attn_output_weights = attn_output_weights.view(b * self.num_heads, target_len, source_len)

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(b, self.num_heads, target_len, source_len)

            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(b * self.num_heads, target_len, source_len)

        attn_output_weights_raw = softmax(attn_output_weights, dim=-1)

        attn_output_weights = self.dropout(attn_output_weights_raw)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(target_len, b, c)
        attn_output = self.out_proj(attn_output)

        if output_weights:
            attn_output_weights_raw = attn_output_weights_raw.view(b, self.num_heads, target_len, source_len)
            return attn_output, attn_output_weights_raw.sum(dim=1) / self.num_heads
        return attn_output

    def init_weights(self):
        xavier_uniform_(self.in_proj_q.weight)
        xavier_uniform_(self.in_proj_k.weight)
        if self.proj_value:
            xavier_uniform_(self.in_proj_v.weight)


class GlobalDecoderLayer(Module):
    """
    Transformer Decoder Layer
    """

    def __init__(self, params):
        super(GlobalDecoderLayer, self).__init__()
        self.emb_dim = params["enc_dim"]
        self.dim_feedforward = params["dec_dim_feedforward"]

        self.self_att = CustomMultiHeadAttention(embed_dim=self.emb_dim,
                                                  num_heads=params["dec_num_heads"],
                                                  proj_value=True,
                                                  dropout=params["dec_att_dropout"])

        self.norm1 = LayerNorm(self.emb_dim)
        self.att = CustomMultiHeadAttention(embed_dim=self.emb_dim,
                                                  num_heads=params["dec_num_heads"],
                                                  proj_value=True,
                                                  dropout=params["dec_att_dropout"])

        self.linear1 = Linear(self.emb_dim, self.dim_feedforward)
        self.linear2 = Linear(self.dim_feedforward, self.emb_dim)

        self.res_dropout = Dropout(params.get("dec_res_dropout", 0.))
        self.lin_dropout = Dropout(params.get('dec_lin_dropout', 0.))

        self.norm2 = LayerNorm(self.emb_dim)
        self.norm3 = LayerNorm(self.emb_dim)

    def forward(self, tgt, memory_key, memory_value=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, predict_last_n_only=None):

        if memory_value is None:
            memory_value = memory_key

        self_att_query = tgt[-predict_last_n_only:] if predict_last_n_only else tgt

        tgt2, weights_self = self.self_att(self_att_query, tgt, tgt, attn_mask=tgt_mask,
                                           key_padding_mask=tgt_key_padding_mask, output_weights=True)
        tgt = self_att_query + self.res_dropout(tgt2)
        tgt = self.norm1(tgt)
        att_query = tgt

        tgt2, weights = self.att(att_query, memory_key, memory_value, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask, output_weights=True)

        tgt = att_query + self.res_dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.lin_dropout(relu(self.linear1(tgt))))
        tgt = tgt + self.res_dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt, weights, weights_self


class GlobalAttDecoder(Module):
    """
    Stack of transformer decoder layers
    """

    def __init__(self, params):
        super(GlobalAttDecoder, self).__init__()

        self.decoder_layers = ModuleList([GlobalDecoderLayer(params) for _ in range(params["dec_num_layers"])])

    def forward(self, tgt, memory_key, memory_value, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                use_cache=False, cache=None, predict_last_n_only=False, keep_all_weights=False):
        output = tgt
        cache_t = list()
        all_weights = {
            "self": list(),
            "mix": list()
        }

        for i, dec_layer in enumerate(self.decoder_layers):
            output, weights, weights_self = dec_layer(output, memory_key=memory_key,
                                        memory_value=memory_value,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,  # limits the self-attn with window size in decoder
                                        memory_key_padding_mask=memory_key_padding_mask, # mask that excludes paddings in 2D feature map
                                        predict_last_n_only=predict_last_n_only)
            if use_cache:
                cache_t.append(output)
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)
            if keep_all_weights:
                all_weights["self"].append(weights_self)
                all_weights["mix"].append(weights)

        if use_cache:
            cache = torch.cat([cache, torch.stack(cache_t, dim=0)], dim=1) if cache is not None else torch.stack(cache_t, dim=0)

        if predict_last_n_only:
            output = output[-predict_last_n_only:]

        if keep_all_weights:
            return output, all_weights, cache

        return output, weights, cache


class FeaturesUpdater(Module):
    """
    Module that handle 2D positional encoding
    """
    def __init__(self, params):
        super(FeaturesUpdater, self).__init__()
        self.enc_dim = params["enc_dim"]
        self.enc_h_max = params["h_max"]
        self.enc_w_max = params["w_max"]
        self.pe_2d = PositionalEncoding2D(self.enc_dim, self.enc_h_max, self.enc_w_max, params["device"])
        self.use_2d_positional_encoding = "use_2d_pe" not in params or params["use_2d_pe"]

    def get_pos_features(self, features):
        if self.use_2d_positional_encoding:
            return self.pe_2d(features)
        return features


class GlobalHTADecoder(Module):
    """
    DAN decoder module
    """
    def __init__(self, params):
        super(GlobalHTADecoder, self).__init__()
        self.enc_dim = params["enc_dim"]
        self.dec_l_max = params["l_max"]

        self.dropout = Dropout(params["dec_pred_dropout"])
        # self.dec_attn_win = params["attention_win"] if params["attention_win"] is not None else 1
        if isinstance(params["attention_win"], dict):
            self.dec_attn_win = params['attention_win']['size']
            self.non_causal_global = params['attention_win'].get('non_causal_global', False)
            # self.dec_attn_win_max = params['attention_win']['max_size']
        else:
            self.dec_attn_win = params["attention_win"]
            self.non_causal_global = False
            # self.dec_attn_win_max = 500

        self.use_1d_pe = "use_1d_pe" not in params or params["use_1d_pe"]
        self.use_lstm = params["use_lstm"]

        self.features_updater = FeaturesUpdater(params)
        self.att_decoder = GlobalAttDecoder(params)

        num_emb = params["vocab_size"]+3 # class + end + skip(optional) + start + pad
        if 'skip' in params:
            if 'paragraph' in params['skip']:
                num_emb += 1
            if 'line_break' in params['skip']:
                num_emb += 1
            if 'word' in params['skip']:
                num_emb += 1

        self.emb = Embedding(num_embeddings=num_emb, embedding_dim=self.enc_dim)
        self.pe_1d = PositionalEncoding1D(self.enc_dim, self.dec_l_max, params["device"])

        if self.use_lstm:
            self.lstm_predict = LSTM(self.enc_dim, self.enc_dim)

        vocab_size = params["vocab_size"] + 1
        self.end_conv = Conv1d(self.enc_dim, vocab_size, kernel_size=1)

        self.gc_tokens = params['global_context_tokens'] if 'global_context_tokens' in params else []

        self.pred_relu = params.get('dec_pred_relu', True)



    def forward(self, raw_features_1d, enhanced_features_1d, tokens, reduced_size, features_size, start=0, hidden_predict=None, cache=None, cache_mask=None, num_pred=None, keep_all_weights=False):
        device = raw_features_1d.device

        # Token to Embedding
        pos_tokens = self.emb(tokens).permute(0, 2, 1)

        # Add 1D Positional Encoding
        if self.use_1d_pe:
            pos_tokens = self.pe_1d(pos_tokens, start=start)
        pos_tokens = pos_tokens.permute(2, 0, 1) # (nb_tokens, batch_size, feature_size)

        if num_pred is None:
            num_pred = tokens.size(1)

        target_mask = self.generate_target_mask(tokens, start, device) # Use only already predicted tokens (causal)

        memory_mask = None  # Use all feature position

        # Generate static masks
        """这个好像没有作用"""
        # key_target_mask = self.generate_token_mask(token_len, tokens.size(), device)  # Use all token except padding
        key_target_mask = None
        key_memory_mask = self.generate_enc_mask(reduced_size, features_size, device)  # Use all feature position except padding

        cache_full = None

        if num_pred == 1:  # prediction
            # if tokens.shape[-1] in [2, 101, 201]:
            #     print('debug')
            tokens_to_keep = torch.logical_not(torch.prod(target_mask[:, -1, :], dim=0, dtype=torch.bool))  # and operation

            target_mask_full = target_mask
            target_mask = [target_mask[b, -1:, tokens_to_keep] for b in range(tokens.size(0))]
            target_mask = torch.stack(target_mask, dim=0)
            pos_tokens = pos_tokens[tokens_to_keep]
            """kv_caches: [(nb_layers, nb_tokens-1, 1, feature_size)] * batch_size"""
            """cache_mask: (batch_size, nb_tokens-1)"""
            if cache is not None:
                # cache_full = cache
                cache_full = torch.zeros((cache[0].shape[0], tokens.shape[-1]-1, tokens.shape[0], cache[0].shape[-1]),
                                         dtype=cache[0].dtype, device=cache[0].device) # ()
                # mask = cache_mask & torch.logical_not(target_mask_full[:, -1, :-1])
                for b in range(tokens.shape[0]):
                    cache_full[:, cache_mask[b], b, :] = cache[b]
                # cache_old = cache
                cache = cache_full[:, tokens_to_keep[:-1], :, :]

        output, weights, cache = self.att_decoder(pos_tokens, memory_key=enhanced_features_1d,
                                                    memory_value=raw_features_1d,
                                                    tgt_mask=target_mask,
                                                    memory_mask=memory_mask,
                                                    tgt_key_padding_mask=key_target_mask,
                                                    memory_key_padding_mask=key_memory_mask,
                                                    use_cache=True,
                                                    cache=cache,
                                                    predict_last_n_only=num_pred,
                                                    keep_all_weights=keep_all_weights)

        if num_pred == 1: # prediction
            cache_new = []
            cache_mask = torch.logical_not(target_mask_full[:, -1, :])
            for b in range(tokens.size(0)):
                if cache_full is not None:
                    cache_new.append(torch.cat([cache_full[:, cache_mask[b, :-1], b, :], cache[:, -1:, b, :]], dim=1))
                else:
                    cache_new.append(cache[:, -1:, b, :])

        else:
            cache_new = cache
            cache_mask = None

        if self.use_lstm:
            output, hidden_predict = self.lstm_predict(output, hidden_predict)

        if self.pred_relu:
            output = relu(output)
        dp_output = self.dropout(output)

        preds = self.end_conv(dp_output.permute(1, 2, 0))

        """Debug"""
        if keep_all_weights:
            weights['mix'] = [torch.sum(w, dim=1, keepdim=True).reshape(-1, 1, features_size[2], features_size[3]) for w in weights['mix']]

        if not keep_all_weights:
            weights = torch.sum(weights, dim=1, keepdim=True).reshape(-1, 1, features_size[2], features_size[3])
        return output, preds, hidden_predict, cache_new, cache_mask, weights

    def generate_enc_mask(self, batch_reduced_size, total_size, device):
        """
        Generate mask for encoded features
        """
        batch_size = len(batch_reduced_size)
        _, _, h_max, w_max = total_size
        mask = torch.ones((batch_size, h_max, w_max), dtype=torch.bool, device=device)
        for i, (h, w) in enumerate(batch_reduced_size):
            mask[i, :h, :w] = False
        return torch.flatten(mask, start_dim=1, end_dim=2)

    def generate_token_mask(self, token_len, total_size, device):
        """
        Generate mask for tokens per sample
        """
        batch_size, len_max = total_size
        mask = torch.zeros((batch_size, len_max), dtype=torch.bool, device=device)
        for i, len_ in enumerate(token_len):
            mask[i, :len_] = False
        return mask

    def generate_target_mask(self, tokens, start, device):
        """
        Generate mask for tokens per time step (teacher forcing)
        """
        target_len = tokens.size(1)

        if self.dec_attn_win < 1: # global_context
            mask = torch.tril(torch.ones((tokens.size(0), target_len, target_len), dtype=torch.bool, device=device), diagonal=0)
        else:
            mask = torch.logical_and(torch.tril(torch.ones((tokens.size(0), target_len, target_len), dtype=torch.bool, device=device), diagonal=0),
                                     torch.triu(torch.ones((tokens.size(0), target_len, target_len), dtype=torch.bool, device=device), diagonal=-self.dec_attn_win + 1))
        if len(self.gc_tokens):
            gc_mask = torch.zeros((tokens.size(0), target_len), dtype=torch.bool, device=device)
            for gct in self.gc_tokens:
                gc_mask[tokens == gct] = True

            if self.non_causal_global:
                mask = torch.logical_or(mask, gc_mask.unsqueeze(dim=1).expand(-1, target_len, -1))
            else:
                mask = torch.logical_or(mask, torch.tril(gc_mask.unsqueeze(dim=1).expand(-1, target_len, -1), diagonal=0))
            # mask_[torch.sum(mask_[:, -1, :], dim=1) > self.dec_attn_win_max] = mask[torch.sum(mask_[:, -1, :], dim=1) > self.dec_attn_win_max] # resort to local window when the actual size exceeds a thershold
            # mask = mask_

        # if isinstance(start, list):
        #     for i in range(tokens.size(0)):
        #         mask[i, :start[i], :] = mask[i, :, :start[i]] = False
        return torch.logical_not(mask)


