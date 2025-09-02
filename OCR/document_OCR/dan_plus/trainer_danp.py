from OCR.ocr_manager import OCRManager
from torch.nn import CrossEntropyLoss
import torch
from OCR.ocr_utils import LM_ind_to_str
import numpy as np
from torch.cuda.amp import autocast
import time
import os
from basic.utils import randint, rand
import random



def get_segment(tokens, chr_token):
    runs = []
    is_content = []
    i_ = 0
    for i in range(len(tokens)):
        if i > 0 and ((tokens[i].item() in chr_token) ^ (tokens[i-1].item() in chr_token)):
            runs.append(tokens[i_: i])
            is_content.append(tokens[i-1].item() in chr_token)
            i_ = i
    if i_ < len(tokens):
        runs.append(tokens[i_:])
        is_content.append(tokens[i_].item() in chr_token)
    lens = [len(r) for r in runs]
    assert np.sum(lens) == tokens.size(0)
    is_content = torch.tensor(is_content, dtype=torch.bool, device=tokens.device)
    return runs, is_content





class Manager(OCRManager):

    def __init__(self, params):
        super(Manager, self).__init__(params)

        if isinstance(params['model_params']['attention_win'], dict):
            gc_tokens = [] # global context tokens
            if 'sem' in params['model_params']['attention_win'] and params['model_params']['attention_win']['sem']:
                gc_tokens += self.dataset.tokens['sem']
            if 'line_break' in params['model_params']['attention_win'] and params['model_params']['attention_win']['line_break']:
                gc_tokens += [self.dataset.tokens['lb']]
            if 'start' in params['model_params']['attention_win'] and params['model_params']['attention_win']['start']:
                gc_tokens += [self.dataset.tokens['start']]
            gc_tokens = sorted(gc_tokens)
            self.params['model_params']['global_context_tokens'] = gc_tokens


    def load_save_info(self, info_dict):
        if "curriculum_config" in info_dict.keys():
            if self.dataset.train_dataset is not None:
                self.dataset.train_dataset.curriculum_config = info_dict["curriculum_config"]

    def add_save_info(self, info_dict):
        info_dict["curriculum_config"] = self.dataset.train_dataset.curriculum_config
        return info_dict

    def get_init_hidden(self, batch_size):
        num_layers = 1
        hidden_size = self.params["model_params"]["enc_dim"]
        return torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(num_layers, batch_size, hidden_size)


    def insert_skip(self, y, y_len, config):
        y_inp = []
        y_target = []

        skips = []
        if 'skip_ph' in self.dataset.tokens:
            skips.append('skip_ph')
        if 'skip_lb' in self.dataset.tokens:
            skips.append('skip_lb')
        if 'skip_wd' in self.dataset.tokens:
            skips.append('skip_wd')

        layout_tokens = []
        if 'sem' in self.dataset.tokens and len(self.dataset.tokens['sem']) > 0:
            layout_tokens += self.dataset.tokens['sem']
        if '\n' in self.dataset.class_set:
            layout_tokens += [self.dataset.class_set.index('\n')]
        # if ' ' in self.dataset.class_set:
        #     layout_tokens += [self.dataset.class_set.index(' ')]

        for b in range(y.size(0)):
            inp = []
            tar = []
            pos = randint(1, y_len[b])
            if y[b, pos] < len(self.dataset.class_set) and y[b, pos] not in layout_tokens:
                inp += [y[b, :pos]]
                tar += [y[b, 1:pos+1]]
                skip_type = skips[randint(0, len(skips))]
                for j in range(pos+1, y_len[b]+1):
                    if skip_type == 'skip_ph' and (y[b, j] in self.dataset.tokens['sem'] or y[b, j] == self.dataset.tokens['end']):
                        inp += [torch.tensor(self.dataset.tokens['skip_ph'], dtype=y.dtype, device=y.device).unsqueeze(dim=0)]
                        break
                    if skip_type == 'skip_lb' and (y[b, j] == self.dataset.tokens['lb'] or y[b, j] in self.dataset.tokens['sem'] or y[b, j] == self.dataset.tokens['end']):
                        inp += [torch.tensor(self.dataset.tokens['skip_lb'], dtype=y.dtype, device=y.device).unsqueeze(dim=0)]
                        break
                    if skip_type == 'skip_wd' and (y[b, j] in layout_tokens or y[b, j] == self.dataset.tokens['end']):
                        inp += [torch.tensor(self.dataset.tokens['skip_wd'], dtype=y.dtype, device=y.device).unsqueeze(dim=0)]
                        break
                inp += [y[b, j:y_len[b] + 1]]
                tar += [y[b, j:y_len[b] + 1]]
                assert len(inp) == 3
                y_inp.append(torch.cat(inp, dim=0))
                y_target.append(torch.cat(tar, dim=0))
            else:
                y_inp.append(y[b, :y_len[b]+1])
                y_target.append(y[b, 1:y_len[b]+1])
            # inp_tar = self.pad_tokens_1D([y_target[-1], y_inp[-1]], self.dataset.tokens['pad'], before=False).cpu().numpy()

        y_len = [t.size(0) for t in y_target]
        y_inp = self.pad_tokens_1D(y_inp, self.dataset.tokens['pad'], before=False)
        y_target = self.pad_tokens_1D(y_target, self.dataset.tokens['pad'], before=False)

        return y_inp, y_target, y_len




    def apply_teacher_forcing(self, y, y_len, error_rate):
        y_error = y.clone()
        for b in range(len(y_len)):
            for i in range(1, y_len[b]):
                if np.random.rand() < error_rate and y[b][i] <= self.dataset.tokens["end"]:
                    y_error[b][i] = np.random.randint(0, len(self.dataset.class_set) + 1) # class + end
        # y_target = y[:, 1:]
        return y_error


    def train_batch(self, batch_data, metric_names):
        loss_func = CrossEntropyLoss(ignore_index=self.dataset.tokens["pad"])

        sum_loss = 0
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        if 'skip' in self.params['model_params'] and rand() < self.params['model_params']['skip']['proba']:
            y, y_target, y_len = self.insert_skip(y, y_len, self.params['model_params']['skip'])
        else:
            y_target = y[:, 1:]

        # add errors in teacher forcing
        if "teacher_forcing_error_rate" in self.params["training_params"] and self.params["training_params"]["teacher_forcing_error_rate"] is not None:
            error_rate = self.params["training_params"]["teacher_forcing_error_rate"]
            simulated_y_pred = self.apply_teacher_forcing(y, y_len, error_rate)
        elif "teacher_forcing_scheduler" in self.params["training_params"]:
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred = self.apply_teacher_forcing(y, y_len, error_rate)
        else:
            simulated_y_pred = y

        with autocast(enabled=self.params["training_params"]["use_amp"]):
            hidden_predict = None
            cache = None

            raw_features = self.models["encoder"](x)
            if y.size(0) > raw_features.size(0):
                raw_features = torch.cat([raw_features, raw_features], dim=0)
                reduced_size += reduced_size
            features_size = raw_features.size()
            b, c, h, w = features_size

            pos_features = self.models["decoder"].features_updater.get_pos_features(raw_features)
            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            enhanced_features = pos_features
            enhanced_features = torch.flatten(enhanced_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            output, pred, hidden_predict, cache, cache_mask, weights = self.models["decoder"](features, enhanced_features,
                                                                                              simulated_y_pred[:, :-1],
                                                                                              reduced_size,
                                                                                              # [max(y_len) for _ in range(b)],
                                                                                              features_size,
                                                                                               start=0,
                                                                                               hidden_predict=hidden_predict,
                                                                                               cache=cache,
                                                                                               keep_all_weights=False)

            loss_ce = loss_func(pred, y_target)
            sum_loss += loss_ce
            with autocast(enabled=False):
                self.backward_loss(sum_loss)
                self.step_optimizers()
                self.zero_optimizers()
            predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
            predicted_tokens = [predicted_tokens[i, :y_len[i]] for i in range(b)]
            return_list = self.params['dataset_params']['lan'] == 'bo' and not self.params['dataset_params'].get('use_comp', False)
            str_x = [LM_ind_to_str(self.dataset.class_set, t, oov_symbol="", return_list=return_list) for t in predicted_tokens]
            y_target = [y_target.detach().cpu().numpy()[i, :y_len[i]] for i in range(b)]
            str_y = [LM_ind_to_str(self.dataset.class_set, t, oov_symbol="", return_list=return_list) for t in y_target]

        values = {
            "nb_samples": b,
            "str_y": str_y,
            "str_x": str_x,
            "loss": sum_loss.item(),
            "loss_ce": loss_ce.item(),
            "syn_max_lines": self.dataset.train_dataset.get_syn_max_lines() if self.params["dataset_params"]["config"].get("synthetic_data", None) else 0,
        }

        return values


    def evaluate_batch(self, batch_data, metric_names, **kwargs):
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]

        max_chars = self.params["training_params"]["max_char_prediction"]

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            reached_end = torch.zeros((b, ), dtype=torch.bool, device=self.device)
            prediction_len = torch.zeros((b, ), dtype=torch.int, device=self.device)
            predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.dataset.tokens["start"]
            predicted_tokens_len = torch.ones((b, ), dtype=torch.int, device=self.device)

            whole_output = list()
            confidence_scores = list()
            cache = None
            cache_mask = None
            hidden_predict = None
            if b > 1:
                features_list = list()
                for i in range(b):
                    pos = batch_data["imgs_position"]
                    features_list.append(self.models["encoder"](x[i:i+1, :, pos[i][0][0]:pos[i][0][1], pos[i][1][0]:pos[i][1][1]]))
                max_height = max([f.size(2) for f in features_list])
                max_width = max([f.size(3) for f in features_list])
                features = torch.zeros((b, features_list[0].size(1), max_height, max_width), device=self.device, dtype=features_list[0].dtype)
                for i in range(b):
                    features[i, :, :features_list[i].size(2), :features_list[i].size(3)] = features_list[i]
            else:
                features = self.models["encoder"](x)
            features_size = features.size()
            # coverage_vector = torch.zeros((features.size(0), 1, features.size(2), features.size(3)), device=self.device)
            coverage_layers = {
                # 'chars': torch.zeros((features.size(0), 1, features.size(2), features.size(3)), device=self.device),
                'sem_start': torch.zeros((features.size(0), 1, features.size(2), features.size(3)), device=self.device),
                'sem_end': torch.zeros((features.size(0), 1, features.size(2), features.size(3)), device=self.device),
                'line_break': torch.zeros((features.size(0), 1, features.size(2), features.size(3)),
                                          device=self.device),
            }
            coverage_tokens = {
                # 'chars': torch.arange(1, 89, dtype=torch.int64, device=self.device),
                'sem_start': torch.tensor([605, ], dtype=torch.int64, device=self.device),
                'sem_end': torch.tensor([600, ], dtype=torch.int64, device=self.device),
                'line_break': torch.tensor([0, ], dtype=torch.int64, device=self.device),
            }
            pos_features = self.models["decoder"].features_updater.get_pos_features(features)
            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            enhanced_features = pos_features
            enhanced_features = torch.flatten(enhanced_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            """For debug"""
            keep_all_weights = True

            if keep_all_weights:
                all_weights = []

            for i in range(0, max_chars):
                # if i in [300, 500]:
                #     print('debug')
                output, pred, hidden_predict, cache, cache_mask, weights = self.models["decoder"](features,
                                                                                                  enhanced_features,
                                                                                                  predicted_tokens,
                                                                                                  reduced_size, features_size,
                                                                                                  start=0, hidden_predict=hidden_predict,
                                                                                                  cache=cache, cache_mask=cache_mask,
                                                                                                  num_pred=1, keep_all_weights=keep_all_weights)
                whole_output.append(output)
                confidence_scores.append(torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values)
                predicted_tokens = torch.cat([predicted_tokens, torch.argmax(pred[:, :, -1], dim=1, keepdim=True)], dim=1)
                if keep_all_weights:
                    for ckey in coverage_layers.keys():
                        cmask = torch.isin(predicted_tokens[:, -1], coverage_tokens[ckey]) & torch.logical_not(reached_end)
                        coverage_layers[ckey][cmask] = torch.clamp(coverage_layers[ckey][cmask] + weights['mix'][-1][cmask], 0, 1)
                    # coverage_vector = torch.clamp(coverage_vector + weights['mix'][-1], 0, 1)
                    weights['self'] = [w.detach().cpu().numpy() for w in weights['self']]
                    weights['mix'] = [w.detach().cpu().numpy() for w in weights['mix']]
                    all_weights.append(weights)
                else:
                    coverage_vector = torch.clamp(coverage_vector + weights, 0, 1)

                # predicted_tokens[:, -1] = self.dataset.class_set.index('n')
                reached_end = torch.logical_or(reached_end, torch.eq(predicted_tokens[:, -1], self.dataset.tokens["end"]))
                predicted_tokens_len += 1

                prediction_len[reached_end == False] = i + 1
                if torch.all(reached_end):
                    break

            confidence_scores = torch.cat(confidence_scores, dim=1).cpu().detach().numpy()
            predicted_tokens = predicted_tokens[:, 1:]
            prediction_len[torch.eq(reached_end, False)] = max_chars - 1
            predicted_tokens = [predicted_tokens[i, :prediction_len[i]] for i in range(b)]
            confidence_scores = [confidence_scores[i, :prediction_len[i]].tolist() for i in range(b)]
            return_list = self.params['dataset_params']['lan'] == 'bo' and not self.params['dataset_params'].get('use_comp', False)
            str_x = [LM_ind_to_str(self.dataset.class_set, t, oov_symbol="", return_list=return_list) for t in predicted_tokens]

        process_time = time.time() - start_time

        # for ckey in coverage_layers.keys():
        #     coverage_layers[ckey] = [coverage_layers[ckey][b, 0, :reduced_size[b][0], :reduced_size[b][1]] for b in range(x.size(0))]
        #
        # self.visualize_coverage(batch_data['paths'], coverage_layers)

        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "confidence_score": confidence_scores,
            "time": process_time,
        }
        return values

    def test_batch1(self, batch_data, metric_names, **kwargs):
        # print('test_batch')
        two_phase = kwargs['two_phase'] and 'skip' in self.dataset.tokens
        pre_predict_len = 20
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            B = x.size(0)
            if B > 1:
                features_list = list()
                for i in range(B):
                    pos = batch_data["imgs_position"]
                    features_list.append(self.models["encoder"](x[i:i+1, :, pos[i][0][0]:pos[i][0][1], pos[i][1][0]:pos[i][1][1]]))
                max_height = max([f.size(2) for f in features_list])
                max_width = max([f.size(3) for f in features_list])
                features = torch.zeros((B, features_list[0].size(1), max_height, max_width), device=self.device, dtype=features_list[0].dtype)
                for i in range(B):
                    features[i, :, :features_list[i].size(2), :features_list[i].size(3)] = features_list[i]
            else:
                features = self.models["encoder"](x)
            features_size = torch.tensor(features.size())
            coverage_vector = torch.zeros((features.size(0), 1, features.size(2), features.size(3)), device=self.device)
            pos_features = self.models["decoder"].features_updater.get_pos_features(features)
            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            enhanced_features = pos_features
            enhanced_features = torch.flatten(enhanced_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            """For debug"""
            keep_all_weights = True

            if keep_all_weights:
                all_weights = []

            phases = ['transcript']
            reached_end = torch.zeros((B,), dtype=torch.bool, device=self.device)
            prediction_len = torch.zeros((B,), dtype=torch.long, device=self.device)
            max_chars = self.params["training_params"]["max_char_prediction"]
            corrupted = torch.zeros((B,), dtype=torch.bool, device=self.device)

            if two_phase:
                phases = ['layout'] + phases
                # max_chars = self.params["training_params"]["max_char_prediction"]  // 10
                layout_tokens = self.dataset.tokens['sem'] + [self.dataset.tokens['end']]  # are layout tokens
                if self.params['model_params']['skip_content']['stop_at_line_break']:
                    layout_tokens.append(self.dataset.tokens['lb'])
                layout_tokens = torch.tensor(layout_tokens, dtype=torch.long, device=self.device)
                saved_tokens = [[] for b in range(B)]
                saved_cache = [[] for b in range(B)]
                saved_cache_mask = [[] for b in range(B)]
                # saved_stop = [[] for b in range(B)]
                to_insert = torch.zeros((B,), dtype=torch.bool, device=self.device)
                # inserted = to_insert.clone()
                char_cnt = torch.zeros((B,), dtype=torch.long, device=self.device)

                # tail = [[] for j in range(b)]

            predicted_tokens = torch.ones((B, max_chars+1), dtype=torch.long, device=self.device) * self.dataset.tokens["pad"]
            predicted_tokens[:, 0] = self.dataset.tokens['start']

            whole_output = list()
            confidence_scores = list()
            cache = None
            cache_mask = None
            hidden_predict = None

            # if 12 in batch_data['ids']:
            #     print('debug')

            for phase in phases:
                for i in range(0, max_chars):
                    # if i in [3, ] and phase == 'transcript':
                    #     print('debug')
                    sample_mask = prediction_len == i
                    if not torch.any(sample_mask):
                        continue
                    output, pred, hidden_predict, cache_t, cache_mask_t, weights = self.models["decoder"](features[:, sample_mask],
                                                                                          enhanced_features[:, sample_mask],
                                                                                          predicted_tokens[sample_mask, :i+1],
                                                                                          [reduced_size[idx] for idx, mask in enumerate(sample_mask) if mask],
                                                                                          features_size,
                                                                                          start=0, hidden_predict=hidden_predict,
                                                                                          cache=[cache[idx] for idx, mask in enumerate(sample_mask) if mask] if cache is not None else cache,
                                                                                          cache_mask = cache_mask[sample_mask, :i] if cache_mask is not None else cache_mask,
                                                                                          num_pred=1, keep_all_weights=keep_all_weights)
                    if cache is not None:
                        for idx, m in enumerate(sample_mask):
                            if m:
                                cache[idx] = cache_t[0]
                                cache_t = cache_t[1:]
                        if cache_mask.shape[1] <= i:
                            cache_mask = torch.cat([
                                cache_mask,
                                torch.zeros((cache_mask.shape[0], 1), dtype=cache_mask[0].dtype, device=cache_mask[0].device)
                            ], dim=1)
                        cache_mask[sample_mask, :i+1] = cache_mask_t[:, :i+1]
                    else:
                        cache = cache_t
                        cache_mask = cache_mask_t


                    # whole_output.append(output)
                    # confidence_scores.append(torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values)
                    # if keep_all_weights:
                    #     coverage_vector = torch.clamp(coverage_vector + weights['mix'][-1], 0, 1)
                    #     weights['mix'] = [w.detach().cpu().numpy() for w in weights['mix']]
                    #     all_weights.append(weights)
                    # else:
                    #     coverage_vector = torch.clamp(coverage_vector + weights, 0, 1)
                    new_tokens = torch.argmax(pred[:, :, -1], dim=1, keepdim=False)

                    if phase == 'layout':
                        reached_end[sample_mask] = torch.eq(new_tokens, self.dataset.tokens["end"])
                        is_layout = torch.isin(new_tokens, layout_tokens)
                        corrupted[sample_mask] = to_insert[sample_mask] & torch.logical_not(is_layout)
                        char_cnt_ = char_cnt[sample_mask]
                        char_cnt_[is_layout] = 0
                        char_cnt_[torch.logical_not(is_layout)] += 1
                        to_insert = torch.zeros((B,), dtype=torch.bool, device=self.device)
                        to_insert[sample_mask] = char_cnt_ >= pre_predict_len + 1
                        char_cnt[sample_mask] = char_cnt_
                        idx = 0
                        for b, smask in enumerate(sample_mask):
                            if to_insert[b]:
                                saved_tokens[b] += [torch.cat([predicted_tokens[b, :i+1], new_tokens[idx].unsqueeze(dim=-1)], dim=0).clone()]
                                saved_cache[b] += [cache[b]]
                                saved_cache_mask[b] += [cache_mask[b]]
                            # if inserted[b]:
                            #     saved_stop[b] += [new_tokens[idx].unsqueeze(dim=-1)]
                            if smask:
                                idx += 1
                        new_tokens[to_insert[sample_mask]] = self.dataset.tokens['skip']
                        # inserted = to_insert.clone()

                    elif two_phase and torch.all(torch.logical_not(corrupted)):
                        reached_end[sample_mask] = torch.isin(new_tokens, layout_tokens)
                    else:
                        reached_end[sample_mask] = torch.eq(new_tokens, self.dataset.tokens["end"])

                    predicted_tokens[sample_mask, i+1] = new_tokens

                    prediction_len[sample_mask & torch.logical_not(reached_end)] = i+1
                    if torch.all(reached_end):
                        break
                if phase == 'layout':
                    corrupted = corrupted | torch.logical_not(reached_end)
                    if torch.all(torch.logical_not(corrupted)):
                        """no corrupted layout, prepare for parallel partial prediction"""
                        sample_inds = []
                        predicted_tokens_ = predicted_tokens.clone()
                        prediction_len_ = prediction_len.clone()
                        cache_ = cache.copy()
                        cache_mask_ = cache_mask.clone()
                        nb_thread = np.sum([len(th) for th in saved_tokens])

                        predicted_tokens = torch.ones((nb_thread, max_chars+1), dtype=torch.long, device=self.device) * self.dataset.tokens["pad"]
                        prediction_len = []
                        cache = []
                        cache_mask = []
                        saved_stop = [[] for b in range(B)]
                        for b in range(B):
                            for j in range(len(saved_tokens[b])):
                                sample_inds.append(b)
                                prediction_len.append(saved_tokens[b][j].size(0)-1)
                                if j < len(saved_tokens[b]) - 1:
                                    saved_stop[b].append(predicted_tokens_[b, saved_tokens[b][j].size(0):saved_tokens[b][j+1].size(0)-pre_predict_len-1])
                                else:
                                    saved_stop[b].append(predicted_tokens_[b, saved_tokens[b][j].size(0):prediction_len_[b]+1])
                                predicted_tokens[len(sample_inds)-1, :prediction_len[-1]+1] = saved_tokens[b][j]
                                cache.append(saved_cache[b][j])
                                cache_mask.append(saved_cache_mask[b][j])
                                # cache.append(cache_[b][:, :prediction_len[-1], :])

                        prediction_len = torch.tensor(prediction_len, dtype=torch.long, device=self.device)


                        if len(sample_inds) > B * 15 and 'lb' in self.dataset.tokens:
                            end_tokens = []
                            for b in range(B):
                                end_tokens += saved_stop[b]
                            end_tokens = torch.stack([e[0] for e in end_tokens])
                            is_block = end_tokens != self.dataset.tokens['lb']
                            layout_tokens = layout_tokens[layout_tokens != self.dataset.tokens['lb']]
                            """逻辑有错误，需要修改"""
                            sample_mask = torch.cat([torch.ones((1,), dtype=torch.bool, device=is_block.device),
                                                     is_block[:-1]], dim=0)
                            predicted_tokens = predicted_tokens[sample_mask]
                            prediction_len = prediction_len[sample_mask]
                            sample_inds = [sample_inds[j] for j in range(sample_mask.size(0)) if sample_mask[j]]
                            cache = [cache[j] for j in range(sample_mask.size(0)) if sample_mask[j]]
                            cache_mask = [cache_mask[j] for j in range(sample_mask.size(0)) if sample_mask[j]]
                            saved_tokens_ = [[] for b in range(B)]
                            saved_stop_ = [[] for b in range(B)]
                            idx = 0
                            for b in range(B):
                                for j in range(len(saved_tokens[b])):
                                    if sample_mask[idx]:
                                        saved_tokens_[b].append(saved_tokens[b][j])
                                    if is_block[idx]:
                                        saved_stop_[b].append(saved_stop[b][j])
                                    idx += 1
                            saved_tokens = saved_tokens_
                            saved_stop = saved_stop_
                            # max_chars *= 3

                        cache_mask = self.pad_cache_mask(cache_mask)
                        # max_chars += torch.max(prediction_len).item()


                        features = torch.stack([features[:, ind, :] for ind in sample_inds], dim=1)
                        enhanced_features = torch.stack([enhanced_features[:, ind, :] for ind in sample_inds], dim=1)
                        features_size[0] = len(sample_inds)

                        # coverage_vector = torch.zeros((features_size[0], 1, features_size[2], features_size[3]), device=self.device)
                        reduced_size = [reduced_size[ind] for ind in sample_inds]
                        reached_end = torch.zeros((len(sample_inds),), dtype=torch.bool, device=self.device)
                        # confidence_scores_layout = torch.cat(confidence_scores, dim=1).detach().cpu().numpy()
                        # if keep_all_weights:
                        #     all_weights_layout = all_weights.copy()
                    else:
                        """exist corrupted layout, resort to sequential full prediction"""
                        predicted_tokens = torch.ones((B, 1), dtype=torch.long, device=self.device) * self.dataset.tokens["start"]
                        predicted_tokens_len = 1
                        cache = None
                        cache_mask = None
                        coverage_vector = torch.zeros((features_size[0], 1, features_size[2], features_size[3]), device=self.device)
                        reached_end = torch.zeros((B,), dtype=torch.bool, device=self.device)
                        prediction_len = torch.zeros((B,), dtype=torch.int, device=self.device)
                        # max_chars = self.params["training_params"]["max_char_prediction"]

                    whole_output = list()
                    confidence_scores = list()
                    if keep_all_weights:
                        all_weights = []

            # confidence_scores = torch.cat(confidence_scores, dim=1).detach().cpu().numpy()
            if two_phase and torch.all(torch.logical_not(corrupted)):
                """merge the partial predictions"""
                predicted_tokens_full = []
                idx = 0
                for b in range(B):
                    tokens = []
                    for j in range(len(saved_stop[b])):
                        start = saved_tokens[b][j].size(0) - pre_predict_len - 1 if j > 0 else 1
                        tokens.append(predicted_tokens[idx, start: prediction_len[idx]+1])
                        tokens.append(saved_stop[b][j])
                        idx += 1
                    predicted_tokens_full.append(torch.cat(tokens, dim=0))
                predicted_tokens = predicted_tokens_full
            else:
                predicted_tokens = predicted_tokens[:, 1:]
                prediction_len[torch.eq(reached_end, False)] = max_chars - 1
                predicted_tokens = [predicted_tokens[b, :prediction_len[b]] for b in range(B)]
                # confidence_scores = [confidence_scores[b, :prediction_len[b]].tolist() for b in range(B)]
            return_list = self.params['dataset_params']['lan'] == 'bo' and not self.params['dataset_params'].get('use_comp', False)
            str_x = [LM_ind_to_str(self.dataset.class_set, t, oov_symbol="", return_list=return_list) for t in predicted_tokens]
            # str_y = [''.join(t) for t in batch_data['raw_labels']]

        process_time = time.time() - start_time

        values = {
            "nb_samples": B,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "confidence_score": confidence_scores,
            "time": process_time,
        }
        if two_phase:
            values['nb_corrupted'] = torch.sum(corrupted).item()
        return values


    def update_cache(self, cache, cache_t, sample_mask):
        if cache is None:
            cache = cache_t
            return cache
        if cache.shape[1] < cache_t.shape[1]:
            cache = torch.cat([cache, torch.zeros((cache.shape[0], 1, cache.shape[2], cache.shape[3]), dtype=cache.dtype, device=cache.device)], dim=1)
        cache[:, :cache_t.shape[1], sample_mask, :] = cache_t
        return cache

    def pad_tokens_1D(self, tokens, padding_value, before=True):
        x_lens = [token.shape[0] for token in tokens]
        max_len = max(x_lens)
        padded_data = torch.ones((len(tokens), max_len), dtype=torch.long, device=self.device) * padding_value
        for i, x_len in enumerate(x_lens):
            if before:
                padded_data[i, -x_len:] = tokens[i]
            else:
                padded_data[i, :x_len] = tokens[i]
        return padded_data

    # def pad_cache(self, cache):
    #     x_lens = [c.shape[1] for c in cache]
    #     max_len = max(x_lens)
    #     padded_cache = torch.zeros((cache[0].shape[0], max_len, len(cache), cache[0].shape[-1]), dtype=cache[0].dtype, device=self.device)
    #     num_pad = []
    #     for i, x_len in enumerate(x_lens):
    #         padded_cache[:, :x_len, i, :] = cache[i]
    #         # num_pad.append(max_len-x_len)
    #     return padded_cache

    def pad_cache_mask(self, cache_mask):
        x_lens = [c.size(0) for c in cache_mask]
        max_len = max(x_lens)
        padded_cache_mask = torch.zeros((len(cache_mask), max_len), dtype=cache_mask[0].dtype, device=cache_mask[0].device)
        for i, x_len in enumerate(x_lens):
            padded_cache_mask[i, :x_len] = cache_mask[i]
            # num_pad.append(max_len-x_len)
        return padded_cache_mask