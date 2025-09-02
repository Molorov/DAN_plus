#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in Python  whose purpose is to
#  provide public implementation of deep learning works, in pytorch.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.
import numpy.random

from basic.generic_dataset_manager import DatasetManager, GenericDataset
from basic.utils import pad_images, pad_image_width_right, resize_max, pad_image_width_random, pad_sequences_1D, pad_image_height_random, pad_image_width_left, pad_image
from basic.utils import randint, rand, rand_uniform
from basic.generic_dataset_manager import apply_preprocessing
from Datasets.dataset_formatters.read2016_formatter import SEM_MATCHING_TOKENS as READ_MATCHING_TOKENS
from Datasets.dataset_formatters.rimes_formatter import order_text_regions as order_text_regions_rimes
from Datasets.dataset_formatters.rimes_formatter import SEM_MATCHING_TOKENS as RIMES_MATCHING_TOKENS
from Datasets.dataset_formatters.rimes_formatter import SEM_MATCHING_TOKENS_STR as RIMES_MATCHING_TOKENS_STR
from Datasets.dataset_formatters.kangyur_formatter import SEM_MATCHING_TOKENS as KANGYUR_MATCHING_TOKENS
from basic.utils import remove_duplicates_for_charlist, split_char_list, syllable_delims, remove_duplicates_for_charlist, strip_charlist
from OCR.ocr_utils import LM_str_to_ind
import random
import cv2
import os
import copy
import pickle
import numpy as np
import torch
import re
import matplotlib
from PIL import Image, ImageDraw, ImageFont
from basic.transforms import RandomRotation, apply_transform, Tightening
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
from torchvision.utils import save_image

import unicodedata as ucd
from Datasets.dataset_formatters.utils_dataset import get_sorted_lan_dict
from Datasets.dataset_formatters.kangyur_formatter import is_kangyur
from basic.metric_manager import get_matching_tokens_and_pp_module
import random

"""
获取用于合成印刷文本的charset
"""
# def char_only_set_for_Tibetan(params):
#     datasets = params["datasets"]
#     charset = set()
#     for key in datasets.keys():
#         with open(os.path.join(datasets[key], "labels.pkl"), "rb") as f:
#             info = pickle.load(f)
#             gt = info['ground_truth']
#             for set_name, set_data in gt.items():
#                 for page_id, page_data in set_data.items():
#                     c_charset = set([c for c in page_data['text']])
#                     charset = charset.union(c_charset)
#             charset
#
#     if '\xa0' in charset:
#         """None-breaking space没什么用，删除"""
#         charset.remove('\xa0')
#     if "\n" in charset and "remove_linebreaks" in params["config"]["constraints"]:
#         charset.remove("\n")
#     if "" in charset:
#         charset.remove("")
#     return sorted(list(charset))

def get_runs(tokens, chr_token):
    runs = []
    is_content = []
    i_ = 0
    for i in range(len(tokens)):
        if i > 0 and ((tokens[i] in chr_token) ^ (tokens[i-1] in chr_token)):
            runs.append(tokens[i_: i])
            is_content.append(tokens[i-1] in chr_token)
            i_ = i
    if i_ < len(tokens):
        runs.append(tokens[i_:])
        is_content.append(tokens[i_] in chr_token)
    return runs, is_content


def get_char_only_set(charset):
    char_only_set = set()
    for chr in charset:
        for comp in chr:
            char_only_set = char_only_set.union(comp)
    return sorted(list(char_only_set))

def char_only_dict_by_lan(char_only_set):
    lan_dict = get_sorted_lan_dict(char_only_set)
    for lan in lan_dict:
        lan_dict[lan] = sorted(set(lan_dict[lan]))

    return lan_dict



class OCRDatasetManager(DatasetManager):
    """
    Specific class to handle OCR/HTR tasks
    """

    def __init__(self, params):
        super(OCRDatasetManager, self).__init__(params)

        self.charset = params["charset"] if "charset" in params else self.get_merged_charsets()
        use_comp = params.get('use_comp', False)
        if use_comp:
            self.class_set = params['compset'] if 'compset' in params else self.get_merged_compsets()
        else:
            self.class_set = self.charset.copy()

        if "synthetic_data" in self.params["config"] and self.params["config"]["synthetic_data"] and "config" in self.params["config"]["synthetic_data"]:
            self.char_only_set = get_char_only_set(self.charset.copy())

            for token_dict in [RIMES_MATCHING_TOKENS, READ_MATCHING_TOKENS, KANGYUR_MATCHING_TOKENS]:
                for key in token_dict:
                    if key in self.char_only_set:
                        self.char_only_set.remove(key)
                    if token_dict[key] in self.char_only_set:
                        self.char_only_set.remove(token_dict[key])
            for token in ["\n", ]:
                if token in self.char_only_set:
                    self.char_only_set.remove(token)
            self.char_only_dict = char_only_dict_by_lan(self.char_only_set)
            font_dir = params['config']['synthetic_data'].get('font_dir', None)
            self.params["config"]["synthetic_data"]["config"]["valid_fonts"] = get_valid_fonts(self.char_only_dict, font_dir)

        # if "new_tokens" in params:
        #     self.charset = sorted(list(set(self.charset).union(set(params["new_tokens"]))))

        self.tokens = {
            'sem': [],
            "pad": params["config"]["padding_token"],
        }

        dataset_name = os.path.basename(list(params["datasets"].values())[0])
        sem_tokens, _, _ = get_matching_tokens_and_pp_module(dataset_name)
        if len(sem_tokens):
            sem_tokens = list(sem_tokens.keys()) + list(sem_tokens.values())
            sem_tokens = [self.class_set.index(c) for c in sem_tokens]
            self.tokens['sem'] = sorted(sem_tokens)
        if '\n' in self.class_set:
            self.tokens['lb'] = self.class_set.index('\n')

        if self.params["config"]["charset_mode"].lower() == "ctc":
            self.tokens["blank"] = len(self.class_set)
            self.tokens["pad"] = self.tokens["pad"] if self.tokens["pad"] else len(self.class_set) + 1
            self.params["config"]["padding_token"] = self.tokens["pad"]
        elif self.params["config"]["charset_mode"] == "seq2seq":
            self.tokens["end"] = len(self.class_set)
            num_tokens = len(self.class_set) + 1
            if 'skip' in self.params['config']:
                if 'paragraph' in self.params['config']['skip'] and self.params['config']['skip']['paragraph']:
                    self.tokens['skip_ph'] = num_tokens  # skip a entire paragraph
                    num_tokens += 1
                if 'line_break' in self.params['config']['skip'] and self.params['config']['skip']['line_break']:
                    self.tokens['skip_lb'] = num_tokens  # skip a text line
                    num_tokens += 1
                if 'word' in self.params['config']['skip'] and self.params['config']['skip']['word']:
                    self.tokens['skip_wd'] = num_tokens  # skip a word
                    num_tokens += 1
            if 'pause' in self.params['config']:
                self.tokens['pause'] = num_tokens
                num_tokens += 1
            self.tokens["start"] = num_tokens
            self.tokens["pad"] = self.tokens["pad"] if self.tokens["pad"] else self.tokens['start'] + 1
            self.params["config"]["padding_token"] = self.tokens["pad"]
        return

    def get_merged_charsets(self):
        """
        Merge the charset of the different datasets used
        """
        datasets = self.params["datasets"]
        charset = set()
        label_file = self.params.get('label_file', "labels.pkl")
        for key in datasets.keys():
            with open(os.path.join(datasets[key], label_file), "rb") as f:
                info = pickle.load(f)
                charset = charset.union(set(info["charset"]))

        if "\n" in charset and "remove_linebreaks" in self.params["config"]["constraints"]:
            charset.remove("\n")
        if "" in charset:
            charset.remove("")
        if '\xa0' in charset:
            charset.remove('\xa0')
        return sorted(list(charset))

    def get_merged_compsets(self):
        """
        Merge the charset of the different datasets used
        """
        datasets = self.params["datasets"]
        compset = set()
        label_file = self.params.get('label_file', "labels.pkl")
        for key in datasets.keys():
            with open(os.path.join(datasets[key], label_file), "rb") as f:
                info = pickle.load(f)
                compset = compset.union(set(info["compset"]))
        if "\n" in compset and "remove_linebreaks" in self.params["config"]["constraints"]:
            compset.remove("\n")
        if "" in compset:
            compset.remove("")
        if '\xa0' in compset:
            compset.remove('\xa0')
        return sorted(list(compset))

    def apply_specific_treatment_after_dataset_loading(self, dataset):
        dataset.class_set = self.class_set
        dataset.tokens = self.tokens
        dataset.convert_labels()
        """由于是彩色图像，因此在图像变换时最好指定平均像素值作为背景填充（非必要）"""
        if ("READ_2016" in dataset.name or dataset.params['lan'].lower() == 'bo') and "augmentation" in dataset.params["config"] and dataset.params["config"]["augmentation"]:
            dataset.params["config"]["augmentation"]["fill_value"] = tuple([int(i) for i in dataset.mean])
        if "padding" in dataset.params["config"] and dataset.params["config"]["padding"]["min_height"] == "max":
            dataset.params["config"]["padding"]["min_height"] = max([s["img"].shape[0] for s in self.train_dataset.samples])
        if "padding" in dataset.params["config"] and dataset.params["config"]["padding"]["min_width"] == "max":
            dataset.params["config"]["padding"]["min_width"] = max([s["img"].shape[1] for s in self.train_dataset.samples])


class OCRDataset(GenericDataset):
    """
    Specific class to handle OCR/HTR datasets
    """

    def __init__(self, params, set_name, custom_name, paths_and_sets):
        super(OCRDataset, self).__init__(params, set_name, custom_name, paths_and_sets)
        self.charset = None
        self.tokens = None
        self.reduce_dims_factor = np.array([params["config"]["height_divisor"], params["config"]["width_divisor"], 1])
        self.collate_function = OCRCollateFunction
        self.synthetic_id = 0

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])

        if not self.load_in_memory:
            sample["img"] = self.get_sample_img(idx)
            sample = apply_preprocessing(sample, self.params["config"]["preprocessings"])

        if "synthetic_data" in self.params["config"] and self.params["config"]["synthetic_data"] and self.set_name == "train":
            sample = self.generate_synthetic_data(sample)

        # Data augmentation
        sample["img"], sample["applied_da"] = self.apply_data_augmentation(sample["img"])

        if "max_size" in self.params["config"] and self.params["config"]["max_size"]:
            max_ratio = max(sample["img"].shape[0] / self.params["config"]["max_size"]["max_height"], sample["img"].shape[1] / self.params["config"]["max_size"]["max_width"])
            if max_ratio > 1:
                new_h, new_w = int(np.ceil(sample["img"].shape[0] / max_ratio)), int(np.ceil(sample["img"].shape[1] / max_ratio))
                sample["img"] = cv2.resize(sample["img"], (new_w, new_h))

        # Normalization if requested
        if "normalize" in self.params["config"] and self.params["config"]["normalize"]:
            sample["img"] = (sample["img"] - self.mean) / self.std

        sample["img_shape"] = sample["img"].shape
        sample["img_reduced_shape"] = np.ceil(sample["img_shape"] / self.reduce_dims_factor).astype(int)

        # Padding to handle CTC requirements
        if self.set_name == "train":
            max_label_len = 0
            height = 1
            ctc_padding = False
            if "CTC_line" in self.params["config"]["constraints"]:
                max_label_len = sample["label_len"]
                ctc_padding = True
            if "CTC_va" in self.params["config"]["constraints"]:
                max_label_len = max(sample["line_label_len"])
                ctc_padding = True
            if "CTC_pg" in self.params["config"]["constraints"]:
                max_label_len = sample["label_len"]
                height = max(sample["img_reduced_shape"][0], 1)
                ctc_padding = True
            if ctc_padding and 2 * max_label_len + 1 > sample["img_reduced_shape"][1]*height:
                sample["img"] = pad_image_width_right(sample["img"], int(np.ceil((2 * max_label_len + 1) / height) * self.reduce_dims_factor[1]), self.padding_value)
                sample["img_shape"] = sample["img"].shape
                sample["img_reduced_shape"] = np.ceil(sample["img_shape"] / self.reduce_dims_factor).astype(int)
            sample["img_reduced_shape"] = [max(1, t) for t in sample["img_reduced_shape"]]

        sample["img_position"] = [[0, sample["img_shape"][0]], [0, sample["img_shape"][1]]]
        # Padding constraints to handle model needs
        if "padding" in self.params["config"] and self.params["config"]["padding"]:
            if self.set_name == "train" or not self.params["config"]["padding"]["train_only"]:
                min_pad = self.params["config"]["padding"]["min_pad"]
                max_pad = self.params["config"]["padding"]["max_pad"]
                pad_width = randint(min_pad, max_pad) if min_pad is not None and max_pad is not None else None
                pad_height = randint(min_pad, max_pad) if min_pad is not None and max_pad is not None else None

                sample["img"], sample["img_position"] = pad_image(sample["img"], padding_value=self.padding_value,
                                          new_width=self.params["config"]["padding"]["min_width"],
                                          new_height=self.params["config"]["padding"]["min_height"],
                                          pad_width=pad_width,
                                          pad_height=pad_height,
                                          padding_mode=self.params["config"]["padding"]["mode"],
                                          return_position=True)
        sample["img_reduced_position"] = [np.ceil(p / factor).astype(int) for p, factor in zip(sample["img_position"], self.reduce_dims_factor[:2])]

        return sample

    def insert_skip(self, seq):

        pass

    def get_subseq(self, seq, min_len, min_sil, max_sil):
        start = 0
        if min_len < len(seq):
            start = randint(1, len(seq)-min_len)
        subseq = seq[start:]

        # assert min_sil < min_len
        num_sil = randint(min_sil, min(len(subseq)-1, max_sil))

        return subseq, start, num_sil

    def get_charset(self):
        charset = set()
        for i in range(len(self.samples)):
            charset = charset.union(set(self.samples[i]["label"]))
        return charset

    def convert_labels(self):
        """
        Label str to token at character level
        """
        for i in range(len(self.samples)):
            self.samples[i] = self.convert_sample_labels(self.samples[i])

    def get_layout_tokens(self):
        if 'BJK' in self.params['name'] or 'LJK' in self.params['name'] or 'DGK' in self.params['name']:
            return KANGYUR_MATCHING_TOKENS
        elif 'READ_2016' in self.params['name']:
            return READ_MATCHING_TOKENS
        elif 'RIMES' in self.params['name']:
            return RIMES_MATCHING_TOKENS
        else:
            return {}

    def convert_sample_labels(self, sample):
        label = sample["label"]
        layout_tokens = self.get_layout_tokens()
        layout_tokens = list(layout_tokens.keys()) + list(layout_tokens.values())
        if type(sample["label"]) is str:
            label = [c for c in label]

        line_labels = split_char_list(label, delim=['\n']+layout_tokens, retain_zero_lens=False)
        unchanged_line_labels = split_char_list(sample['unchanged_label'], delim=['\n']+layout_tokens, retain_zero_lens=False)
        if "remove_linebreaks" in self.params["config"]["constraints"]:
            full_label = [l if l != '\n' else ' ' for l in label]
        else:
            full_label = label


        if type(sample["label"]) is str:
            full_label = ''.join(full_label)
            line_labels = [''.join(ll) for ll in line_labels]
        if type(sample['unchanged_label']) is str:
            unchanged_line_labels = [''.join(ll) for ll in unchanged_line_labels]
            # word_labels = [''.join(wl) for wl in word_labels]

        sample["label"] = full_label
        sample["token_label"] = LM_str_to_ind(self.class_set, full_label)
        if "add_eot" in self.params["config"]["constraints"]:
            sample["token_label"].append(self.tokens["end"])
        sample["label_len"] = len(sample["token_label"])
        if "add_sot" in self.params["config"]["constraints"]:
            sample["token_label"].insert(0, self.tokens["start"])


        sample["line_label"] = line_labels
        sample["token_line_label"] = [LM_str_to_ind(self.class_set, l) for l in line_labels]
        sample["line_label_len"] = [len(l) for l in line_labels]
        sample["nb_lines"] = len(line_labels)
        # sample['nb_entities'] = len(split_char_list(label, layout_tokens, retain_zero_lens=False))
        sample["unchanged_line_label"] = unchanged_line_labels


        sample['line_lan'] = []
        for line_lab in line_labels:
            lan_dict = get_sorted_lan_dict(''.join(line_lab), self.params['lan'])
            sample['line_lan'].append(next(iter(lan_dict)))
        if 'lan' not in sample:
            lan_dict = get_sorted_lan_dict(''.join(full_label), self.params['lan'])
            sample['lan'] = next(iter(lan_dict))

        return sample

    def generate_synthetic_data(self, sample):
        config = self.params["config"]["synthetic_data"]

        if not (config["init_proba"] == config["end_proba"] == 1):
            nb_samples = self.training_info['trained_samples'] if 'trained_samples' in self.training_info else self.training_info["step"] * self.params["batch_size"]
            if config["start_scheduler_at_max_line"]:
                max_step = config["num_steps_proba"]
                current_step = max(0, min(nb_samples-config["curr_step"]*(config["max_nb_lines"]-config["min_nb_lines"]), max_step))
                proba = config["init_proba"] if self.get_syn_max_lines() < config["max_nb_lines"] else \
                    config["proba_scheduler_function"](config["init_proba"], config["end_proba"], current_step, max_step)
            else:
                proba = config["proba_scheduler_function"](config["init_proba"], config["end_proba"],
                                                       min(nb_samples, config["num_steps_proba"]),
                                                       config["num_steps_proba"])
            if rand() > proba:
                return sample

        # self.params["config"]["synthetic_data"]["config"]['lan'] = self.params['lan']
        if "mode" in config and config["mode"] == "line_hw_to_printed":
            # writ_d = 'ttb' if self.params['lan'] == 'bo' and sample['lan'] == 'cjk' else 'ltr' # writing direction
            writ_d = 'ltr'
            sample = self.generate_typed_text_line_image(sample["unchanged_label"], sample['lan'], writ_d)
            img = sample['img']
            if writ_d == 'ttb':  # rotate 90deg counterclock wise
                img = img.transpose(1, 0, 2)
                img = np.flip(img, axis=0)
            sample = GenericDataset.preprocess_sample_label(sample, self.params)
            sample['name'] = "synthetic_data_{}".format(self.synthetic_id)
            self.synthetic_id += 1
            sample['img'] = img
            sample["nb_cols"] = 1
            sample = self.convert_sample_labels(sample)
            sample['path'] = None
            return sample

        return self.generate_synthetic_page_sample()

    def get_syn_max_lines(self):
        config = self.params["config"]["synthetic_data"]
        if config["curriculum"]:
            nb_samples = self.training_info['trained_samples'] if 'trained_samples' in self.training_info else self.training_info["step"] * self.params["batch_size"]
            max_nb_lines = min(config["max_nb_lines"], (nb_samples-config["curr_start"]) // config["curr_step"]+1)
            return max(config["min_nb_lines"], max_nb_lines)
        return config["max_nb_lines"]

    def generate_synthetic_page_sample(self):
        config = self.params["config"]["synthetic_data"]
        max_nb_lines_per_page = self.get_syn_max_lines()
        crop = config["crop_curriculum"] and max_nb_lines_per_page < config["max_nb_lines"]
        sample = {
            "name": "synthetic_data_{}".format(self.synthetic_id),
            "path": None
        }
        self.synthetic_id += 1
        nb_pages = 2 if "double" in config["dataset_level"] else 1
        background_sample = copy.deepcopy(self.samples[randint(0, len(self))])
        pages = list()
        backgrounds = list()

        h, w, c = background_sample["img"].shape
        page_width = w // 2 if nb_pages == 2 else w
        for i in range(nb_pages):
            nb_lines_per_page = max_nb_lines_per_page
            background = np.ones((h, page_width, c), dtype=background_sample["img"].dtype) * 255
            if i == 0 and nb_pages == 2:
                background[:, -2:, :] = 0
            backgrounds.append(background)
            if "READ_2016" in self.params["datasets"].keys():
                side = background_sample["pages_label"][i]["side"]
                coords = {
                    "left": int(0.15 * page_width) if side == "left" else int(0.05 * page_width),
                    "right": int(0.95 * page_width) if side == "left" else int(0.85 * page_width),
                    "top": int(0.05 * h),
                    "bottom": int(0.85 * h),
                }
                pages.append(self.generate_synthetic_read2016_page(background, coords, side=side, crop=crop,
                                                               nb_lines=nb_lines_per_page))
            elif "RIMES" in self.params["datasets"].keys():
                pages.append(self.generate_synthetic_rimes_page(background, nb_lines=nb_lines_per_page, crop=crop))
            elif 'IAM' in self.params["datasets"].keys():
                pages.append(self.generate_synthetic_iam_page(background, nb_lines=nb_lines_per_page, crop=crop))
            elif is_kangyur(next(iter(self.params["datasets"].keys()))):
                pad_left, pad_right, pad_top, pad_bottom = config.get('paddings_ratio', [0.05, 0.95, 0.05, 0.95])
                coords = {
                    "left": int(pad_left * page_width) if self.params['level'] == 'page' else 0,
                    "right": int(pad_right * page_width) if self.params['level'] == 'page' else page_width,
                    "top": int(pad_top * h) if self.params['level'] == 'page' else 0,
                    "bottom": int(pad_bottom * h) if self.params['level'] == 'page' else h,
                }
                pages.append(self.generate_synthetic_Kangyur_page(background, coords, crop=crop, nb_lines=nb_lines_per_page))
            else:
                raise NotImplementedError

        if nb_pages == 1:
            sample["img"] = pages[0][0]
            sample["label_raw"] = pages[0][1]["raw"]
            sample["label_begin"] = pages[0][1]["begin"]
            sample["label_sem"] = pages[0][1]["sem"]
            sample["label"] = pages[0][1]
            sample["nb_cols"] = pages[0][2]
        else:
            if pages[0][0].shape[0] != pages[1][0].shape[0]:
                max_height = max(pages[0][0].shape[0], pages[1][0].shape[0])
                backgrounds[0] = backgrounds[0][:max_height]
                backgrounds[0][:pages[0][0].shape[0]] = pages[0][0]
                backgrounds[1] = backgrounds[1][:max_height]
                backgrounds[1][:pages[1][0].shape[0]] = pages[1][0]
                pages[0][0] = backgrounds[0]
                pages[1][0] = backgrounds[1]
            sample["label_raw"] = pages[0][1]["raw"] + "\n" + pages[1][1]["raw"]
            sample["label_begin"] = pages[0][1]["begin"] + pages[1][1]["begin"]
            sample["label_sem"] = pages[0][1]["sem"] + pages[1][1]["sem"]
            sample["img"] = np.concatenate([pages[0][0], pages[1][0]], axis=1)
            sample["nb_cols"] = pages[0][2] + pages[1][2]
        sample["label"] = sample["label_raw"]
        if "ⓑ" in self.class_set:
            sample["label"] = sample["label_begin"]
        if "Ⓑ" in self.class_set:
            sample["label"] = sample["label_sem"]
        if self.params.get('use_comp', False):
            sample['label'] = ''.join(sample['label'])
        sample["unchanged_label"] = sample["label"]
        sample = self.convert_sample_labels(sample)
        # sample['save_path'] = os.path.join()

        return sample

    def generate_synthetic_rimes_page(self, background, nb_lines=20, crop=False):
        max_nb_lines = self.get_syn_max_lines()
        def larger_lines(label):
            lines = label.split("\n")
            new_lines = list()
            while len(lines) > 0:
                if len(lines) == 1:
                    new_lines.append(lines[0])
                    del lines[0]
                elif len(lines[0]) + len(lines[1]) < max_len:
                    new_lines.append("{} {}".format(lines[0], lines[1]))
                    del lines[1]
                    del lines[0]
                else:
                    new_lines.append(lines[0])
                    del lines[0]
            return "\n".join(new_lines)
        config = self.params["config"]["synthetic_data"]
        max_len = 100
        matching_tokens = RIMES_MATCHING_TOKENS
        matching_tokens_str = RIMES_MATCHING_TOKENS_STR
        h, w, c = background.shape
        num_lines = list()
        for s in self.samples:
            l = sum([len(p["label"].split("\n")) for p in s["paragraphs_label"]])
            num_lines.append(l)
        stats = self.stat_sem_rimes()
        ordered_modes = ['Corps de texte', 'PS/PJ', 'Ouverture', 'Date, Lieu', 'Coordonnées Expéditeur', 'Coordonnées Destinataire', ]
        object_ref = ['Objet', 'Reference']
        random.shuffle(object_ref)
        ordered_modes = ordered_modes[:3] + object_ref + ordered_modes[3:]
        kept_modes = list()
        for mode in ordered_modes:
            if rand_uniform(0, 1) < stats[mode]:
                kept_modes.append(mode)

        paragraphs = dict()
        for mode in kept_modes:
            paragraphs[mode] = self.get_paragraph_rimes(mode=mode, mix=True)
            # proba to merge multiple body textual contents
            if mode == "Corps de texte" and rand_uniform(0, 1) < 0.2:
                nb_lines = min(nb_lines+10, max_nb_lines) if max_nb_lines < 30 else nb_lines+10
                concat_line = randint(0, 2) == 0
                if concat_line:
                    paragraphs[mode]["label"] = larger_lines(paragraphs[mode]["label"])
                while (len(paragraphs[mode]["label"].split("\n")) <= 30):
                    body2 = self.get_paragraph_rimes(mode=mode, mix=True)
                    paragraphs[mode]["label"] += "\n" + larger_lines(body2["label"]) if concat_line else body2["label"]
                    paragraphs[mode]["label"] = "\n".join(paragraphs[mode]["label"].split("\n")[:40])
        # proba to set whole text region to uppercase
        if rand_uniform(0, 1) < 0.1 and "Corps de texte" in paragraphs:
            paragraphs["Corps de texte"]["label"] = paragraphs["Corps de texte"]["label"].upper().replace("È", "E").replace("Ë", "E").replace("Û", "U").replace("Ù", "U").replace("Î", "I").replace("Ï", "I").replace("Â", "A").replace("Œ", "OE")
        # proba to duplicate a line and place it randomly elsewhere, in a body region
        if rand_uniform(0, 1) < 0.1 and "Corps de texte" in paragraphs:
            labels = paragraphs["Corps de texte"]["label"].split("\n")
            duplicated_label = labels[randint(0, len(labels))]
            labels.insert(randint(0, len(labels)), duplicated_label)
            paragraphs["Corps de texte"]["label"] = "\n".join(labels)
        # proba to merge successive lines to have longer text lines in body
        if rand_uniform(0, 1) < 0.1 and "Corps de texte" in paragraphs:
            paragraphs["Corps de texte"]["label"] = larger_lines(paragraphs["Corps de texte"]["label"])
        for mode in paragraphs.keys():
            line_labels = paragraphs[mode]["label"].split("\n")
            if len(line_labels) == 0:
                print("ERROR")
            paragraphs[mode]["lines"] = list()
            for line_label in line_labels:
                if len(line_label) > 100:
                    for chunk in [line_label[i:i + max_len] for i in range(0, len(line_label), max_len)]:
                        paragraphs[mode]["lines"].append(chunk)
                else:
                    paragraphs[mode]["lines"].append(line_label)
        page_labels = {
            "raw": "",
            "begin": "",
            "sem": ""
        }
        top_limit = 0
        bottom_limit = h
        max_bottom_crop = 0
        min_top_crop = h
        has_opening = has_object = has_reference = False
        top_opening = top_object = top_reference = 0
        right_opening = right_object = right_reference = 0
        has_reference = False
        date_on_top = False
        date_alone = False
        for mode in kept_modes:
            pg = paragraphs[mode]
            if len(pg["lines"]) > nb_lines:
                pg["lines"] = pg["lines"][:nb_lines]
            nb_lines -= len(pg["lines"])
            pg_image = self.generate_typed_text_paragraph_image(pg["lines"], padding_value=255, max_pad_left_ratio=1, same_font_size=True)
            # proba to remove some interline spacing
            if rand_uniform(0, 1) < 0.1:
                pg_image = apply_transform(pg_image, Tightening(color=255, remove_proba=0.75))
            # proba to rotate text region
            if rand_uniform(0, 1) < 0.1:
                pg_image = apply_transform(pg_image, RandomRotation(degrees=10, expand=True, fill=255))
            pg["added"] = True
            if mode == 'Corps de texte':
                pg_image = resize_max(pg_image, max_height=int(0.5*h), max_width=w)
                img_h, img_w = pg_image.shape[:2]
                min_top = int(0.4*h)
                max_top = int(0.9*h - img_h)
                top = randint(min_top, max_top + 1)
                left = randint(0, int(w - img_w) + 1)
                bottom_body = top + img_h
                top_body = top
                bottom_limit = min(top, bottom_limit)
            elif mode == "PS/PJ":
                pg_image = resize_max(pg_image, max_height=int(0.03*h), max_width=int(0.9*w))
                img_h, img_w = pg_image.shape[:2]
                min_top = bottom_body
                max_top = int(min(h - img_h, bottom_body + 0.15*h))
                top = randint(min_top, max_top + 1)
                left = randint(0, int(w - img_w) + 1)
                bottom_limit = min(top, bottom_limit)
            elif mode == "Ouverture":
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                min_top = int(top_body - 0.05 * h)
                max_top = top_body - img_h
                top = randint(min_top, max_top + 1)
                left = randint(0, min(int(0.15*w), int(w - img_w)) + 1)
                has_opening = True
                top_opening = top
                right_opening = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == "Objet":
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                max_top = top_reference - img_h if has_reference else top_opening - img_h if has_opening else top_body - img_h
                min_top = int(max_top - 0.05 * h)
                top = randint(min_top, max_top + 1)
                left = randint(0, min(int(0.15*w), int(w - img_w)) + 1)
                has_object = True
                top_object = top
                right_object = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == "Reference":
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                max_top = top_object - img_h if has_object else top_opening - img_h if has_opening else top_body - img_h
                min_top = int(max_top - 0.05 * h)
                top = randint(min_top, max_top + 1)
                left = randint(0, min(int(0.15*w), int(w - img_w)) + 1)
                has_reference = True
                top_reference = top
                right_reference = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == 'Date, Lieu':
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.45 * w))
                img_h, img_w = pg_image.shape[:2]
                if h - max_bottom_crop - 10 > img_h and randint(0, 10) == 0:
                    top = randint(max_bottom_crop, h)
                    left = randint(0, w-img_w)
                else:
                    min_top = top_body - img_h
                    max_top = top_body - img_h
                    min_left = 0
                    # Check if there is anough place to put the date at the right side of opening, reference or object
                    if object_ref == ['Objet', 'Reference']:
                        have = [has_opening, has_object, has_reference]
                        rights = [right_opening, right_object, right_reference]
                        tops = [top_opening, top_object, top_reference]
                    else:
                        have = [has_opening, has_reference, has_object]
                        rights = [right_opening, right_reference, right_object]
                        tops = [top_opening, top_reference, top_object]
                    for right_r, top_r, has_r in zip(rights, tops, have):
                        if has_r:
                            if right_r + img_w >= 0.95*w:
                                max_top = min(top_r - img_h, max_top)
                                min_left = 0
                            else:
                                min_left = max(min_left, right_r+0.05*w)
                                min_top = top_r - img_h if min_top == top_body - img_h else min_top
                    if min_left != 0 and randint(0, 5) == 0:
                        min_left = 0
                        for right_r, top_r, has_r in zip(rights, tops, have):
                            if has_r:
                                max_top = min(max_top, top_r-img_h)

                    max_left = max(min_left, w - img_w)

                    # No placement found at right-side of opening, reference or object
                    if min_left == 0:
                        # place on the top
                        if randint(0, 2) == 0:
                            min_top = 0
                            max_top = int(min(0.05*h, max_top))
                            date_on_top = True
                        # place just before object/reference/opening
                        else:
                            min_top = int(max(0, max_top - 0.05*h))
                            date_alone = True
                            max_left = min(max_left, int(0.1*w))

                    min_top = min(min_top, max_top)
                    top = randint(min_top, max_top + 1)
                    left = randint(int(min_left), max_left + 1)
                    if date_on_top:
                        top_limit = max(top_limit, top + img_h)
                    else:
                        bottom_limit = min(top, bottom_limit)
                    date_right = left + img_w
                    date_bottom = top + img_h
            elif mode == "Coordonnées Expéditeur":
                max_height = min(0.25*h, bottom_limit-top_limit)
                if max_height <= 0:
                    pg["added"] = False
                    print("ko", bottom_limit, top_limit)
                    break
                pg_image = resize_max(pg_image, max_height=int(max_height), max_width=int(0.45 * w))
                img_h, img_w = pg_image.shape[:2]
                top = randint(top_limit, bottom_limit-img_h+1)
                left = randint(0, int(0.5*w-img_w)+1)
            elif mode == "Coordonnées Destinataire":
                if h - max_bottom_crop - 10 > 0.2*h and randint(0, 10) == 0:
                    pg_image = resize_max(pg_image, max_height=int(0.2*h), max_width=int(0.45 * w))
                    img_h, img_w = pg_image.shape[:2]
                    top = randint(max_bottom_crop, h)
                    left = randint(0, w-img_w)
                else:
                    max_height = min(0.25*h, bottom_limit-top_limit)
                    if max_height <= 0:
                        pg["added"] = False
                        print("ko", bottom_limit, top_limit)
                        break
                    pg_image = resize_max(pg_image, max_height=int(max_height), max_width=int(0.45 * w))
                    img_h, img_w = pg_image.shape[:2]
                    if date_alone and w - date_right - img_w > 11:
                        top = randint(0, date_bottom-img_h+1)
                        left = randint(max(int(0.5*w), date_right+10), w-img_w)
                    else:
                        top = randint(top_limit, bottom_limit-img_h+1)
                        left = randint(int(0.5*w), int(w - img_w)+1)

            bottom = top+img_h
            right = left+img_w
            min_top_crop = min(top, min_top_crop)
            max_bottom_crop = max(bottom, max_bottom_crop)
            try:
                background[top:bottom, left:right, ...] = pg_image
            except:
                pg["added"] = False
                nb_lines = 0
            pg["coords"] = {
                "top": top,
                "bottom": bottom,
                "right": right,
                "left": left
            }

            if nb_lines <= 0:
                break
        sorted_pg = order_text_regions_rimes(paragraphs.values())
        for pg in sorted_pg:
            if "added" in pg.keys() and pg["added"]:
                pg_label = "\n".join(pg["lines"])
                mode = pg["type"]
                begin_token = matching_tokens_str[mode]
                end_token = matching_tokens[begin_token]
                page_labels["raw"] += pg_label
                page_labels["begin"] += begin_token + pg_label
                page_labels["sem"] += begin_token + pg_label + end_token
        if crop:
            if min_top_crop > max_bottom_crop:
                print("KO - min > MAX")
            elif min_top_crop > h:
                print("KO - min > h")
            else:
                background = background[min_top_crop:max_bottom_crop]
        return [background, page_labels, 1]

    def stat_sem_rimes(self):
        try:
            return self.rimes_sem_stats
        except:
            stats = dict()
            for sample in self.samples:
                for pg in sample["paragraphs_label"]:
                    mode = pg["type"]
                    if mode == 'Coordonnées Expéditeur':
                        if len(pg["label"]) < 50 and "\n" not in pg["label"]:
                            mode = "Reference"
                    if mode not in stats.keys():
                        stats[mode] = 0
                    else:
                        stats[mode] += 1
            for key in stats:
                stats[key] = max(0.10, stats[key]/len(self.samples))
            self.rimes_sem_stats = stats
            return stats

    def get_paragraph_rimes(self, mode="Corps de texte", mix=False):
        while True:
            sample = self.samples[randint(0, len(self))]
            random.shuffle(sample["paragraphs_label"])
            for pg in sample["paragraphs_label"]:
                pg_mode = pg["type"]
                if pg_mode == 'Coordonnées Expéditeur':
                    if len(pg["label"]) < 50 and "\n" not in pg["label"]:
                        pg_mode = "Reference"
                if mode == pg_mode:
                    if mode == "Corps de texte" and mix:
                        return self.get_mix_paragraph_rimes(mode, min(5, len(pg["label"].split("\n"))))
                    else:
                        return pg

    def get_mix_paragraph_rimes(self, mode="Corps de texte", num_lines=10):
        res = list()
        while len(res) != num_lines:
            sample = self.samples[randint(0, len(self))]
            random.shuffle(sample["paragraphs_label"])
            for pg in sample["paragraphs_label"]:
                pg_mode = pg["type"]
                if pg_mode == 'Coordonnées Expéditeur':
                    if len(pg["label"]) < 50 and "\n" not in pg["label"]:
                        pg_mode = "Reference"
                if mode == pg_mode:
                    lines = pg["label"].split("\n")
                    res.append(lines[randint(0, len(lines))])
                    break
        return {
            "label": "\n".join(res),
            "type": mode,
        }

    @staticmethod
    def place_vtitles(tit_labels, tit_imgs,  tit_lan, background, area):
        area_left, area_top, area_right, area_bottom = area
        i_tit = 0
        if len(tit_imgs) == 0:
            return tit_labels[:i_tit], tit_imgs[:i_tit]
        width = int(np.sum([img.shape[1] for img in tit_imgs]))
        height = int(np.max([img.shape[0] for img in tit_imgs]))
        right_indent = randint(0, area_right-area_left-width+1) if area_right-area_left > width else 0
        top_indent = randint(0, area_bottom-area_top-height+1) if area_bottom-area_top > height else 0
        right = area_right - right_indent # random right indent
        top = area_top + top_indent # random top indent
        remaining_width = right - area_left
        for (label, img) in zip(tit_labels, tit_imgs):
            if img.shape[1] > remaining_width:
                break
            background[top:top + img.shape[0], right-img.shape[1]:right] = img
            right -= img.shape[1]
            remaining_width -= img.shape[1]
            i_tit += 1
        return tit_labels[:i_tit], tit_imgs[:i_tit]

    def generate_synthetic_Kangyur_page(self, background, coords, crop=False, nb_lines=8):
        syn_config = self.params["config"]["synthetic_data"]

        matching_token = KANGYUR_MATCHING_TOKENS
        page_labels = {
            "raw": [],
            "begin": ["ⓟ"],
            "sem": ["ⓟ"],
        }
        # area_top = 0 if crop else coords["top"]
        area_top = coords["top"]
        area_left = coords["left"]
        area_right = coords["right"]
        area_bottom = coords["bottom"]

        ratio_body = rand_uniform(0.8, 0.9) if syn_config['dataset_level'] == 'page' else 1.0  # 文本区域宽度占比

        complex = syn_config.get('start_with_complex_layout', False)

        if nb_lines > 0:
            nb_body_lines = randint(0, nb_lines+1) if (not crop or complex) and syn_config['dataset_level'] == 'page' else nb_lines
            nb_lt_lines = randint(0, min(3, nb_lines - nb_body_lines + 1))  # maximum 3 lines for left title
            nb_rt_lines = randint(0, min(3, nb_lines - nb_body_lines - nb_lt_lines + 1))  # maximum 3 lines for right title
            nb_body_lines = nb_lines - nb_lt_lines - nb_rt_lines

            body_labels = list()
            body_imgs = list()
            # if nb_body_lines == 0:
            #     print('debug')
            while nb_body_lines > 0:
                syn_line = self.get_printed_line_Kangyur("body")
                label = list(syn_line['label'])
                img = syn_line['img']
                nb_body_lines -= 1
                body_labels.append(label)
                body_imgs.append(img)

            lt_lan = None
            lt_labels = list()
            lt_imgs = list()
            while nb_lt_lines > 0:
                syn_line = self.get_printed_line_Kangyur("ltitle", lt_lan)
                label = list(syn_line['label'])
                img = syn_line['img']
                lt_lan = syn_line['lan']
                nb_lt_lines -= 1
                lt_labels.append(label)
                lt_imgs.append(img)

            rt_lan = None
            rt_labels = list()
            rt_imgs = list()
            while nb_rt_lines > 0:
                syn_line = self.get_printed_line_Kangyur("rtitle", rt_lan)
                label = list(syn_line['label'])
                img = syn_line['img']
                rt_lan = syn_line['lan']
                nb_rt_lines -= 1
                rt_labels.append(label)
                rt_imgs.append(img)

            max_width_body = int(np.floor(ratio_body*(area_right-area_left)))
            max_width_lt = max_width_rt = (area_right-area_left-max_width_body) // 2
            max_height = area_bottom - area_top

            """resize text lines to fit the text area width"""
            for i in range(len(body_imgs)):
                if body_imgs[i].shape[1] > max_width_body:
                    ratio = max_width_body / body_imgs[i].shape[1]
                    new_h = int(np.floor(ratio * body_imgs[i].shape[0]))
                    new_w = int(np.floor(ratio * body_imgs[i].shape[1]))
                    body_imgs[i] = cv2.resize(body_imgs[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            """resize titles to fit the height"""
            for img_list, max_width in zip([lt_imgs, rt_imgs], [max_height, max_height]):
                for i in range(len(img_list)):
                    if img_list[i].shape[0] > max_height:
                        ratio = max_height/img_list[i].shape[0]
                        new_h = int(np.floor(ratio*img_list[i].shape[0]))
                        new_w = int(np.floor(ratio*img_list[i].shape[1]))
                        img_list[i] = cv2.resize(img_list[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            """ place text line images on main text area"""
            body_top = area_top
            body_height = 0
            i_body = 0
            for (label, img) in zip(body_labels, body_imgs):
                remaining_height = area_bottom - body_top
                if img.shape[0] > remaining_height:
                    break
                indent =  0 # randint(0, min(max_width_body - img.shape[1] + 1, 30))
                background[body_top:body_top+img.shape[0], area_left+max_width_lt+indent:area_left+max_width_lt+indent+img.shape[1]] = img
                body_height += img.shape[0]
                body_top += img.shape[0]
                nb_lines -= 1
                i_body += 1

            """ labels for main text area """
            body_full_labels = {
                "raw": [],
                "begin": [],
                "sem": [],
            }
            if i_body > 0:
                for key in ["sem", "begin"]:
                    body_full_labels[key] += ['ⓜ', "ⓑ"] # symbols indicate body in main text area
                # body_full_labels["raw"] += ["\n"]
                for key in body_full_labels.keys():
                    for i in range(i_body):
                        if i != 0:
                            body_full_labels[key] += ["\n"]
                        body_full_labels[key] += body_labels[i]
                body_full_labels["sem"] += matching_token["ⓑ"] + matching_token['ⓜ']

            """place text line images on left title area"""
            lt_area = (area_left, area_top, area_left+max_width_lt, area_bottom)
            lt_labels, lt_imgs = self.place_vtitles(lt_labels, lt_imgs, lt_lan, background, lt_area)

            """labels for left title"""
            lt_full_labels = {
                "raw": [],
                "begin": [],
                "sem": [],
            }
            if len(lt_labels):
                for key in ["sem", "begin"]:
                    lt_full_labels[key] += ['ⓛ'] + ["ⓑ"]  # symbol indicates body
                if len(body_labels):
                    lt_full_labels["raw"] += ["\n"]
                for key in lt_full_labels.keys():
                    for i, lt_label in enumerate(lt_labels):
                        if i > 0:
                            lt_full_labels[key] += ["\n"]
                        lt_full_labels[key] += lt_label
                lt_full_labels["sem"] += matching_token["ⓑ"] + matching_token['ⓛ']  # symbol indicates body


            rt_area = (area_right - max_width_rt, area_top, area_right, area_bottom)
            rt_labels, rt_imgs = self.place_vtitles(rt_labels, rt_imgs, rt_lan, background, rt_area)

            """labels for right title"""
            rt_full_labels = {
                "raw": [],
                "begin": [],
                "sem": [],
            }
            if len(rt_labels):
                for key in ["sem", "begin"]:
                    rt_full_labels[key] += ['ⓡ'] + ["ⓑ"]  # symbol indicates body
                if len(body_labels) + len(lt_labels):
                    rt_full_labels["raw"] += ["\n"]
                for key in rt_full_labels.keys():
                    for i, rt_label in enumerate(rt_labels):
                        if i > 0:
                            rt_full_labels[key] += ["\n"]
                        rt_full_labels[key] += rt_label
                rt_full_labels["sem"] += matching_token["ⓑ"] + matching_token['ⓡ']

            for key in page_labels.keys():
                page_labels[key] += body_full_labels[key] + lt_full_labels[key] + rt_full_labels[key]

        if crop:
            left = area_left + max_width_lt if len(lt_labels) == 0 else area_left
            right = area_right - max_width_rt if len(rt_labels) == 0 else area_right
            if len(rt_labels) == 0 and len(body_labels) == 0:
                right -= max_width_body
            top = area_top
            bottom = body_top if len(lt_labels) == 0 and len(rt_labels) == 0 else area_bottom
            background = background[top:bottom, left: right]

        page_labels["sem"] += matching_token["ⓟ"]

        # for key in page_labels.keys():
        #     page_labels[key] = page_labels[key].strip()

        return [background, page_labels, 1]

    def generate_synthetic_iam_page(self, background, nb_lines=20, crop=False):
        page_labels = {
            "raw": "",
        }
        area_top = 0
        area_left = 0
        area_right = background.shape[1]
        area_bottom = background.shape[0]
        max_width_body = area_right - area_left

        config = self.params['config']['synthetic_data']
        width_divisor = self.params['config']['width_divisor']
        random_line_space = 'random_line_space' in config and config['random_line_space'] and nb_lines == config['max_nb_lines']
        if random_line_space:
            nb_lines = randint(int(nb_lines/2), nb_lines+1)

        img_list = []
        label_list = []

        remaining_height = area_bottom - area_top

        while nb_lines > 0:
            label, img = self.get_printed_line_iam()

            if (img.shape[1] / width_divisor) < 1. * len(label): # line is too short
                ratio = 1. * len(label) * width_divisor / img.shape[1]
                new_h = int(np.floor(ratio * img.shape[0]))
                new_w = int(np.floor(ratio * img.shape[1]))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if img.shape[1] > max_width_body: # line is too long
                ratio = max_width_body / img.shape[1]
                new_h = int(np.floor(ratio * img.shape[0]))
                new_w = int(np.floor(ratio * img.shape[1]))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if remaining_height < img.shape[0]:
                break

            img_list.append(img)
            label_list.append(label)
            remaining_height -= img.shape[0]
            nb_lines -= 1
        
        line_spaces = []
        for i in range(len(img_list)):
            if random_line_space:
                line_space = randint(0, remaining_height+1)
                line_spaces.append(line_space)
                remaining_height -= line_space
            else:
                line_spaces.append(0)
        random.shuffle(line_spaces)

        for i, (img, label) in enumerate(zip(img_list, label_list)):
            indent = randint(0, area_right-area_left-img.shape[1]+1) if config.get('random_indent', False) else 0
            # line_space = randint(0, remaining_height+1) if random_line_space else 0
            line_space = line_spaces[i]
            background[area_top+line_space:area_top+line_space+img.shape[0], area_left+indent:area_left+indent+img.shape[1]] = img
            area_top += line_space + img.shape[0]
            if len(page_labels['raw']) > 0:
                page_labels['raw'] += '\n'
            page_labels['raw'] += label
            # remaining_height -= line_space

        if crop:
            background = background[:area_top]

        page_labels["sem"] = page_labels['raw']
        page_labels['begin'] = page_labels['raw']


        return [background, page_labels, 1]

    def generate_synthetic_read2016_page(self, background, coords, side="left", nb_lines=20, crop=False):
        config = self.params["config"]["synthetic_data"]
        two_column = False
        matching_token = READ_MATCHING_TOKENS
        page_labels = {
            "raw": "",
            "begin": "ⓟ",
            "sem": "ⓟ",
        }
        area_top = 0 if crop else coords["top"]
        area_left = coords["left"]
        area_right = coords["right"]
        area_bottom = coords["bottom"]
        num_page_text_label = str(randint(0, 1000))
        num_page_sample = self.generate_typed_text_line_image(num_page_text_label)
        # num_page_text_label = num_page_sample['text']
        num_page_img = num_page_sample['img']

        if side == "left":
            background[area_top:area_top+num_page_img.shape[0], area_left:area_left+num_page_img.shape[1]] = num_page_img
        else:
            background[area_top:area_top + num_page_img.shape[0], area_right-num_page_img.shape[1]:area_right] = num_page_img
        for key in ["sem", "begin"]:
            page_labels[key] += "ⓝ"
        for key in page_labels.keys():
            page_labels[key] += num_page_text_label
        page_labels["sem"] += matching_token["ⓝ"]
        nb_lines -= 1
        area_top = area_top + num_page_img.shape[0] + randint(1, 20)
        ratio_ann = rand_uniform(0.6, 0.7)
        while nb_lines > 0:
            nb_body_lines = randint(1, nb_lines+1)
            max_ann_lines = min(nb_body_lines, nb_lines-nb_body_lines)
            body_labels = list()
            body_imgs = list()
            while nb_body_lines > 0:
                current_nb_lines = 1
                label, img = self.get_printed_line_read_2016("body")
                nb_body_lines -= current_nb_lines
                body_labels.append(label)
                body_imgs.append(img)
            nb_ann_lines = randint(0, min(6, max_ann_lines+1))
            ann_labels = list()
            ann_imgs = list()
            while nb_ann_lines > 0:
                current_nb_lines = 1
                label, img = self.get_printed_line_read_2016("annotation")
                nb_ann_lines -= current_nb_lines
                ann_labels.append(label)
                ann_imgs.append(img)
            max_width_body = int(np.floor(ratio_ann*(area_right-area_left)))
            max_width_ann = area_right-area_left-max_width_body
            for img_list, max_width in zip([body_imgs, ann_imgs], [max_width_body, max_width_ann]):
                for i in range(len(img_list)):
                    if img_list[i].shape[1] > max_width:
                        ratio = max_width/img_list[i].shape[1]
                        new_h = int(np.floor(ratio*img_list[i].shape[0]))
                        new_w = int(np.floor(ratio*img_list[i].shape[1]))
                        img_list[i] = cv2.resize(img_list[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            body_top = area_top
            body_height = 0
            i_body = 0
            for (label, img) in zip(body_labels, body_imgs):
                remaining_height = area_bottom - body_top
                if img.shape[0] > remaining_height:
                    nb_lines = 0
                    break
                indent = randint(0, max_width_body-img.shape[1]+1) if config.get('random_indent', False) else 0
                background[body_top:body_top+img.shape[0], area_left+max_width_ann+indent:area_left+max_width_ann+indent+img.shape[1]] = img
                body_height += img.shape[0]
                body_top += img.shape[0]
                nb_lines -= 1
                i_body += 1

            ann_height = int(np.sum([img.shape[0] for img in ann_imgs]))
            ann_top = area_top + randint(0, body_height-ann_height+1) if ann_height < body_height else area_top
            largest_ann = max([a.shape[1] for a in ann_imgs]) if len(ann_imgs) > 0 else max_width_ann
            pad_ann = randint(0, max_width_ann-largest_ann+1) if max_width_ann > largest_ann else 0

            ann_label_blocks = [list(), ]
            i_ann = 0
            ann_height = 0
            for (label, img) in zip(ann_labels, ann_imgs):
                remaining_height = body_top - ann_top
                if img.shape[0] > remaining_height:
                    break
                indent = randint(0, max_width_ann - img.shape[1] + 1) if config.get('random_indent', False) else 0
                background[ann_top:ann_top+img.shape[0], area_left+pad_ann+indent:area_left+pad_ann+indent+img.shape[1]] = img
                ann_height += img.shape[0]
                ann_top += img.shape[0]
                nb_lines -= 1
                two_column = True
                ann_label_blocks[-1].append(ann_labels[i_ann])
                i_ann += 1
                if randint(0, 10) == 0:
                    ann_label_blocks.append(list())
                    ann_top += randint(0, max(15, body_top-ann_top-20))

            area_top = area_top + max(ann_height, body_height)
            if nb_lines > 0:
               area_top += randint(25, 100)

            ann_full_labels = {
                "raw": "",
                "begin": "",
                "sem": "",
            }
            for ann_label_block in ann_label_blocks:
                if len(ann_label_block) > 0:
                    for key in ["sem", "begin"]:
                        ann_full_labels[key] += "ⓐ"
                    ann_full_labels["raw"] += "\n"
                    for key in ann_full_labels.keys():
                        ann_full_labels[key] += "\n".join(ann_label_block)
                    ann_full_labels["sem"] += matching_token["ⓐ"]

            body_full_labels = {
                "raw": "",
                "begin": "",
                "sem": "",
            }
            if i_body > 0:
                for key in ["sem", "begin"]:
                    body_full_labels[key] += "ⓑ"
                body_full_labels["raw"] += "\n"
                for key in body_full_labels.keys():
                    body_full_labels[key] += "\n".join(body_labels[:i_body])
                body_full_labels["sem"] += matching_token["ⓑ"]

            section_labels = dict()
            for key in ann_full_labels.keys():
                section_labels[key] = ann_full_labels[key] + body_full_labels[key]
            for key in section_labels.keys():
                if section_labels[key] != "":
                    if key in ["sem", "begin"]:
                        section_labels[key] = "ⓢ" + section_labels[key]
                    if key == "sem":
                        section_labels[key] = section_labels[key] + matching_token["ⓢ"]
            for key in page_labels.keys():
                page_labels[key] += section_labels[key]

        if crop:
            background = background[:area_top]

        page_labels["sem"] += matching_token["ⓟ"]

        for key in page_labels.keys():
            page_labels[key] = page_labels[key].strip()

        return [background, page_labels, 2 if two_column else 1]

    def get_n_consecutive_lines_read_2016(self, n=1, mode="body"):
        while True:
            sample = self.samples[randint(0, len(self))]
            paragraphs = list()
            for page in sample["pages_label"]:
                paragraphs.extend(page["paragraphs"])
                random.shuffle(paragraphs)
                for pg in paragraphs:
                    if ((mode == "body" and pg["mode"] == "body") or
                        (mode == "ann" and pg["mode"] == "annotation")) and len(pg["lines"]) >= n:
                        line_idx = randint(0, len(pg["lines"])-n+1)
                        lines = pg["lines"][line_idx:line_idx+n]
                        label = "\n".join([l["text"] for l in lines])
                        top = min([l["top"] for l in lines])
                        bottom = max([l["bottom"] for l in lines])
                        left = min([l["left"] for l in lines])
                        right = max([l["right"] for l in lines])
                        img = sample["img"][top:bottom, left:right]
                        return label, img

    def get_printed_line_read_2016(self, mode="body"):
        while True:
            sample = self.samples[randint(0, len(self))]
            for page in sample["pages_label"]:
                paragraphs = list()
                paragraphs.extend(page["paragraphs"])
                random.shuffle(paragraphs)
                for pg in paragraphs:
                    random.shuffle(pg["lines"])
                    for line in pg["lines"]:
                        if (mode == "body" and len(line["text"]) > 5) or (mode == "annotation" and len(line["text"]) < 15 and not line["text"].isdigit()):
                            label = line["text"]
                            config = self.params["config"]["synthetic_data"]["config"]
                            if 'random_permutation' in config and rand() < config['random_permutation']['proba']:
                                label = random_permutation(label)
                            syn_sample = self.generate_typed_text_line_image(label)
                            img = syn_sample['img']
                            label = GenericDataset.preprocess_sample_label(syn_sample, self.params)['label']
                            return label, img

    def get_printed_line_iam(self):
        while True:
            sample = self.samples[randint(0, len(self))]
            lines = list()
            lines.extend(sample['raw_line_seg_label'])
            random.shuffle(lines)
            for line in lines:
                label = line["text"]
                config = self.params["config"]["synthetic_data"]["config"]
                if 'random_permutation' in config and rand() < config['random_permutation']['proba']:
                    label = random_permutation(label)
                syn_sample = self.generate_typed_text_line_image(label)
                img = syn_sample['img']
                label = GenericDataset.preprocess_sample_label(syn_sample, self.params)['label']
                return label, img

    def double_column_cjk_line(self, text):
        syn_config = self.params["config"]["synthetic_data"]["config"]
        min_len = syn_config['min_len_sub_line']
        unchanged_text = text
        text = [c for c in text if c not in [' ']] # delete white spaces for double-column vertical Chinese text
        if len(text) >= min_len*3: # double column text
            dc_len = randint(min_len, len(text)//3+1)
            sc_len1 = randint(0, len(text) - dc_len*2+1)
            if sc_len1 < min_len:
                sc_len1 = 0
                sc_len4 = len(text) - dc_len * 2
            elif len(text) - sc_len1 - dc_len * 2 < min_len:
                dc_len += (len(text) - sc_len1 - dc_len * 2) // 2
                sc_len1 = len(text) - dc_len * 2
                sc_len4 = 0
            else:
                sc_len4 = len(text) - sc_len1 - dc_len * 2

            subline_lens = [sc_len1, dc_len, dc_len, sc_len4]  # single-column + 2*double-column + single-column
            imgs = [None] * 4
            label = []
            bboxes = [None] * 4
            font_size_wide = syn_config['font_size']['cjk'][1]
            start = 0
            for i, sub_len in enumerate(subline_lens):
                if sub_len == 0:
                    continue
                font_size = font_size_wide if i % 3 == 0 else font_size_wide // 2
                sub_line = self.generate_typed_text_line_image(text[start: start+sub_len], 'cjk', 'ttb', font_size)
                imgs[i] = sub_line['img']
                if len(label):
                    label += ['\n']
                label += sub_line['char_text']
                bboxes[i] = sub_line['bbox']
                start += sub_len
            subline_width = [img.shape[1] if img is not None else 0 for img in imgs]
            subline_height = [img.shape[0] if img is not None else 0 for img in imgs]
            width = max(subline_width[0], subline_width[1] + subline_width[2], subline_width[3])
            height = subline_height[0] + max(subline_height[1], subline_height[2]) + subline_height[3]
            sample_img = [img for img in imgs if img is not None][0]
            line_img = np.ones((height, width, sample_img.shape[2]), dtype=sample_img.dtype) * 255
            """place sub lines"""
            top = 0
            right = line_img.shape[1]
            for i in range(len(imgs)):
                if imgs[i] is None:
                    continue
                indent = randint(0, line_img.shape[1]-imgs[i].shape[1]+1) if i % 3 == 0 and line_img.shape[1] > imgs[i].shape[1] else 0
                line_img[top: top+imgs[i].shape[0], right-indent-imgs[i].shape[1]:right-indent] = imgs[i]
                if i % 3 == 0: # wide subline
                    top += imgs[i].shape[0]
                elif i % 3 == 1: # first subline
                    right -= imgs[i].shape[1]
                elif i % 3 == 2: # second subline
                    right = line_img.shape[1]
                    top += max(imgs[i].shape[0], imgs[i-1].shape[0])
                bl, bt, br, bb = bboxes[i]
                bboxes[i] = (right-indent-imgs[i].shape[1]+bl, top+bt, right-indent-imgs[i].shape[1]+br, top+bb)
            bboxes = [b for b in bboxes if b is not None]
            return {
                'img': line_img,
                'bbox': bboxes,
                'text': ''.join(label),
                'char_text': label,
            }
        return self.generate_typed_text_line_image(unchanged_text, 'cjk', 'ttb')


    def get_printed_line_Kangyur(self, mode="body", lan=None, double_column=False):
        while True:
            sample = self.samples[randint(0, len(self))]
            merged_line_labels = copy.deepcopy(sample['unchanged_line_label']) # curial
            merged_line_labels, merged_line_lan = merge_line_labels(merged_line_labels, sample['line_lan'], 'cjk')
            assert len(merged_line_labels) == len(merged_line_lan)
            indices = list(range(0, len(merged_line_labels)))
            random.shuffle(indices)
            for i in indices:
                is_str = isinstance(merged_line_labels[i], str)
                line_text = merged_line_labels[i]
                line_lan = merged_line_lan[i]
                if len(line_text) == 0:
                    continue
                if lan is not None and line_lan != lan:
                    continue
                config = self.params["config"]["synthetic_data"]["config"]
                if 'random_permutation' in config and rand() < config['random_permutation']['proba']:
                    line_text = random_permutation(line_text)
                if mode == 'body':
                    syn_sample = self.generate_typed_text_line_image(line_text, line_lan, 'ltr')
                elif 'title' in mode and line_lan == 'cjk': # vertical cjk text
                    syn_sample = self.double_column_cjk_line(line_text) # double column text line
                else: # vertical Tibetan text
                    if len(line_text) > 30:
                        ri = randint(0, len(line_text)-30+1)
                        line_text = line_text[ri:ri+30]
                        line_text = strip_charlist(line_text, [' ', '\xa0'])
                    syn_sample = self.generate_typed_text_line_image(line_text, line_lan, 'ltr')
                    """rotate 90deg clock wise"""
                    syn_sample['img'] = syn_sample['img'].transpose(1, 0, 2)
                    syn_sample['img'] = np.flip(syn_sample['img'], axis=1)
                    bbox = syn_sample['bbox']
                    syn_sample['bbox'] = (bbox[3], bbox[0], bbox[3] + bbox[3] - bbox[1], bbox[2])
                img = syn_sample['img']
                bbox = syn_sample['bbox']
                if is_str:
                    del syn_sample['char_text']
                label = GenericDataset.preprocess_sample_label(syn_sample, self.params)['label']
                return {
                    'img': img,
                    'bbox': bbox,
                    'label': label,
                    'lan': line_lan
                }



    def generate_typed_text_line_image(self, text, lan='latin', writ_dir='ltr', font_size=-1):
        return generate_typed_text_line_image(text, lan, writ_dir, self.params["config"]["synthetic_data"]["config"], font_size=font_size)

    def generate_typed_text_paragraph_image(self, texts, padding_value=255, max_pad_left_ratio=0.1, same_font_size=False):
        config = self.params["config"]["synthetic_data"]["config"]
        if same_font_size:
            images = list()
            txt_color = config["text_color_default"]
            bg_color = config["background_color_default"]
            font_size = randint(config["font_size_min"], config["font_size_max"] + 1)
            for text in texts:
                font_path = config["valid_fonts"][randint(0, len(config["valid_fonts"]))]
                fnt = ImageFont.truetype(font_path, font_size)
                text_width, text_height = fnt.getsize(text)
                padding_top = int(rand_uniform(config["padding_top_ratio_min"], config["padding_top_ratio_max"]) * text_height)
                padding_bottom = int(rand_uniform(config["padding_bottom_ratio_min"], config["padding_bottom_ratio_max"]) * text_height)
                padding_left = int(rand_uniform(config["padding_left_ratio_min"], config["padding_left_ratio_max"]) * text_width)
                padding_right = int(rand_uniform(config["padding_right_ratio_min"], config["padding_right_ratio_max"]) * text_width)
                padding = [padding_top, padding_bottom, padding_left, padding_right]
                images.append(generate_typed_text_line_image_from_params(text, fnt, bg_color, txt_color, config["color_mode"], padding))
        else:
            images = [self.generate_typed_text_line_image(t) for t in texts]

        max_width = max([img.shape[1] for img in images])

        padded_images = [pad_image_width_random(img, max_width, padding_value=padding_value, max_pad_left_ratio=max_pad_left_ratio) for img in images]
        return np.concatenate(padded_images, axis=0)



class OCRCollateFunction:
    """
    Merge samples data to mini-batch data for OCR task
    """

    def __init__(self, config):
        self.img_padding_value = float(config["padding_value"])
        self.label_padding_value = config["padding_token"]
        self.config = config

    def __call__(self, batch_data):
        names = [batch_data[i]["name"] for i in range(len(batch_data))]
        # ids = [int(batch_data[i]["name"].split("/")[-1].split("_")[-1].split(".")[0]) for i in range(len(batch_data))]

        try:
            ids = [int(os.path.splitext(os.path.basename(batch_data[i]['name']))[0].split('_')[-1]) for i in range(len(batch_data))]
        except:
            # """藏文数据的输入图片的名字中间带有“.”号，需删除"""
            # ids = [int(os.path.splitext(os.path.basename(batch_data[i]['name']))[0].replace('.', '').split('_')[-1]) for i in
            #        range(len(batch_data))]
            ids = [os.path.splitext(os.path.basename(batch_data[i]['name']))[0] for i in range(len(batch_data))]

        paths = [batch_data[i]['path'] for i in range(len(batch_data))]

        applied_da = [batch_data[i]["applied_da"] for i in range(len(batch_data))]

        labels = [batch_data[i]["token_label"] for i in range(len(batch_data))]
        labels = pad_sequences_1D(labels, padding_value=self.label_padding_value)
        labels = torch.tensor(labels).long()
        # reverse_labels = [[batch_data[i]["token_label"][0], ] + batch_data[i]["token_label"][-2:0:-1] + [batch_data[i]["token_label"][-1], ] for i in range(len(batch_data))]
        # reverse_labels = pad_sequences_1D(reverse_labels, padding_value=self.label_padding_value)
        # reverse_labels = torch.tensor(reverse_labels).long()
        labels_len = [batch_data[i]["label_len"] for i in range(len(batch_data))]
        # start = [batch_data[i]["start"] for i in range(len(batch_data))]
        # num_sil = [batch_data[i]["num_sil"] for i in range(len(batch_data))]

        raw_labels = [batch_data[i]["label"] for i in range(len(batch_data))]
        unchanged_labels = [batch_data[i]["unchanged_label"] for i in range(len(batch_data))]

        nb_cols = [batch_data[i]["nb_cols"] for i in range(len(batch_data))]
        nb_lines = [batch_data[i]["nb_lines"] for i in range(len(batch_data))]
        # nb_entities = [batch_data[i]["nb_entities"] for i in range(len(batch_data))]
        line_raw = [batch_data[i]["line_label"] for i in range(len(batch_data))]
        line_token = [batch_data[i]["token_line_label"] for i in range(len(batch_data))]
        pad_line_token = list()
        line_len = [batch_data[i]["line_label_len"] for i in range(len(batch_data))]
        for i in range(max(nb_lines)):
            current_lines = [line_token[j][i] if i < nb_lines[j] else [self.label_padding_value] for j in range(len(batch_data))]
            pad_line_token.append(torch.tensor(pad_sequences_1D(current_lines, padding_value=self.label_padding_value)).long())
            for j in range(len(batch_data)):
                if i >= nb_lines[j]:
                    line_len[j].append(0)
        line_len = [i for i in zip(*line_len)]

        # nb_words = [batch_data[i]["nb_words"] for i in range(len(batch_data))]
        # word_raw = [batch_data[i]["word_label"] for i in range(len(batch_data))]
        # word_token = [batch_data[i]["token_word_label"] for i in range(len(batch_data))]
        # pad_word_token = list()
        # word_len = [batch_data[i]["word_label_len"] for i in range(len(batch_data))]
        # for i in range(max(nb_words)):
        #     current_words = [word_token[j][i] if i < nb_words[j] else [self.label_padding_value] for j in range(len(batch_data))]
        #     pad_word_token.append(torch.tensor(pad_sequences_1D(current_words, padding_value=self.label_padding_value)).long())
        #     for j in range(len(batch_data)):
        #         if i >= nb_words[j]:
        #             word_len[j].append(0)
        # word_len = [i for i in zip(*word_len)]

        padding_mode = self.config["padding_mode"] if "padding_mode" in self.config else "br" # br: bottom right
        imgs = [batch_data[i]["img"] for i in range(len(batch_data))]
        imgs_shape = [batch_data[i]["img_shape"] for i in range(len(batch_data))]
        imgs_reduced_shape = [batch_data[i]["img_reduced_shape"] for i in range(len(batch_data))]
        imgs_position = [batch_data[i]["img_position"] for i in range(len(batch_data))]
        imgs_reduced_position= [batch_data[i]["img_reduced_position"] for i in range(len(batch_data))]
        imgs = pad_images(imgs, padding_value=self.img_padding_value, padding_mode=padding_mode)
        imgs = torch.tensor(imgs).float().permute(0, 3, 1, 2)

        formatted_batch_data = {
            "names": names,
            'paths': paths,
            "ids": ids,
            "nb_lines": nb_lines,
            # 'nb_entities': nb_entities,
            "nb_cols": nb_cols,
            "labels": labels,
            # 'start': start,
            # 'num_sil': num_sil,
            "labels_len": labels_len,
            # "reverse_labels": reverse_labels,
            "raw_labels": raw_labels,
            "unchanged_labels": unchanged_labels,
            "imgs": imgs,
            "imgs_shape": imgs_shape,
            "imgs_reduced_shape": imgs_reduced_shape,
            "imgs_position": imgs_position,
            "imgs_reduced_position": imgs_reduced_position,
            "line_raw": line_raw,
            "line_labels": pad_line_token,
            "line_labels_len": line_len,
            # "nb_words": nb_words,
            # "word_raw": word_raw,
            # "word_labels": pad_word_token,
            # "word_labels_len": word_len,
            "applied_da": applied_da
        }

        return formatted_batch_data


def generate_typed_text_line_image(text, lan, writ_dir, config, bg_color=(255, 255, 255), txt_color=(0, 0, 0), font_size=-1):
    # if 'random_permutation' in config:
    #     text = random_permutation(text, config['random_permutation'])

    if "text_color_default" in config:
        txt_color = config["text_color_default"]
    if "background_color_default" in config:
        bg_color = config["background_color_default"]

    font_path = config["valid_fonts"][lan][randint(0, len(config["valid_fonts"][lan]))]

    if font_size <= 0:
        font_size_min, font_size_max = config["font_size"][lan]
        font_size = randint(font_size_min, font_size_max+1)

    layout_engine = ImageFont.Layout.RAQM if config['lan'].lower() == 'bo' else None
    fnt = ImageFont.truetype(font_path, font_size, layout_engine=layout_engine)

    label = in_font_chars(text, font_path) # exclude characters that dont have glyphs in the font
    # label = strip_charlist(label)
    if lan == 'cjk' and writ_dir == 'ttb':
        label = [l for l in label if l not in [' ', '\xa0']] # disable spaces for vertical Chinese text
    raw_text = ''.join(label)

    anchr = 'la' if writ_dir == 'ltr' else 'lt'

    left, top, text_width, text_height = fnt.getbbox(raw_text, direction=writ_dir, anchor=anchr)

    # text_width = right - left
    # text_height = bottom - top
    if writ_dir == 'ltr':
        padding_top = get_paddings(config["padding_top_ratio_min"], config["padding_top_ratio_max"], text_height)
        padding_bottom = get_paddings(config["padding_bottom_ratio_min"], config["padding_bottom_ratio_max"], text_height)
        padding_left = get_paddings(config["padding_left_ratio_min"], config["padding_left_ratio_max"], text_width)
        padding_right = get_paddings(config["padding_right_ratio_min"], config["padding_right_ratio_max"], text_width)
    else: # vertical text
        padding_top = get_paddings(config["padding_left_ratio_min"], config["padding_left_ratio_max"], text_height)
        padding_bottom = get_paddings(config["padding_right_ratio_min"], config["padding_right_ratio_max"], text_height)
        padding_left = get_paddings(config["padding_top_ratio_min"], config["padding_top_ratio_max"], text_width)
        padding_right = get_paddings(config["padding_bottom_ratio_min"], config["padding_bottom_ratio_max"], text_width)

    padding = [padding_top, padding_bottom, padding_left, padding_right]

    img, bbox = generate_typed_text_line_image_from_params(raw_text, fnt, bg_color, txt_color, config["color_mode"], padding, writ_dir, anchr)

    # max_line_width = config.get('max_line_width', 10000)
    nb_pixel_min = config.get('pixels_per_char_min', 0.)
    nb_pixel_max = config.get('pixels_per_char_max', 10000)

    # if img.width > 4000:
    #     print('debug')

    if writ_dir == 'ltr' and img.width < nb_pixel_min * len(label):
        w = int(nb_pixel_min * len(label)) + padding_left + padding_right
        h = int((w / img.width) * img.height)
        img = img.resize((w, h))
    if writ_dir == 'ltr' and img.width > nb_pixel_max * len(label):
        w = int(nb_pixel_max * len(label)) + padding_left + padding_right
        h = int((w / img.width) * img.height)
        img = img.resize((w, h))
    if writ_dir == 'ttb' and img.height < nb_pixel_min * len(label):
        h = int(nb_pixel_min * len(label)) + padding_top + padding_bottom
        w = int((h / img.height) * img.width)
        img = img.resize((w, h))
    if writ_dir == 'ttb' and img.height > nb_pixel_max * len(label):
        h = int(nb_pixel_max * len(label)) + padding_top + padding_bottom
        w = int((h / img.height) * img.width)
        img = img.resize((w, h))

    sample = {
        'text': ''.join(label),
        'img': np.array(img),
        'bbox': bbox,
    }
    if isinstance(text, list):
        sample['char_text'] = label

    return sample


def generate_typed_text_line_image_from_params(text, font, bg_color, txt_color, color_mode, padding, writ_dir, anchor):
    padding_top, padding_bottom, padding_left, padding_right = padding
    x, y, text_width, text_height = font.getbbox(text, direction=writ_dir, anchor=anchor)
    img_height = padding_top + padding_bottom + text_height
    img_width = padding_left + padding_right + text_width
    img = Image.new(color_mode, (img_width, img_height), color=bg_color)
    d = ImageDraw.Draw(img)
    d.text((padding_left, padding_top), text, font=font, fill=txt_color, spacing=0, direction=writ_dir, anchor=anchor)
    bbox = (padding_left+x, padding_top+y, text_width-x, text_height-y)
    return img, bbox

def get_paddings(ratio_min, ratio_max, size):
    if ratio_max > ratio_min:
        return int(rand_uniform(ratio_min, ratio_max)*size)
    else:
        return 0

def get_valid_fonts(alphabet=None, font_dir=None):
    valid_fonts = {}
    font_dir = os.path.join("../../../Fonts", font_dir) if font_dir else "../../../Fonts"
    for fold_detail in os.walk(font_dir):
        if fold_detail[2]:
            for font_name in fold_detail[2]:
                if ".ttf" not in font_name.lower() and ".otf" not in font_name.lower():
                    continue
                font_path = os.path.join(fold_detail[0], font_name)
                if alphabet is not None:
                    for lan in alphabet:
                        matched_chars = in_font_chars(alphabet[lan], font_path)
                        if len(matched_chars) / len(alphabet[lan]) >= 0.95:
                            if lan not in valid_fonts:
                                valid_fonts[lan] = list()
                            valid_fonts[lan].append(font_path)
                else:
                    valid_fonts.append(font_path)
    return valid_fonts


def char_in_font(unicode_char, font_path):
    with TTFont(font_path) as font:
        for cmap in font['cmap'].tables:
            if cmap.isUnicode():
                if ord(unicode_char) in cmap.cmap:
                    return True
    return False

def in_font_chars(text, font_path):
    chrs = []
    with TTFont(font_path) as font:
        for chr in text:
            comp_match = 0
            for comp in chr:
                for cmap in font['cmap'].tables:
                    if cmap.isUnicode() and ord(comp) in cmap.cmap:
                        comp_match += 1
                        break
            if comp_match == len(chr):
                chrs.append(chr)
    return chrs


def random_permutation(text):
    is_str = isinstance(text, str)
    text = np.random.permutation(list(text)).tolist()
    text = ''.join(text) if is_str else text
    return text


def randomized_line(text, config):
    is_str = isinstance(text, str)
    text = list(text)
    if 'random_permutation' in config and rand() < config['random_permutation']['proba']:
        text = np.random.permutation(text).tolist()
    if 'random_spacing' in config:
        min_space = config['random_spacing']['min']
        max_space = config['random_spacing']['max']
        text_split = split_char_list(text, [' '], retain_zero_lens=False)
        new_text = list()
        for i, ts in enumerate(text_split):
            if len(new_text):
                ns = randint(min_space, max_space+1)
                new_text += [' '] * ns
            new_text += ts
        text = new_text
    text = ''.join(text) if is_str else text
    return text


def merge_line_labels(line_labels, line_lans, lan_to_merge):
    merged_line_labels = []
    merged_line_lans = []
    for line_label, line_lan in zip(line_labels, line_lans):
        if len(merged_line_lans) and line_lan == merged_line_lans[-1] == lan_to_merge:
            merged_line_labels[-1] += line_label
        else:
            merged_line_labels.append(line_label)
            merged_line_lans.append(line_lan)
    return merged_line_labels, merged_line_lans

