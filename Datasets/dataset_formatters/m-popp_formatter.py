#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Leer Mao
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
import cv2

from Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter
import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import shutil
import json


# Layout begin-token to end-token
SEM_MATCHING_TOKENS = {
            "ⓑ": "Ⓑ",  # main body of text
            'ⓝ': 'Ⓝ',  # name of married couple
            'ⓜ': 'Ⓜ',  # marriage record
            "ⓟ": "Ⓟ",  # page
        }



class MPOPP_DatasetFormatter(OCRDatasetFormatter):
    def __init__(self, level, set_names=["train", "valid", "test"], dpi=300, end_token=True, sem_token=True, named_entity=False, encoding_type=1, data_root=None):
        super(MPOPP_DatasetFormatter, self).__init__("M-POPP", level, f"_sem{encoding_type}" if sem_token else "", set_names, data_root)

        self.map_datasets_files.update({
            "M-POPP": {
                # (350 for train, 50 for validation and 50 for test)
                "page": {
                    "arx_files": ["m-popp_datasets.zip"],
                    "needed_files": [],
                    "format_function": self.format_mpopp_page,
                },
                # (169 for train, 24 for validation and 24 for test)
                "double_page": {
                    "arx_files": ["m-popp_datasets.zip"],
                    "needed_files": [],
                    "format_function": self.format_mpopp_double_page,
                }
            }
        })
        self.dpi = dpi
        self.end_token = end_token
        self.sem_token = sem_token
        self.named_entity = named_entity
        self.encoding_type = encoding_type
        self.matching_token = SEM_MATCHING_TOKENS

    def init_format(self):
        super().init_format()
        os.rename(os.path.join(self.temp_fold, "m-popp_datasets", 'handwritten', 'images', "train"), os.path.join(self.temp_fold, "train"))
        os.rename(os.path.join(self.temp_fold, "m-popp_datasets", 'handwritten', 'images', "valid"), os.path.join(self.temp_fold, "valid"))
        os.rename(os.path.join(self.temp_fold, "m-popp_datasets", 'handwritten', 'images', "test"), os.path.join(self.temp_fold, "test"))
        os.rename(os.path.join(self.temp_fold, "m-popp_datasets", 'handwritten', 'labels'), os.path.join(self.temp_fold, "labels"))

        shutil.rmtree(os.path.join(self.temp_fold, "m-popp_datasets"))


    def preformat_mpopp(self):
        """
        Extract all information from M-POPP dataset and correct some mistakes
        """

        dataset = {
            "train": list(),
            "valid": list(),
            "test": list(),
        }
        json_file_path = os.path.join(self.temp_fold, 'labels', f'labels-handwritten-encoding-{self.encoding_type}.json')
        with open(json_file_path, 'r', encoding='utf-8') as f:
            gt = json.load(f)['ground_truth']
        if not self.named_entity:
            with open('M-POPP_NEs.json', 'r', encoding='utf-8') as f:
                to_exclude = json.load(f)['named_entity_set']
        else:
            to_exclude = []

        for set_name in ["train", "valid", "test"]:
            for img_name, img_annot in gt[set_name].items():
                img_path = os.path.join(self.temp_fold, set_name, img_name)
                img = cv2.imread(img_path)
                label = ''.join([c for c in img_annot['text'] if c not in to_exclude])
                dataset[set_name].append({
                    'label': label,
                    'nb_cols': img_annot['nb_cols'],
                    'img_path': img_path,
                    'width': img.shape[1],
                    'height': img.shape[0]
                })
        return dataset

    def format_mpopp_page(self):
        """
        Format the M-POPP dataset at single-page level
        """
        dataset = self.preformat_mpopp()
        for set_name in ["train", "valid", "test"]:
            for i, page in enumerate(dataset[set_name]):
                new_img_name = "{}_{}.jpeg".format(set_name, i)
                new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)
                self.load_resize_save(page["img_path"], new_img_path, 300, self.dpi)
                new_txt_path =  os.path.join(self.target_fold_path, set_name, os.path.splitext(new_img_name)[0]+'.txt')
                with open(new_txt_path, 'w', encoding='utf-8') as f:
                    f.write(page['label'])
                side = os.path.splitext(os.path.basename(page['img_path']))[0].split('-')[-1].lower()
                assert side in ['left', 'right']

                page_label = {
                    "text": page['label'],
                    # "paragraphs": paragraphs,
                    "nb_cols": page['nb_cols'],
                    "side": side,
                    "page_width": int(np.array(Image.open(new_img_path)).shape[1])
                }

                self.gt[set_name][new_img_name] = {
                    "text": page['label'],
                    "nb_cols": page['nb_cols'],
                    "pages": [page_label, ],
                }
                self.charset = self.charset.union(set(page["label"]))
        return

    def format_mpopp_double_page(self):
        """
        Format the M-POPP dataset at double-page level
        """
        pass

    def delete_named_entities(self):
        pass

    def add_tokens_in_charset(self):
        """
        Add layout tokens to the charset
        """
        if self.sem_token:
            if self.end_token:
                self.charset = self.charset.union(set("ⓢⓑⓐⓝⓈⒷⒶⓃⓟⓅ"))
            else:
                self.charset = self.charset.union(set("ⓢⓑⓐⓝⓟ"))

    def group_by_page_number(self, dataset):
        """
        Group page data by pairs of successive pages
        """
        pass

    def update_label(self, label, start_token):
        """
        Delete named entity tokens in transcription
        """
        if self.sem_token:
            if self.end_token:
                return start_token + label + self.matching_token[start_token]
            else:
                return start_token + label
        return label

import re

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002700-\U000027BF"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)





data_root = 'c:/myDatasets/DAN'

if __name__ == "__main__":
    MPOPP_DatasetFormatter("page", sem_token=True, named_entity=True, encoding_type=1, data_root=data_root).format()
    # MPOPP_DatasetFormatter("page", sem_token=False, data_root=data_root).format()
    # MPOPP_DatasetFormatter("double_page", sem_token=True, data_root=data_root).format()
    # MPOPP_DatasetFormatter("double_page", sem_token=False, data_root=data_root).format()