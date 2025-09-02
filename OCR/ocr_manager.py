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

from basic.generic_training_manager import GenericTrainingManager
from basic.generic_dataset_manager import GenericDataset
import os
from PIL import Image
import pickle
import numpy as np
from basic.utils import randint, rand, rand_uniform
# from basic.utils import split_char_list, syllable_delims
from Datasets.dataset_formatters.utils_dataset import get_sorted_lan_dict
import cv2


class OCRManager(GenericTrainingManager):
    def __init__(self, params):
        super(OCRManager, self).__init__(params)
        if self.dataset is not None:
            self.params["model_params"]["vocab_size"] = len(self.dataset.class_set)

    def generate_syn_line_dataset(self, name):
        """
        Generate synthetic line dataset from currently loaded dataset
        """
        dataset_name = list(self.params['dataset_params']["datasets"].keys())[0]
        path = os.path.join(os.path.dirname(self.params['dataset_params']["datasets"][dataset_name]), name)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        # charset = set()
        charset = self.dataset.charset
        dataset = None
        gt = {
            "train": dict(),
            "valid": dict(),
            "test": dict()
        }
        for set_name in ["train", "valid", "test"]:
            set_path = os.path.join(path, set_name)
            os.makedirs(set_path, exist_ok=True)
            if set_name == "train":
                dataset = self.dataset.train_dataset
            elif set_name == "valid":
                dataset = self.dataset.valid_datasets["{}-valid".format(dataset_name)]
            elif set_name == "test":
                self.dataset.generate_test_loader("{}-test".format(dataset_name), [(dataset_name, "test"), ])
                dataset = self.dataset.test_datasets["{}-test".format(dataset_name)]

            samples = list()
            for sample in dataset.samples:
                for li, line_label in enumerate(sample["unchanged_line_label"]):
                    # lan_dict = get_sorted_lan_dict(''.join(line_label))
                    samples.append({
                        'path': sample['path'],
                        'label': line_label,
                        'line_index': li+1,
                        'lan': sample['line_lan'][li]
                    })

            """保证藏文印刷文本字形的正确"""
            dataset.params['config']['synthetic_data']['config']['lan'] = dataset.params['lan']
            min_height = 999999
            max_height = -1
            min_width = 999999
            max_width = -1

            for i, sample in enumerate(samples):
                # ext = sample['path'].split(".")[-1]
                id, ext = os.path.splitext(os.path.basename(sample['path']))
                ext = ext[1:]
                if dataset.params['lan'].lower() == 'bo':
                    """保持我的命名方式"""
                    img_name = "{}_{}.{}".format(id, sample['line_index'], ext)
                else:
                    img_name = "{}_{}.{}".format(set_name, i, ext)
                img_path = os.path.join(set_path, img_name)
                txt_path = os.path.join(set_path, os.path.splitext(img_name)[0] + '.txt')

                # writ_d = 'ttb' if dataset.params['lan'] == 'bo' and sample['lan'] == 'cjk' else 'ltr' # writing direction
                writ_d = 'ltr'
                syn_sample = dataset.generate_typed_text_line_image(sample["label"], sample['lan'], writ_d)

                if writ_d == 'ttb': # rotate 90deg counterclock wise
                    img = syn_sample['img'].transpose(1, 0, 2)
                    img = np.flip(img, axis=0)
                    syn_sample['img'] = img

                min_height = min(syn_sample['img'].shape[0], min_height)
                min_width = min(syn_sample['img'].shape[1], min_width)
                max_height = max(syn_sample['img'].shape[0], max_height)
                max_width = max(syn_sample['img'].shape[1], max_width)

                Image.fromarray(syn_sample['img']).save(img_path)
                # with open(txt_path, 'w', encoding='utf-8') as f:
                #     f.write(''.join(label))
                gt[set_name][img_name] = {
                    "text": syn_sample['text'],
                }
                if 'char_text' in syn_sample:
                    gt[set_name][img_name]['char_text'] = syn_sample['char_text']
                if "line_label" in sample:
                    gt[set_name][img_name]["lines"] = sample["line_label"]

        # if '\n' in charset:
        #     charset.remove('\n')

        with open(os.path.join(path, "labels.pkl"), "wb") as f:
            pickle.dump({
                "ground_truth": gt,
                "charset": sorted(list(charset)),
            }, f)


        """输出charset到txt文件中"""
        with open(os.path.join(path, "charset.txt"), "w", encoding='utf-8') as f:
            txt = 'index\tcharacter\tunicode\n'
            for i, char in enumerate(charset):
                hex_txt = '+'.join([hex(ord(letter)) for letter in char])
                char = char if char != '\n' else '\\n'
                txt += f'{i}\t{char}\t{hex_txt}\n'
            f.write(txt)

        size_info = f'min_height: {min_height}\n'
        size_info += f'max_height: {max_height}\n'
        size_info += f'min_width: {min_width}\n'
        size_info += f'max_width: {max_width}'
        print(size_info)

        """输出尺寸信息到txt文件中"""
        with open(os.path.join(path, "size_info.txt"), "w", encoding='utf-8') as f:
            f.write(size_info)

    def generate_syn_page_dataset(self, name):
        """
        Generate synthetic document-level dataset from currently loaded dataset
        """
        dataset_name = list(self.params['dataset_params']["datasets"].keys())[0]
        path = os.path.join(os.path.dirname(self.params['dataset_params']["datasets"][dataset_name]), name)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        # charset = set()
        charset = self.dataset.charset
        dataset = None
        gt = {
            "train": dict(),
            "valid": dict(),
            "test": dict()
        }

        min_height = 999999
        max_height = -1
        min_width = 999999
        max_width = -1

        for set_name in ["train", "valid", "test"]:
            set_path = os.path.join(path, set_name)
            os.makedirs(set_path, exist_ok=True)
            if set_name == "train":
                dataset = self.dataset.train_dataset
            elif set_name == "valid":
                dataset = self.dataset.valid_datasets["{}-valid".format(dataset_name)]
            elif set_name == "test":
                self.dataset.generate_test_loader("{}-test".format(dataset_name), [(dataset_name, "test"), ])
                dataset = self.dataset.test_datasets["{}-test".format(dataset_name)]

            batch_size = self.dataset.batch_size['train']

            """保证藏文印刷文本字形的正确"""
            dataset.params['config']['synthetic_data']['config']['lan'] = dataset.params['lan']

            for i, sample in enumerate(dataset.samples):
                trained_samples = randint(0, self.params['training_params']['max_nb_epochs']*len(self.dataset.train_dataset))
                epoch = int(trained_samples / len(self.dataset.train_dataset))
                # trained_samples = self.params['training_params']['max_nb_epochs'] * len(self.dataset.train_dataset)
                dataset.training_info = {
                    "epoch": epoch,
                    "step": trained_samples * batch_size,
                    'trained_samples': trained_samples
                }
                if set_name == 'train' and i == 3:
                    print('debug')
                syn_sample = dataset.generate_synthetic_page_sample()
                img = syn_sample['img']
                label = syn_sample['label']


                min_height = min(img.shape[0], min_height)
                min_width = min(img.shape[1], min_width)
                max_height = max(img.shape[0], max_height)
                max_width = max(img.shape[1], max_width)

                # img_name = os.path.basename(sample['path'])
                # id, _ = os.path.splitext(img_name)
                img_name = syn_sample['name'] + '.png'
                img_path = os.path.join(set_path, img_name)
                Image.fromarray(img).save(img_path)
                txt_path = os.path.join(set_path, syn_sample['name'] + '.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(''.join(label))

                gt[set_name][img_name] = {
                    "text": ''.join(label), # 只保存raw_label即可，因为行预训练的时候会再次处理
                    'lan': sample['lan']
                }
                if type(label) is list:
                    gt[set_name][img_name]['char_text'] = label # 会再次处理，不同担心


        with open(os.path.join(path, "labels.pkl"), "wb") as f:
            pickle.dump({
                "ground_truth": gt,
                "charset": sorted(list(charset)),
            }, f)


        """输出charset到txt文件中"""
        with open(os.path.join(path, "charset.txt"), "w", encoding='utf-8') as f:
            txt = 'index\tcharacter\tunicode\n'
            for i, char in enumerate(charset):
                hex_txt = '+'.join([hex(ord(letter)) for letter in char])
                char = char if char != '\n' else '\\n'
                txt += f'{i}\t{char}\t{hex_txt}\n'
            f.write(txt)

        size_info = f'min_height: {min_height}\n'
        size_info += f'max_height: {max_height}\n'
        size_info += f'min_width: {min_width}\n'
        size_info += f'max_width: {max_width}'
        print(size_info)

        """输出尺寸信息到txt文件中"""
        with open(os.path.join(path, "size_info.txt"), "w", encoding='utf-8') as f:
            f.write(size_info)

    def visualize_coverage(self, img_paths, coverage_layers):
        os.makedirs(os.path.join(self.paths['output_folder'], 'visualize'), exist_ok=True)

        colors = {
            'chars': (0, 0, 255), # 红色
            'sem_start': (0, 0, 255), # red
            'sem_end':   (255, 0, 0), # blue
            'line_break': (0, 255, 0) # green
        }
        for b, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            overlay = np.zeros_like(img)
            for ckey, coverage in coverage_layers.items():
                attn_map = coverage[b].detach().cpu().numpy()
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
                attn_map = cv2.resize(attn_map, (img.shape[1], img.shape[0]))
                # heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
                attn_uint8 = np.uint8(255 * attn_map)
                # 将灰度图映射为指定颜色的伪彩色图
                heatmap = np.zeros_like(img)
                for i in range(3):  # BGR 通道
                    heatmap[:, :, i] = attn_uint8 * (colors[ckey][i] / 255.0)
                overlay = cv2.addWeighted(overlay, 1., heatmap, 1., 0)
            img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
            save_path = os.path.join(self.paths['output_folder'], 'visualize', os.path.basename(img_path))
            if os.path.exists(save_path):
                os.remove(save_path)
            cv2.imwrite(save_path, img)
            # Image.fromarray(img).save(save_path)

        return