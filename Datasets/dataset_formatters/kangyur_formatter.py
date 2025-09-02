import glob
import os
import random
import json
import cv2
import matplotlib.pyplot as plt
import shutil
import pickle
from PIL import Image
from Datasets.dataset_formatters.tibetan_utils import *
from Datasets.dataset_formatters.kangyur_utils import *


# Layout begin-token to end-token
SEM_MATCHING_TOKENS = {
            "ⓑ": "Ⓑ",  # paragraph (body)
            "ⓛ": "Ⓛ",  # section: left title
            "ⓡ": "Ⓡ",  # section: right title
            "ⓜ": "Ⓜ",  # section: main text
            "ⓟ": "Ⓟ",  # page
        }

def is_kangyur(name):
    if 'BJK' in name or 'LJK' in name or 'DGK' in name:
        return True
    return False

def update_charset(charset, char_list):
    for char in char_list:
        if char in charset.keys():
            charset[char] += 1
        else:
            charset[char] = 1
    return charset

def initialize_charset():
    sem_tokens = list(SEM_MATCHING_TOKENS.keys()) + list(SEM_MATCHING_TOKENS.values())
    return {st: 0 for st in sem_tokens}


def save_page_data(gts_src, data_name, sem_token):
    gts_tgt = dict()
    charset = initialize_charset()
    charset_split = dict()
    compset = initialize_charset()
    compset_split = dict()
    data_name += '_page'
    if sem_token:
        data_name += '_sem'
    target_folder = os.path.join('../formatted', data_name)
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)
    for set_name, data_src in gts_src.items():
        gts_tgt[set_name] = dict()
        charset_split[set_name] = initialize_charset()
        compset_split[set_name] = initialize_charset()
        target_img_folder = os.path.join(target_folder, set_name)
        os.makedirs(target_img_folder, exist_ok=True)
        for name, page_data in data_src.items():
            target_img_path = os.path.join(target_img_folder, name)
            page_img = page_data['page_img']
            cv2.imwrite(target_img_path, page_img)

            page_tgt = {
                'text': '',
                'char_text': [],
                'sections': [],
                'nb_cols': 0,
            }
            page_char_text = []

            for section in ['main_text', 'ltitle', 'rtitle']: # iterate over sections following specified reading order
                if section not in page_data.keys():
                    continue
                left = 999999
                right = -999999
                top = 999999
                bottom = -999999
                page_tgt['nb_cols'] += 1
                page_tgt['sections'].append({
                    'text': '',
                    'char_text': [],
                    'mode': 'body' if section == 'main_text' else section,
                    'paragraphs': []
                })
                sect_char_text = []
                for i in range(len(page_data[section])): # iterate over paragraphs
                    para_char_text = []
                    lines = []
                    for j, line_data in enumerate(page_data[section][i]): # iterate over lines
                        lines.append({
                            'text': line_data['text'],
                            'char_text': line_data['char_text'],
                            'left': line_data['left'],
                            'right': line_data['right'],
                            'top': line_data['top'],
                            'bottom': line_data['bottom'],
                            'contour': line_data['contour'],
                            # 'words': line_data['words']
                        })
                        if j > 0:
                            para_char_text += ['\n']
                        para_char_text += line_data['char_text']


                    px, py, pw, ph = get_bbox_from_contours([l['contour'] for l in lines])
                    left = min(left, px)
                    right = max(right, px + pw)
                    top = min(top, py)
                    bottom = max(bottom, py + ph)
                    page_tgt['sections'][-1]['paragraphs'].append({
                        'text': ''.join(para_char_text),
                        'char_text': para_char_text,
                        'lines': lines,
                        'left': px,
                        'right': px+pw,
                        'top': py,
                        'bottom': py+ph
                    })
                    if sem_token:
                        sect_char_text += ['ⓑ'] + para_char_text + ['Ⓑ']
                    else:
                        sect_char_text += para_char_text
                    # if chr(0x25a1) in para_char_text:
                    #     print('debug')
                    # charset = update_charset(charset, para_char_text)
                    # charset_split[set_name] = update_charset(charset_split[set_name], para_char_text)
                    # compset = update_charset(compset, ''.join(para_char_text))
                    # compset_split[set_name] = update_charset(compset_split[set_name], ''.join(para_char_text))

                page_tgt['sections'][-1]['text'] = ''.join(sect_char_text)
                page_tgt['sections'][-1]['char_text'] = sect_char_text
                page_tgt['sections'][-1]['left'] = left
                page_tgt['sections'][-1]['right'] = right
                page_tgt['sections'][-1]['top'] = top
                page_tgt['sections'][-1]['bottom'] = bottom

                if sem_token and section == 'main_text':
                    page_char_text += ['ⓜ'] + sect_char_text + ['Ⓜ']
                elif sem_token and section == 'ltitle':
                    page_char_text += ['ⓛ'] + sect_char_text + ['Ⓛ']
                elif sem_token and section == 'rtitle':
                    page_char_text += ['ⓡ'] + sect_char_text + ['Ⓡ']

            page_char_text = ['ⓟ'] + page_char_text + [SEM_MATCHING_TOKENS['ⓟ']]

            page_tgt['text'] = ''.join(page_char_text)
            page_tgt['char_text'] = page_char_text

            # if '□' in page_char_text:
            #     print('debug')
            charset = update_charset(charset, page_char_text)
            charset_split[set_name] = update_charset(charset_split[set_name], page_char_text)
            compset = update_charset(compset, ''.join(page_char_text))
            compset_split[set_name] = update_charset(compset_split[set_name], ''.join(page_char_text))

            # doc_char_text = ['ⓟ'] + page_char_text +  ['Ⓟ'] if sem_token else page_char_text

            gts_tgt[set_name][name] = page_tgt

        charset_split[set_name] = dict(sorted(charset_split[set_name].items()))
        compset_split[set_name] = dict(sorted(compset_split[set_name].items()))


    charset = dict(sorted(charset.items()))
    compset = dict(sorted(compset.items()))

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gts_tgt,
            "charset": charset,
            "charset_split": charset_split,
            'compset': compset,
            'compset_split': compset_split
        }, f)

    """输出charset到txt文件中"""
    with open(os.path.join(target_folder, "charset.txt"), "w", encoding='utf-8') as f:
        txt = 'index\tcharacter\tunicode\tcount\n'
        for i, (char, cnt) in enumerate(charset.items()):
            hex_txt = '+'.join([hex(ord(letter)) for letter in char])
            char = char if char != '\n' else '\\n'
            txt += f'{i}\t{char}\t{hex_txt}\t{cnt}\n'
        f.write(txt)

    """输出compset到txt文件中"""
    with open(os.path.join(target_folder, "compset.txt"), "w", encoding='utf-8') as f:
        txt = 'index\tcomponent\tunicode\tcount\n'
        for i, (letter, cnt) in enumerate(compset.items()):
            hex_txt = hex(ord(letter))
            letter = letter if letter != '\n' else '\\n'
            txt += f'{i}\t{letter}\t{hex_txt}\t{cnt}\n'
        f.write(txt)


def save_lines_data(gts_src, data_name):
    gts_tgt = dict()
    charset = dict()
    charset_split = dict()
    compset = dict()
    compset_split = dict()
    data_name += '_lines'
    target_folder = os.path.join('../formatted', data_name)
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)
    for set_name, data_src in gts_src.items():
        gts_tgt[set_name] = dict()
        charset_split[set_name] = dict()
        compset_split[set_name] = dict()
        target_img_folder = os.path.join(target_folder, set_name)
        os.makedirs(target_img_folder, exist_ok=True)

        for pn, page_data in data_src.items():
            page_id, ext = os.path.splitext(pn)
            page_img = page_data['page_img']
            for section in ['ltitle', 'main_text', 'rtitle']:
                if section not in page_data.keys():
                    continue
                for i in range(len(page_data[section])):  # iterate over paragraphs
                    for j, line_data in enumerate(page_data[section][i]): # iterate over lines
                        if section == 'main_text':
                            ln = page_id + '_' + str(j+1) + ext
                        else:
                            ln = page_id + '_' + section + '_' + str(i+1) + ext
                        target_img_path = os.path.join(target_img_folder, ln)
                        contour = line_data['contour']
                        line_img = cut_image_from_contour(page_img, contour)
                        if section in ['ltitle', 'rtitle']:
                            pil_img = Image.fromarray(line_img)
                            pil_img = pil_img.transpose(Image.ROTATE_90)
                            line_img = np.array(pil_img)

                        cv2.imwrite(target_img_path, line_img)
                        gts_tgt[set_name][ln] = {
                            'text': line_data['text'],
                            'char_text': line_data['char_text']
                        }
                        charset = update_charset(charset, line_data['char_text'])
                        charset_split[set_name] = update_charset(charset_split[set_name], line_data['char_text'])
                        compset = update_charset(compset, line_data['text'])
                        compset_split[set_name] = update_charset(compset_split[set_name], line_data['text'])

        charset_split[set_name] = dict(sorted(charset_split[set_name].items()))
        compset_split[set_name] = dict(sorted(compset_split[set_name].items()))
    charset = dict(sorted(charset.items()))
    compset = dict(sorted(compset.items()))

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gts_tgt,
            "charset": charset,
            "charset_split": charset_split,
            "compset": compset,
            "compset_split": compset_split
        }, f)

    """输出charset到txt文件中"""
    with open(os.path.join(target_folder, "charset.txt"), "w", encoding='utf-8') as f:
        txt = 'index\tcharacter\tunicode\tcount\n'
        for i, (char, cnt) in enumerate(charset.items()):
            hex_txt = '+'.join([hex(ord(letter)) for letter in char])
            char = char if char != '\n' else '\\n'
            txt += f'{i}\t{char}\t{hex_txt}\t{cnt}\n'
        f.write(txt)
    """输出compset到txt文件中"""
    with open(os.path.join(target_folder, "compset.txt"), "w", encoding='utf-8') as f:
        txt = 'index\tcomponent\tunicode\tcount\n'
        for i, (letter, cnt) in enumerate(compset.items()):
            hex_txt = hex(ord(letter))
            letter = letter if letter != '\n' else '\\n'
            txt += f'{i}\t{letter}\t{hex_txt}\t{cnt}\n'
        f.write(txt)


def search_in_shapes(shapes, pattern):
    ret = []
    for shape in shapes:
        if shape['label'].startswith(pattern):
            ret.append(shape)
    return ret

# from shapely.ops import unary_union
# from shapely.geometry import Polygon
# import geopandas as gpd
from docx import Document
from docx.shared import Inches, Mm, Pt


def get_image_list(folder):
    exts = ['jpg', 'jpeg', 'png']
    image_list = []
    for ext in exts:
        image_list += glob.glob(os.path.join(folder, '*.' + ext))
    # image_list.sort()
    return image_list


def format_data_from_folder(raw_data_dir, crop=False, scale=None):
    """有完整标注的页面数量"""
    # num_complete = 0

    splits = os.listdir(raw_data_dir)
    all_gt_data = {split: {} for split in splits}
    for split in splits:
        image_files = get_image_list(os.path.join(raw_data_dir, split))
        ids = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
        for image_file in image_files:
            id, ext = os.path.splitext(os.path.basename(image_file))
            page_img = cv2.imread(image_file)
            """Keep only documents that have segmentation and text annotations"""
            txt_file = os.path.join(raw_data_dir, split, id + '.txt')
            doc_file = os.path.join(raw_data_dir, split, id + '.docx')
            json_file = os.path.join(raw_data_dir, split, id + '.json')
            if not os.path.exists(json_file):
                continue
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                shapes = json_data['shapes']

            raw_text = []
            if os.path.exists(doc_file):
                """Use revised annotation in docx file"""
                document = Document(doc_file)
                for paragraph in document.paragraphs:
                    line_text = paragraph.text
                    if len(line_text):
                        line_text = line_text.replace('N/A', '')
                        line_text = line_text.replace('N\\A', '')
                        line_text = line_text.replace('?', '')
                        raw_text.append(line_text)
            elif os.path.exists(txt_file):
                """Use unrevised annotation in txt file"""
                with open(txt_file, 'r', encoding='utf-8') as f:
                    raw_text = f.readlines()
                    raw_text = [lt.replace('\n', '') for lt in raw_text]
            else:
                """doesn't have text annotation, discard"""
                continue


            """get line contours for main text area"""
            if search_in_shapes(shapes, 'line_sep-'):
                """是line_sep类型的行切分数据"""
                pts_list = [i['points'] for i in search_in_shapes(shapes, 'line_sep-')]
                contours = get_contours_from_line_seps(pts_list)
            elif search_in_shapes(shapes, 'line-'):
                """是polygon类型的行切分数据"""
                contours = [i['points'] for i in search_in_shapes(shapes, 'line-')]
            contour_dict = {f'line-{i+1}': contour for i, contour in enumerate(contours)}


            """get line contours for left title"""
            pts_list = {i['label']:i['points'] for i in search_in_shapes(shapes, 'ltitle')}
            for label, contour in pts_list.items():
                contour_dict[label] = contour

            pts_list = {i['label']:i['points'] for i in search_in_shapes(shapes, 'rtitle')}
            num_rtitle = len(pts_list)
            """get line contours for right title"""
            for label, contour in pts_list.items():
                contour_dict[label] = contour

            # if id in ['58-1-15a', '58-1-105b']:
            #     print('debug')

            assert len(raw_text) == len(contour_dict)
            gt_data = {
                'page_img': page_img,
                # 'main_text': {'lines': []},
                # 'ltitle': {'lines': []},
                # 'rtitle': {'lines': []},
            }

            if scale:
                img = gt_data['page_img']
                h, w, c = img.shape
                img = cv2.resize(img, (int(np.ceil(w * scale)), int(np.ceil(h * scale))))
                gt_data['page_img'] = img
                for ckey, contour in contour_dict.items():
                    contour_dict[ckey] = (np.array(contour) * scale).tolist()

            if crop:
                left = 99999
                right = -1
                top = 99999
                bottom = -1
                for contour in contour_dict.values():
                    x, y, w, h = get_bbox_from_contour(contour)
                    left = min(x, left)
                    top = min(y, top)
                    right = max(x + w, right)
                    bottom = max(y + h, bottom)
                gt_data['page_img'] = gt_data['page_img'][int(top):int(bottom), int(left):int(right), :]
                for ckey, contour in contour_dict.items():
                    contour_dict[ckey] = (np.array(contour) - np.array([left, top])).tolist()

            for i, ckey in enumerate(contour_dict.keys()):
                line_text = raw_text[i].strip()
                # if 'rtitle' in ckey:
                #     line_text = zhconv.convert(line_text, 'zh-hant') # To Traditional Chinese
                line_char_text = string2char_list(line_text)
                line_char_text = [normalize_char(c) for c in line_char_text]
                # if chr(0xf19) in line_char_text:
                #     print('debug')
                line_contour = contour_dict[ckey]
                x, y, w, h = get_bbox_from_contour(line_contour)
                # line_img = cut_image_from_contour(page_img, line_contour)

                if 'line' in ckey:
                    section = 'main_text'
                elif 'ltitle' in ckey:
                    section = 'ltitle'
                elif 'rtitle' in ckey:
                    section = 'rtitle'
                else:
                    raise NotImplementedError

                if section not in gt_data.keys():
                    gt_data[section] = [[]]
                # elif section in ['ltitle', 'rtitle']:
                #     gt_data[section].append([])
                """lines for each section"""
                gt_data[section][-1].append({
                    'text': ''.join(line_char_text), # line_text,
                    'char_text': line_char_text,
                    'contour': line_contour,
                    'left': x,
                    'right': x + w,
                    'top': y,
                    'bottom': y + h,
                })

            if num_rtitle > 1:
                # sort rtitle according to the reading order of Chinese
                read_order = np.argsort([ck for ck in contour_dict.keys() if ck.startswith('rtitle')])
                sorted_rtitle = []
                for ri in read_order:
                    sorted_rtitle.append(gt_data['rtitle'][-1][ri])
                gt_data['rtitle'][-1] = sorted_rtitle
            all_gt_data[split][id + ext] = gt_data

    return all_gt_data



def format_BJK_185(sem_token, crop):
    raw_data_dir = 'c:/myDatasets/BJK/BJK-RS1-185-2009'
    all_gt_data = format_data_from_folder(raw_data_dir, scale=0.5)
    data_name = os.path.basename(raw_data_dir)
    save_page_data(all_gt_data, data_name, sem_token)
    # save_lines_data(all_gt_data, data_name)

def format_LJK_200(sem_token, crop):
    raw_data_dir = 'c:/myDatasets/LJK/LJK-RS1-200-2010'
    all_gt_data = format_data_from_folder(raw_data_dir, scale=2.)
    data_name = os.path.basename(raw_data_dir)
    save_page_data(all_gt_data, data_name, sem_token)
    # save_lines_data(all_gt_data, data_name)

def format_DGK_199(sem_token, crop):
    raw_data_dir = 'c:/myDatasets/DGK/DGK-RS1-199-2011'
    all_gt_data = format_data_from_folder(raw_data_dir, crop=crop, scale=1.5)
    data_name = os.path.basename(raw_data_dir)
    if crop:
        data_name += '_crop'
    save_page_data(all_gt_data, data_name, sem_token)
    save_lines_data(all_gt_data, data_name)


if __name__ == '__main__':
    format_BJK_185(sem_token=True, crop=False)
    # format_LJK_200(sem_token=True, crop=False)
    # format_DGK_199(sem_token=True, crop=True)



