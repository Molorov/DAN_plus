import os
import sys
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))
sys.path.append(os.path.dirname(os.path.dirname(DOSSIER_PARENT)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DOSSIER_PARENT))))
from torch.optim import Adam
from basic.transforms import aug_config
from OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from OCR.document_OCR.dan.trainer_dan import Manager
from OCR.document_OCR.dan.models_dan import GlobalHTADecoder
from basic.models import FCN_Encoder
from basic.scheduler import exponential_dropout_scheduler, linear_scheduler
import torch
import numpy as np
import random
import torch.multiprocessing as mp
import argparse
import yaml

seed = 0

def train_and_test(rank, params, test):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    params["training_params"]["ddp_rank"] = rank
    model = Manager(params)


    # name = os.path.basename(params['dataset_params']['datasets'][params['dataset_params']['name']]) + '_syn'

    model.generate_syn_page_dataset(params['dataset_params']['name'] + '_syn_page')


parser = argparse.ArgumentParser(description='Generic runner for myVAEs')

parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help = 'path to the config file')

parser.add_argument('--data_root', '-d',
                    dest="data_root",
                    default='c:/myDatasets/DAN')

parser.add_argument('--test', '-t',
                    dest="test",
                    action='store_true',
                    default=False)


args = parser.parse_args()
with open(args.filename, 'r', encoding='utf-8') as file:
    try:
        params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


if __name__ == "__main__":

    dataset_name = params['dataset_params']['name']
    # if dataset_name.lower() not in ['read_2016', 'rimes']:
    #     params['dataset_params']['lan'] = 'bo'
    # else:
    #     params['dataset_params']['lan'] = 'latin'

    dataset_level = params['dataset_params']['level']
    dataset_variant = params['dataset_params'].get('variant', '')

    params['model_params']['models']['encoder'] = globals()[params['model_params']['models']['encoder']]
    params['model_params']['models']['decoder'] = globals()[params['model_params']['models']['decoder']]
    params['model_params']['dropout_scheduler']['function'] = globals()[
        params['model_params']['dropout_scheduler']['function']]

    params['dataset_params']['dataset_manager'] = globals()[params['dataset_params']['dataset_manager']]
    params['dataset_params']['dataset_class'] = globals()[params['dataset_params']['dataset_class']]
    params['dataset_params']['datasets'] = {
        dataset_name: os.path.join(args.data_root, "formatted/{}_{}{}".format(dataset_name, dataset_level, dataset_variant))
    }

    params['dataset_params']['train'] = {
        "name": "{}-train".format(dataset_name),
        "datasets": [(dataset_name, "train"), ],
    }
    params['dataset_params']['valid'] = {
        "{}-valid".format(dataset_name): [(dataset_name, "valid"), ],
    }
    if 'augmentation' not in params['dataset_params']['config']:
        """载入默认的设置"""
        params['dataset_params']['config']['augmentation'] = aug_config(0.9, 0.1)

    if params['dataset_params']['config'].get('synthetic_data', None):
        params['dataset_params']['config']['synthetic_data']['proba_scheduler_function'] = globals()[
            params['dataset_params']['config']['synthetic_data']['proba_scheduler_function']]
        params['dataset_params']['config']['synthetic_data']['dataset_level'] = dataset_level
        params['dataset_params']['config']['synthetic_data']['config']['background_color_default'] = tuple(
            params['dataset_params']['config']['synthetic_data']['config']['background_color_default'])
        params['dataset_params']['config']['synthetic_data']['config']['text_color_default'] = tuple(
            params['dataset_params']['config']['synthetic_data']['config']['text_color_default'])

    # folder name for checkpoint and results
    params['training_params']['output_folder'] = os.path.splitext(os.path.basename(args.filename))[0]
    # maximum time before to stop (in seconds)
    params['training_params']['max_training_time'] = 3600 * 24 * 1.9
    params['training_params']['nb_gpu'] = torch.cuda.device_count()
    for key in params['training_params']['optimizers'].keys():
        params['training_params']['optimizers'][key]['class'] = globals()[
            params['training_params']['optimizers'][key]['class']]
    # Which dataset to focus on to select best weights
    params['training_params']['set_name_focus_metric'] = "{}-valid".format(dataset_name)

    params['training_params']['to_log'] = False  # do not create folder

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
    else:
        train_and_test(0, params, args.test)


