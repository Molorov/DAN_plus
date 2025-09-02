#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
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

import os
import sys
from os.path import dirname
DOSSIER_COURRANT = dirname(os.path.abspath(__file__))
ROOT_FOLDER = dirname(dirname(dirname(DOSSIER_COURRANT)))
sys.path.append(ROOT_FOLDER)
from OCR.line_OCR.ctc.trainer_line_ctc import TrainerLineCTC
from OCR.line_OCR.ctc.models_line_ctc import Decoder
from basic.models import FCN_Encoder, FCN_Encoder16x16
from torch.optim import Adam
from basic.transforms import line_aug_config
from basic.scheduler import exponential_dropout_scheduler, exponential_scheduler
from OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
import torch.multiprocessing as mp
import torch
import numpy as np
import random
import argparse
import yaml


def train_and_test(rank, params, test):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    params["training_params"]["ddp_rank"] = rank
    model = TrainerLineCTC(params)
    model.load_model()

    # Model trains until max_time_training or max_nb_epochs is reached
    if not test:
        model.train()

    # load weights giving best CER on valid set
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()


    # compute metrics on train, valid and test sets (in eval conditions)
    metrics = ["cer", "ser", "time", "worst_cer", "pred"]
    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ["test", "valid", "train", ]:
            model.predict("{}-{}".format(dataset_name, set_name), [(dataset_name, set_name), ], metrics, output=True)


parser = argparse.ArgumentParser(description='Generic runner for myVAEs')

parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help = 'path to the config file')

parser.add_argument('--data_root', '-d',
                    dest="data_root",
                    default='../../../Datasets')

parser.add_argument('--test', '-t',
                    dest="test",
                    action='store_true',
                    default=False)


args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

data_root = args.data_root

if __name__ == "__main__":
    dataset_name = params['dataset_params']['name']

    dataset_level = params['dataset_params']['level']
    dataset_variant = params['dataset_params'].get('variant', '')

    params['model_params']['models']['encoder'] = globals()[params['model_params']['models']['encoder']]
    params['model_params']['models']['decoder'] = globals()[params['model_params']['models']['decoder']]
    if 'dropout_scheduler' in params['model_params']:
        params['model_params']['dropout_scheduler']['function'] = globals()[
            params['model_params']['dropout_scheduler']['function']]

    params['dataset_params']['dataset_manager'] = globals()[params['dataset_params']['dataset_manager']]
    params['dataset_params']['dataset_class'] = globals()[params['dataset_params']['dataset_class']]
    params['dataset_params']['datasets'] = {
        dataset_name: os.path.join(args.data_root,
                                   "formatted/{}_{}{}".format(dataset_name, dataset_level, dataset_variant))
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
        params['dataset_params']['config']['augmentation'] = line_aug_config(0.9, 0.1)

    if params['dataset_params']['config'].get('synthetic_data', None):
        params['dataset_params']['config']['synthetic_data']['proba_scheduler_function'] = globals()[
            params['dataset_params']['config']['synthetic_data']['proba_scheduler_function']]
        params['dataset_params']['config']['synthetic_data']['dataset_level'] = dataset_level
        params['dataset_params']['config']['synthetic_data']['config']['background_color_default'] = tuple(
            params['dataset_params']['config']['synthetic_data']['config']['background_color_default'])
        params['dataset_params']['config']['synthetic_data']['config']['text_color_default'] = tuple(
            params['dataset_params']['config']['synthetic_data']['config']['text_color_default'])
        params['dataset_params']["config"]["synthetic_data"]["config"]['lan'] = params['dataset_params']['lan']

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

    params['training_params']['to_log'] = True

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
    else:
        train_and_test(0, params, args.test)

