# DAN+: Enhancing Transformer-Based Document Recognizer with Dynamic Attention Sink and Structured Skipping
This repository is a public implementation of the paper: "DAN+: Enhancing Transformer-Based Document Recognizer with Dynamic Attention Sink and Structured Skipping".

This work focus on handwritten document recognition, which unifies handwritten text recognition and document layout analysis within a single architecture.

The proposed model uses a Transformer-based encoder-decoder framework for end-to-end segmentation-free handwritten document recognition. The model is evaluated on five publicly available datasets.

We obtained the following results:

|                         | CER (%) | W/SER (%) | LOER (%) | mAP_cer (%) |
|:-----------------------:|---------|:-------:|:--------:|-------------|
|  READ 2016 (single page)   | 3.12    |  12.58  |   4.27   | 95.30       |
|  READ 2016 (double page)   | 3.33    |  13.59  |   3.91   | 95.42       |
| BJK-185 (page)             | 2.66    |  6.51   |   0.00   | 69.79       |
| LJK-200 (page)             | 2.90    |  5.37   |   0.29   | 69.83       |
| DGK-199 (page)             | 1.18    |  2.75   |   0.00   | 66.92       |


Pretrained model weights are available [here](https://zenodo.org/uploads/17033520).

Table of contents:
1. [Getting Started](#Getting-Started)
2. [Datasets](#Datasets)
3. [Training And Evaluation](#Training-and-evaluation)

## Getting Started
We used Python 3.9.1, Pytorch 1.8.2 and CUDA 10.2 for the scripts.

Clone the repository:

```
git clone https://github.com/Molorov/DAN_plus
```

Install the dependencies:

```
pip install -r requirements.txt
```


## Datasets
This section is dedicated to the datasets used in the paper: download and formatting instructions are provided 
for experiment replication purposes.

READ 2016 dataset corresponds to the one used in the [ICFHR 2016 competition on handwritten text recognition](https://ieeexplore.ieee.org/document/7814136).
It can be found [here](https://zenodo.org/record/1164045#.YiINkBvjKEA)

Formatted version for BJK-185 is available [here](https://pan.baidu.com/s/1g7DeMMJk0lJqstTa2xylbA?pwd=7y8x)

Formatted version for LJK-200 is available [here](https://pan.baidu.com/s/1me3Uzj10n4Kz2ffVNb1tKg?pwd=zcm5)

Formatted version for DGK-199 is available [here](https://pan.baidu.com/s/1tfHSRM8keO3SHruqrf0Vkw?pwd=tqh9)

Raw dataset files must be placed in Datasets/raw/{dataset_name} \
where dataset name is "READ 2016"

## Training And Evaluation
### Step 1: Download the dataset

### Step 2: Format the dataset
```
python3 Datasets/dataset_formatters/read2016_formatter.py
```

### Step 3: Add any font you want as .ttf file in the folder Fonts

### Step 4 : Generate synthetic line dataset for pre-training
```
python3 OCR/line_OCR/main_syn_line.py --config ../../../syn_line_config/READ_2016_synline.yaml
```

### Step 5 : Pre-training on synthetic lines
```
python3 OCR/line_OCR/ctc/main_line_ctc.py --config ../../../line_config/READ_2016_line.yaml
```

Weights and evaluation results are stored in OCR/line_OCR/ctc/outputs

### Step 6 : Training the DAN+
```
python3 OCR/document_OCR/dan_plus/main_danp.py --config ../../../config_danp/READ_2016_double.yaml
```
The following lines in the configuration file must be adapted to the pre-training folder names:
```
transfer_learning:
    # model_name: [state_dict_name, checkpoint_path, learnable, strict]
    encoder: ["encoder", "../../line_OCR/ctc/outputs/READ_2016_line/checkpoints/best_408.pt", True, True]
    decoder: ["decoder", "../../line_OCR/ctc/outputs/READ_2016_line/checkpoints/best_408.pt", True, False]

```

Weights and evaluation results are stored in OCR/document_OCR/dan_plus/outputs

### Step 7 : Evaluation
```
python3 OCR/document_OCR/dan_plus/main_danp.py --config ../../../config_danp/READ_2016_double.yaml --test
```



### Remarks (for pre-training and training)
All hyperparameters are specified and editable in the configuration file.\
Evaluation is performed just after training process ends.\
The outputs files are split into two subfolders: "checkpoints" and "results". \
"checkpoints" contains model weights for the last trained epoch and for the epoch giving the best valid CER. \
"results" contains tensorboard log for loss and metrics as well as text file for used hyperparameters and results of evaluation.


## Citation

```bibtex
@article{LeerMao,
  author = {Mao, Leer and Wang, Weilan and Li, Qiaoqiao},
  title = {DAN+: Enhancing Transformer-Based Document Recognizer with Dynamic Attention Sink and Structured Skipping},
  doi={}
  journal={Knowledge-Based Systems},
  volume={},
  number={},
  pages={},
  year = {2025},
}
```

## License

This whole project is under Cecill-C license.
