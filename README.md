# AWDA

Official PyTorch implementation of **AWDA: Adversarial and Weighted Domain Adaptation for Cross-Dataset Change Detection**.

This public version contains the training code for the proposed AWDA method with a ResNet-50 backbone.


## Installation

```bash
conda create -n awda python=3.10 -y
conda activate awda
pip install -r requirements.txt
bash models/backbones/get_resnet50_pretrained_model.sh
```

The pretrained file is intentionally excluded from Git because it exceeds GitHub's regular file-size limit.

## Data layout

Each dataset directory must contain paired images, binary labels, and split files:

```text
DATASET/
|-- A/
|-- B/
|-- label/
`-- list/
    |-- train.txt
    `-- val.txt
```

Each line in a split file contains three paths relative to the dataset root:

```text
A/0001.png B/0001.png label/0001.png
```

Update the three `data_dir` values in `configs/awda_resnet50.json` before training.

## Training

```bash
python train.py --config configs/awda_resnet50.json --device cuda
```


