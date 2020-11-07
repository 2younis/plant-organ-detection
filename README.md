Detection of Plant Organs from Digitized Herbarium Scans
============================

This repository provides the official PyTorch implementation of the following publication.

> Detection and Annotation of Plant Organs from Digitized Herbarium Scans using Deep Learning
> https://arxiv.org/abs/2007.13106

The supplementary dataset for the above publication is also publically available on PANGEA.

> Plant organ detections and annotations on digitized herbarium scans
> https://doi.org/10.1594/PANGAEA.920895

## Dependencies
* [Python 3.7+](https://www.python.org)
* [PyTorch 1.6.0](http://pytorch.org)
* [Torchvision 0.7.0](http://pytorch.org)
* [Detectron2 0.2+](https://github.com/facebookresearch/detectron2)

## Setup

Before training the organ detection model, the dataset needs to be prepared. The first step is downloading the Herbarium Senckenbergianum (FR) annotated dataset from PANGEA. The following commands will download and extract the dataset.

```sh
$ cd dataset/
dataset$ wget https://download.pangaea.de/dataset/920895/files/Dataset.rar
dataset$ unar Dataset.rar
```

The next step is downloading the MNHN Paris Herbarium dataset. The following script will automatically download herbarium scans used for training and move both datasets to their corresponding directories.

```sh
$ python prepare_dataset.py
```

For reference the directory layout of the files and modules is drawn below.

    .
    ├── demo.py
    ├── prepare_dataset.py
    ├── train_net.py
    ├── README.md
    ├── configs/
    ├── dataset/
    │   └── HerbarFR/
    └── utils/


## Usage

The first step for detecting plant organs, according to the paper, is training a Faster-RCNN model on the training set of MNHN Paris Herbarium dataset and evaluating it on the test set.

```sh
$ python train_net.py DATASETS.TRAIN hrb_paris_train DATASETS.TEST hrb_paris_test SOLVER.MAX_ITER 9000
```

After the model has been trained on the training set, inference can be performed on the test set to visualize the bounding boxes for plant organs on images.

```sh
$ python demo.py --input dataset/HerbarParis/test/*.jpg --opts MODEL.WEIGHTS output/model_final.pth
```

Then the organ detection model can be trained on all of the MNHN Paris Herbarium scans and evaluated on the Herbarium Senckenbergianum (FR) dataset by running the following script.

```sh
$ python train_net.py DATASETS.TRAIN hrb_paris_all DATASETS.TEST hrb_fr
```

Likewise inference can also be performed on the Herbarium Senckenbergianum (FR) dataset to visualize the detected organs on the scans.

```sh
$ python demo.py --input dataset/HerbarFR/scans/*.jpg --opts MODEL.WEIGHTS output/model_final.pth`
```
A [trained model](https://github.com/2younis/plant-organ-detection/releases/download/v1.0/model_final.pth) on MNHN Paris Herbarium dataset can be downloaded directly for inference, without the need to train it again.

```sh
$ wget https://github.com/2younis/plant-organ-detection/releases/download/v1.0/model_final.pth -P output/
```
