# PGD: Semantic Prior-Guided Dual Decoding for Long-Tailed Human-Object Interaction Detection

This repository contains the official PyTorch implementation for the paper
> Jeongae Lee and Jongho Nang; Semantic Prior-Guided Dual Decoding for Long-Tailed Human-Object Interaction Detection_;

## Installation

### Clone this Repo
The code was tested with CUDA 12.0 python 3.9 Pytorch 2.0.1
    ```
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    pip install matplotlib==3.6.3 scipy==1.10.0 tqdm==4.64.1
    pip install numpy==1.24.1 timm==0.6.12
    pip install wandb==0.13.9 seaborn==0.13.0
    # Clone the repo and submodules
    git clone https://github.com/Jeong-ae/pgd.git
    cd pgd
    git submodule init
    git submodule update
    pip install -e pocket
    # Build CUDA operator for MultiScaleDeformableAttention
    cd h_detr/models/ops
    python setup.py build install
    ```
### Conda
To set up an Anaconda environment, run the following commands from:
```
    git clone https://github.com/Jeong-ae/pgd.git
    cd pgd
    conda env create --file final_environment.yaml
    conda pgd
    git submodule init
    git submodule update
    pip install -e pocket
    # Build CUDA operator for MultiScaleDeformableAttention
    cd h_detr/models/ops
    python setup.py build install
```

## Datasets

1. Prepare the [HICO-DET dataset](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk).
    1. If you have not downloaded the dataset before, run the following script.
        ```bash
        cd /path/to/pgd/hicodet
        bash download.sh
        ```
    2. If you have previously downloaded the dataset, simply create a soft link.
        ```bash
        cd /path/to/pgd/hicodet
        ln -s /path/to/hicodet_20160224_det ./hico_20160224_det
        ```
2. Prepare the V-COCO dataset (contained in [MS COCO](https://cocodataset.org/#download)).
    1. If you have not downloaded the dataset before, run the following script
        ```bash
        cd /path/to/pgd/vcoco
        bash download.sh
        ```
    2. If you have previously downloaded the dataset, simply create a soft link
        ```bash
        cd /path/to/pgd/vcoco
        ln -s /path/to/coco ./mscoco2014
        ```

## Train
Refer to the [documentation](docs.md) for model checkpoints and training/testing commands.
To train a model, run:
```
CUDA_VISIBLE_DEVICES=0,1 DETR=base python main.py --pretrained checkpoints/detr-r50-hicodet.pth \
                         --output-dir outputs/pgd-detr-r50-hicodet
```


## Evaluation
Evaluate using the checkpoints:
```
CUDA_VISIBLE_DEVICES=0,1 DETR=base python main.py --resume outputs/pgd-detr-r50-hicodet/best.pth \
                         --world-size 2 --batchsize 16
```
## Acknowledgement

This repository is built upon the official implementation of[PVIC](https://github.com/fredzzhang/pvic?tab=readme-ov-file).
We thank the authors for their excellent codebase and open-source contribution.
