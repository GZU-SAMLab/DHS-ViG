# DHS-ViG: Dynamic Hierarchical Selection for Comprehensive and Robust Feature Perception in Vision GNNs

The implementation of DHS-ViG, we will release all code after the paper is accepted (object detection & semantic segmentation).

## Requirements

python 3.7, 
torch 1.7.0, 
cuda 10.2, 
apex 0.04, 
timm 0.3.2, 
torchprofile 0.0.4, 
einops, 
matplotlib

## Training

    python -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/imagenet/ --model dhs_vig_ti_224_gelu --sched cosine --epochs 300 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 2e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /path/to/save/models/


## Evaluation

    python train.py /path/to/imagenet/ --model dhs_vig_ti_224_gelu -b 128 --pretrain_path /path/to/pretrained/model/ --evaluate

