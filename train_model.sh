#!/bin/bash -x

python train_3dface.py --train_dataset_csv '/data/3d-reconstruction/notebooks/Face3D-Pytorch/3dvggface2_1_align2_07/train.csv' \
    --eval_dataset_csv '/data/3d-reconstruction/notebooks/Face3D-Pytorch/3dvggface2_1_align2_07/eval.csv' \
    --pretrained_on_imagenet \
    --input_channels 4 \
    --num_of_classes 1200 \
    --is_vgg16 True \
    --num_of_epochs 50 \
    --num_of_workers 8 \
    --logs_base_dir './3dface_models/logs_vgg16_'

