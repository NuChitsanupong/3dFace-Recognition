#!/bin/bash -x

python train_3dface.py --train_dataset_csv '/data/3d-reconstruction/notebooks/Face3D-Pytorch/3dface_th_align/train.csv' \
    --eval_dataset_csv '/data/3d-reconstruction/notebooks/Face3D-Pytorch/3dface_th_align/eval.csv' \
    --pretrained_model_path '/data/3d-reconstruction/notebooks/Face3D-Pytorch/3dface_models/logs_vgg16_01-05.18-13/3dface-model.pkl' \
    --pretrained_on_imagenet \
    --input_channels 4 \
    --num_of_classes 83 \
    --num_of_pretrained_classes 1200 \
    --is_vgg16 True \
    --is_predicted True \
    --num_of_epochs 50 \
    --num_of_workers 8 \
    --logs_base_dir './3dface_models/logs_vgg16_with_th_'
#    --logs_base_dir './logs_mobileNet_v2_with_th_'
