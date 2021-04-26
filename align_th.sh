#!/bin/bash -x
unset PYTHONPATH
export PYTHONPATH=${PWD}/facenet/src
#export PYTHONPATH=/media/hdd10T/3d-reconstruction/notebooks/Face3D-Pytorch/facenet/src

python preprocess/align/align_dataset_mtcnn.py \
    --input_dir ${PWD}/3dface_th \
    --output_dir ${PWD}/3dface_th_align \
    --image_size 224 \
    --margin 44 \
    --random_order \
    --thread_num 3 \
    --gpu_memory_fraction 0.88
