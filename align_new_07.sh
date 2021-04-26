#!/bin/bash -x
unset PYTHONPATH
export PYTHONPATH=${PWD}/facenet/src

python preprocess/align/align_dataset_mtcnn.py \
    --input_dir ${PWD}/3dvggface2_1 \
    --output_dir ${PWD}/3dvggface2_1_align2_07 \
    --image_size 182 \
    --margin 44 \
    --random_order \
    --thread_num 3 \
    --gpu_memory_fraction 0.88
