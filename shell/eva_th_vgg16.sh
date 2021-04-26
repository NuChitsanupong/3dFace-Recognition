python ../evaluation.py \
    --test_dataset_csv ../3dface_th_align/test.csv \
    --pretrained_model_path ../3dface_models/logs_vgg16_with_th_01-11.10-00/3dface-model.pkl \
    --num_of_workers 8 \
    --num_of_classes 83 \
    --is_vgg16 True
