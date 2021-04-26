python ../evaluation.py \
    --test_dataset_csv ../3dface_th_align/test.csv \
    --pretrained_model_path ../3dface_models/logs_resnext50_with_th_01-05.15-03/3dface-model.pkl \
    --num_of_workers 8 \
    --num_of_classes 83 \
    --is_resnext50 True
