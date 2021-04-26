python ../evaluation.py \
    --test_dataset_csv ../3dvggface2_1_align2_07/test.csv \
    --pretrained_model_path ../3dface_models/logs_resnext50_12-23.22-54/3dface-model.pkl \
    --num_of_workers 8 \
    --num_of_classes 1200 \
    --is_resnext50 True
