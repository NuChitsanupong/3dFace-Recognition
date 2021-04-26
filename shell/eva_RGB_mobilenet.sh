python ../evaluation.py \
    --test_dataset_csv ../3dface_th_align/test.csv \
    --pretrained_model_path ../3dface_models/logs_RGB_mobileNet_v2_03-02.23-21/3dface-model.pkl \
    --input_channels 3 \
    --num_of_workers 8 \
    --num_of_classes 83 \
    --is_mobilenet True
