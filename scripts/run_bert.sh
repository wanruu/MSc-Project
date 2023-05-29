python ../src/run_bert.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --train_file ../data/bio_13-20230412/train.json \
    --test_file ../data/bio_13-20230412/test.json \
    --features_file ../data/bio_13-20230412/features.json \
    --output_dir ../tmp/roberta \
    --do_train \
    --do_predict