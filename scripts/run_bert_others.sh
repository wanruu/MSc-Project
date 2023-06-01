python ../src/run_bert_others.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --train_file ../data/bio_11-20230531/train.json \
    --test_file ../data/bio_11-20230531/test.json \
    --features_file ../data/bio_11-20230531/features.json \
    --output_dir ../tmp/roberta_others_bio11 \
    --do_train \
    --do_predict