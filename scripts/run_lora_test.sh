python ../src/run_lora.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name conll2003 \
    --output_dir /tmp/test-ner \
    --do_train \
    --do_eval