export CUDA_VISIBLE_DEVICES=0
python ./train_token.py \
  --model_name_or_path fnlp/elasticbert-base \
  --data_dir ./ELUE_data \
  --task_name MRPC \
  --lr 5e-5 \
  --batch_size 32 \
  --n_epochs 4 \
  --warmup 0.1 \
  --weight_decay 0.01 \
  --logging_steps 50 \
  --gpu 0 \
  --hash frequency \
  --max_layer 7 \
  --num_buckets 6 \
  --do_lower_case \
  --seed 2021 \
  --debug