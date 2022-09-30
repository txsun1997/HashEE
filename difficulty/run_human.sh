python ./train_human_difficulty.py \
  --task_name SNLI \
  --lr 3e-5 \
  --batch_size 32 \
  --n_epochs 10 \
  --warmup 0.1 \
  --logging_steps 10 \
  --gpu 1 \
  --do_lower_case