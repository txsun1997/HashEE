python ./train_exit_model.py \
  --task_name SNLI \
  --lr 3e-5 \
  --batch_size 32 \
  --n_epochs 5 \
  --warmup 0.1 \
  --logging_steps 50 \
  --gpu 1 \
  --do_lower_case \
  --debug