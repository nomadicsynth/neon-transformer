#!/bin/bash
python train.py --output_dir ./outputs/spark-diffattn-expressive-baseline --overwrite_output_dir True \
                --bf16 True --bf16_full_eval True \
                --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
                --learning_rate 3.2e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
                --max_steps 10000 --warmup_steps 500 \
                --logging_strategy steps --logging_steps 100 \
                --report_to wandb --project_name neon-test --run_name spark-diffattn-expressive-baseline --watch all \
                --eval_strategy steps --eval_steps 1000 --eval_on_start True \
                --save_strategy steps --save_steps 1000 --save_total_limit 10 --load_best_model_at_end True \
                --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
                --num_train_samples 0 --num_eval_samples 5000 \
                --streaming True --packing True --dataloader_num_workers 2 \
                --per_device_train_batch_size 8 --per_device_eval_batch_size 16 \
                --max_seq_length 2048 --gradient_accumulation_steps 16 \
                --seed 42 --diff_attention_mode expressive

python train.py --output_dir ./outputs/spark-diffattn-constrained-baseline --overwrite_output_dir True \
                --bf16 True --bf16_full_eval True \
                --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
                --learning_rate 3.2e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
                --max_steps 10000 --warmup_steps 500 \
                --logging_strategy steps --logging_steps 100 \
                --report_to wandb --project_name neon-test --run_name spark-diffattn-constrained-baseline --watch all \
                --eval_strategy steps --eval_steps 1000 --eval_on_start True \
                --save_strategy steps --save_steps 1000 --save_total_limit 10 --load_best_model_at_end True \
                --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
                --num_train_samples 0 --num_eval_samples 5000 \
                --streaming True --packing True --dataloader_num_workers 2 \
                --per_device_train_batch_size 8 --per_device_eval_batch_size 16 \
                --max_seq_length 2048 --gradient_accumulation_steps 16 \
                --seed 42 --diff_attention_mode constrained