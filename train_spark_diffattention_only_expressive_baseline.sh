#!/bin/bash
echpython train.py --output_dir ./outputs --overwrite_output_dir True \
                --bf16 True --bf16_full_eval True \
                --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
                --learning_rate 2e-4 --weight_decay 0.01 \
                --max_steps 40000 --warmup_steps 375 \
                --logging_strategy steps --logging_steps 100 \
                --report_to wandb --project_name neon-test --run_name spark-diffattn-baseline --watch all \
                --eval_strategy steps --eval_steps 1000 --eval_on_start True \
                --save_strategy steps --save_steps 1000 --save_total_limit 10 --load_best_model_at_end True \
                --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
                --num_train_samples 0 --num_eval_samples 5000 \
                --streaming True --packing True --dataset_text_field text --dataloader_num_workers 2 \
                --per_device_train_batch_size 8 --per_device_eval_batch_size 16 \
                --max_seq_length 2048 --gradient_accumulation_steps 16 \
                --seed 42 --diff_attention_mode expressive
