#!/bin/bash
# python train.py --output_dir "./outputs/test-diffattn-memory" --overwrite_output_dir True \
#                 --model_size test --bf16 True --bf16_full_eval True \
#                 --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
#                 --learning_rate 3.6e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
#                 --max_steps 40000 --warmup_steps 500 \
#                 --logging_strategy steps --logging_steps 100 \
#                 --report_to wandb --watch all --wandb_log_model "end" \
#                 --project_name neon-test --run_name "test-diffattn-memory" \
#                 --eval_strategy steps --eval_steps 200 --eval_on_start True \
#                 --save_strategy steps --save_steps 200 --save_total_limit 10 --load_best_model_at_end True \
#                 --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
#                 --num_train_samples 0 --num_eval_samples 512 \
#                 --streaming True --packing True --dataloader_num_workers 2 --dataloader_prefetch_factor 8 \
#                 --per_device_train_batch_size 320 --per_device_eval_batch_size 256 \
#                 --max_seq_length 128 --gradient_accumulation_steps 1 \
#                 --seed 42 --diff_attention_mode "expressive" \
#                 --num_global_memories 32 --num_layer_memories 16

# python train.py --output_dir "./outputs/test-diffattn-memory" --overwrite_output_dir True \
#                 --model_size test --bf16 True --bf16_full_eval True \
#                 --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
#                 --learning_rate 4.2e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
#                 --max_steps 40000 --warmup_steps 500 \
#                 --logging_strategy steps --logging_steps 100 \
#                 --report_to wandb --watch all --wandb_log_model "end" \
#                 --project_name neon-test --run_name "test-diffattn-memory-32-32" \
#                 --eval_strategy steps --eval_steps 200 --eval_on_start True \
#                 --save_strategy steps --save_steps 200 --save_total_limit 10 --load_best_model_at_end True \
#                 --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
#                 --num_train_samples 0 --num_eval_samples 512 \
#                 --streaming True --packing True --dataloader_num_workers 4 --dataloader_prefetch_factor 8 \
#                 --per_device_train_batch_size 320 --per_device_eval_batch_size 256 \
#                 --max_seq_length 128 --gradient_accumulation_steps 1 \
#                 --seed 42 --diff_attention_mode "expressive" \
#                 --num_global_memories 32 --num_layer_memories 32

# python train.py --output_dir "./outputs/test-diffattn-memory" --overwrite_output_dir True \
#                 --model_size test --bf16 True --bf16_full_eval True \
#                 --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
#                 --learning_rate 4.2e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
#                 --max_steps 40000 --warmup_steps 500 \
#                 --logging_strategy steps --logging_steps 100 \
#                 --report_to wandb --watch all --wandb_log_model "end" \
#                 --project_name neon-test --run_name "test-diffattn-memory-24-8" \
#                 --eval_strategy steps --eval_steps 200 --eval_on_start True \
#                 --save_strategy steps --save_steps 200 --save_total_limit 10 --load_best_model_at_end True \
#                 --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
#                 --num_train_samples 0 --num_eval_samples 512 \
#                 --streaming True --packing True --dataloader_num_workers 2 --dataloader_prefetch_factor 8 \
#                 --per_device_train_batch_size 320 --per_device_eval_batch_size 256 \
#                 --max_seq_length 128 --gradient_accumulation_steps 1 \
#                 --seed 42 --diff_attention_mode "expressive" \
#                 --num_global_memories 32 --num_layer_memories 8

python train.py --output_dir "./outputs/test-diffattn-memory" --overwrite_output_dir True \
                --model_size test --bf16 True --bf16_full_eval True \
                --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
                --learning_rate 4.2e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
                --max_steps 40000 --warmup_steps 500 \
                --logging_strategy steps --logging_steps 100 \
                --report_to wandb --watch all --wandb_log_model "end" \
                --project_name neon-test --run_name "test-diffattn-memory-24-8" \
                --eval_strategy steps --eval_steps 200 --eval_on_start True \
                --save_strategy steps --save_steps 200 --save_total_limit 10 --load_best_model_at_end True \
                --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
                --num_train_samples 0 --num_eval_samples 512 \
                --streaming True --packing True --dataloader_num_workers 2 --dataloader_prefetch_factor 8 \
                --per_device_train_batch_size 320 --per_device_eval_batch_size 256 \
                --max_seq_length 128 --gradient_accumulation_steps 1 \
                --seed 42 --diff_attention_mode "expressive" \
                --num_global_memories 24 --num_layer_memories 8

python train.py --output_dir "./outputs/test-diffattn-memory" --overwrite_output_dir True \
                --model_size test --bf16 True --bf16_full_eval True \
                --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
                --learning_rate 4.2e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
                --max_steps 40000 --warmup_steps 500 \
                --logging_strategy steps --logging_steps 100 \
                --report_to wandb --watch all --wandb_log_model "end" \
                --project_name neon-test --run_name "test-diffattn-memory-24-8" \
                --eval_strategy steps --eval_steps 200 --eval_on_start True \
                --save_strategy steps --save_steps 200 --save_total_limit 10 --load_best_model_at_end True \
                --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
                --num_train_samples 0 --num_eval_samples 512 \
                --streaming True --packing True --dataloader_num_workers 2 --dataloader_prefetch_factor 8 \
                --per_device_train_batch_size 320 --per_device_eval_batch_size 256 \
                --max_seq_length 128 --gradient_accumulation_steps 1 \
                --seed 42 --diff_attention_mode "expressive" \
                --num_global_memories 24 --num_layer_memories 24

python train.py --output_dir "./outputs/test-diffattn-memory" --overwrite_output_dir True \
                --model_size test --bf16 True --bf16_full_eval True \
                --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
                --learning_rate 4.2e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
                --max_steps 40000 --warmup_steps 500 \
                --logging_strategy steps --logging_steps 100 \
                --report_to wandb --watch all --wandb_log_model "end" \
                --project_name neon-test --run_name "test-diffattn-memory-24-8" \
                --eval_strategy steps --eval_steps 200 --eval_on_start True \
                --save_strategy steps --save_steps 200 --save_total_limit 10 --load_best_model_at_end True \
                --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
                --num_train_samples 0 --num_eval_samples 512 \
                --streaming True --packing True --dataloader_num_workers 2 --dataloader_prefetch_factor 8 \
                --per_device_train_batch_size 320 --per_device_eval_batch_size 256 \
                --max_seq_length 128 --gradient_accumulation_steps 1 \
                --seed 42 --diff_attention_mode "expressive" \
                --num_global_memories 16 --num_layer_memories 24

python train.py --output_dir "./outputs/test-diffattn-memory" --overwrite_output_dir True \
                --model_size test --bf16 True --bf16_full_eval True \
                --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
                --learning_rate 4.2e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
                --max_steps 40000 --warmup_steps 500 \
                --logging_strategy steps --logging_steps 100 \
                --report_to wandb --watch all --wandb_log_model "end" \
                --project_name neon-test --run_name "test-diffattn-memory-24-8" \
                --eval_strategy steps --eval_steps 200 --eval_on_start True \
                --save_strategy steps --save_steps 200 --save_total_limit 10 --load_best_model_at_end True \
                --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
                --num_train_samples 0 --num_eval_samples 512 \
                --streaming True --packing True --dataloader_num_workers 2 --dataloader_prefetch_factor 8 \
                --per_device_train_batch_size 320 --per_device_eval_batch_size 256 \
                --max_seq_length 128 --gradient_accumulation_steps 1 \
                --seed 42 --diff_attention_mode "expressive" \
                --num_global_memories 64 --num_layer_memories 64

python train.py --output_dir "./outputs/test-diffattn-memory" --overwrite_output_dir True \
                --model_size test --bf16 True --bf16_full_eval True \
                --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
                --learning_rate 4.2e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
                --max_steps 40000 --warmup_steps 500 \
                --logging_strategy steps --logging_steps 100 \
                --report_to wandb --watch all --wandb_log_model "end" \
                --project_name neon-test --run_name "test-diffattn-memory-24-8" \
                --eval_strategy steps --eval_steps 200 --eval_on_start True \
                --save_strategy steps --save_steps 200 --save_total_limit 10 --load_best_model_at_end True \
                --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
                --num_train_samples 0 --num_eval_samples 512 \
                --streaming True --packing True --dataloader_num_workers 2 --dataloader_prefetch_factor 8 \
                --per_device_train_batch_size 320 --per_device_eval_batch_size 256 \
                --max_seq_length 128 --gradient_accumulation_steps 1 \
                --seed 42 --diff_attention_mode "expressive" \
                --num_global_memories 64 --num_layer_memories 48
