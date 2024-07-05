PATH_TO_EVOL_INSTRUCT=/home/taozhengwei/hf_datasets/magicoder-evol-instruct/data-evol_instruct-decontaminated.jsonl
MAGICODER_OUTPUT_DIR=output
MODEL_NAME_OR_PATH=/home/taozhengwei/hf_models/deepseek-coder-6.7b-base

deepspeed --include localhost:0,1,2,5 --master_port=29501 train.py \
    --use_flash_attention False \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --max_training_seq_length 1024 \
    --datafile_paths \
        ${PATH_TO_EVOL_INSTRUCT} \
    --output_dir $MAGICODER_OUTPUT_DIR \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --group_by_length False \
    --ddp_find_unused_parameters False \
    --logging_steps 1 \
    --log_level info \
    --optim adafactor \
    --max_grad_norm -1 \
    --warmup_steps 15 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --deepspeed deepspeed/zero3.json