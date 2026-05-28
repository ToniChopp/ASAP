$PATH_TO_ASAP_CHECKPOINT="Your path to the checkpoint of ASAP"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --master_port=1234 --nproc_per_node=4 global_train_lora.py \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --save_steps  100 \
        --num_train_epochs 5 \
        --lora_rank  8 \
        --save_total_limit 5 \
        --learning_rate 5e-5 \
        --weight_decay 0. \
        --warmup_steps 500 \
        --num_training_steps 5075 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --bf16 True \
        --vision_learnable False \
        --vision_backbone "ViT" \
        --llm_max_length 384 \
        --run_name asap_qwen3_global_lora_qformer \
        --output_dir ./output/asap \
        --vision_pretrained $PATH_TO_ASAP_CHECKPOINT