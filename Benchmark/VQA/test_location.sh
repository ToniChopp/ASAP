CUDA_VISIBLE_DEVICES=6 python inference.py \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --task "location" \
        --run_name asap_location \
        --output_dir ./output/asap/checkpoint-1015/
CUDA_VISIBLE_DEVICES=6 python inference.py \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --task "location" \
        --run_name asap_location \
        --output_dir ./output/asap/checkpoint-2030/
CUDA_VISIBLE_DEVICES=6 python inference.py \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --task "location" \
        --run_name asap_location \
        --output_dir ./output/asap/checkpoint-3045/
CUDA_VISIBLE_DEVICES=6 python inference.py \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --task "location" \
        --run_name asap_location \
        --output_dir ./output/asap/checkpoint-4060/
CUDA_VISIBLE_DEVICES=6 python inference.py \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --task "location" \
        --run_name asap_location \
        --output_dir ./output/asap/checkpoint-5075/