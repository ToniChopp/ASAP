$PATH_TO_ASAP_CHECKPOINT="Your path to the checkpoint of ASAP"
$PATH_TO_INSPECT="Your path to the INSPECT dataset"
$PATH_TO_STOIC="Your path to the STOIC dataset"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

## ASAP
## INSPECT
CUDA_VISIBLE_DEVICES=4 python train.py --name asap --stage train --model vit_base_patch16 --task INSPECT \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_INSPECT \
    --output_dir "output/INSPECT/asap/1" --data_volume '1' --num_steps 300  --eval_batch_size 48 \
    --learning_rate 5e-3 --warmup_steps 25 --fp16 --fp16_opt_level O2 --train_batch_size 32 \
    --patience 10 --start_epoch 5
CUDA_VISIBLE_DEVICES=1 python train.py --name asap --stage train --model vit_base_patch16 --task INSPECT \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_INSPECT \
    --output_dir "output/INSPECT/asap/10" --data_volume '10' --num_steps 3000  --eval_batch_size 48 \
    --learning_rate 5e-3 --warmup_steps 125 --fp16 --fp16_opt_level O2 --train_batch_size 64 \
    --patience 10 --start_epoch 10
CUDA_VISIBLE_DEVICES=5 python train.py --name asap --stage train --model vit_base_patch16 --task INSPECT \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_INSPECT \
    --output_dir "output/INSPECT/asap/100" --data_volume '100' --num_steps 20000  --eval_batch_size 48 \
    --learning_rate 5e-3 --warmup_steps 600 --fp16 --fp16_opt_level O2 --train_batch_size 128 \
    --patience 10 --start_epoch 10


## Stoic
CUDA_VISIBLE_DEVICES=3 python train_stoic.py --name asap --stage train --model vit_base_patch16 --task Stoic \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_STOIC \
    --output_dir "output/Stoic/asap/1" --data_volume '1' --num_steps 100  --eval_batch_size 48 \
    --learning_rate 1.5e-3 --warmup_steps 10 --fp16 --fp16_opt_level O2 --train_batch_size 9 \
    --patience 10 --start_epoch 5
CUDA_VISIBLE_DEVICES=7 python train_stoic.py --name asap --stage train --model vit_base_patch16 --task Stoic \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_STOIC \
    --output_dir "output/Stoic/asap/10" --data_volume '10' --num_steps 300  --eval_batch_size 48 \
    --learning_rate 3e-3 --warmup_steps 30 --fp16 --fp16_opt_level O2 --train_batch_size 32 \
    --patience 10 --start_epoch 10
CUDA_VISIBLE_DEVICES=0 python train_stoic.py --name asap --stage train --model vit_base_patch16 --task Stoic \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_STOIC \
    --output_dir "output/Stoic/asap/100" --data_volume '100' --num_steps 4000  --eval_batch_size 48 \
    --learning_rate 1e-2 --warmup_steps 80 --fp16 --fp16_opt_level O2 --train_batch_size 64 \
    --patience 10 --start_epoch 10