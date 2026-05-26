$PATH_TO_ASAP_CHECKPOINT="Your path to the checkpoint of ASAP"
$PATH_TO_CTRATE="Your path to the CT-Rate dataset"
$PATH_TO_STOIC="Your path to the STOIC dataset"
$PATH_TO_RADCHESTCT="Your path to the RadChestCT dataset"
$PATH_TO_LUNA16="Your path to the LUNA16 dataset"
$PATH_TO_INSPECT="Your path to the INSPECT dataset"
$PATH_TO_CC_CCII="Your path to the CC-CCII dataset"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


## ASAP
## CT-RATE 
CUDA_VISIBLE_DEVICES=0 python train.py --name asap --stage train --model vit_base_patch16 --task CT-Rate --num_classes 18 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_CTRATE \
    --output_dir "output/CT-Rate/asap/1/" --data_volume '1' --num_steps 6000  --eval_batch_size 48 \
    --learning_rate 1.5e-3 --warmup_steps 150 --fp16 --train_batch_size 32 --weight_decay 1e-2 \
    --patience 10 --start_epoch 20 
CUDA_VISIBLE_DEVICES=3 python train.py --name asap --stage train --model vit_base_patch16 --task CT-Rate --num_classes 18 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_CTRATE \
    --output_dir "output/CT-Rate/asap/10" --data_volume '10' --num_steps 30000  --eval_batch_size 48 \
    --learning_rate 3e-3 --warmup_steps 750 --fp16 --train_batch_size 64 --weight_decay 1e-2 \
    --patience 10 --start_epoch 20
CUDA_VISIBLE_DEVICES=2 python train.py --name asap --stage train --model vit_base_patch16 --task CT-Rate --num_classes 18 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_CTRATE \
    --output_dir "output/CT-Rate/asap/100" --data_volume '100' --num_steps 100000  --eval_batch_size 48 \
    --learning_rate 5e-3 --warmup_steps 1800 --fp16 --train_batch_size 128 \
    --patience 10 --start_epoch 20


## Stoic
CUDA_VISIBLE_DEVICES=3 python train.py --name asap --stage train --model vit_base_patch16 --task Stoic --num_classes 1 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_STOIC \
    --output_dir "output/Stoic/asap/1" --data_volume '1' --num_steps 100  --eval_batch_size 48 \
    --learning_rate 1.5e-3 --warmup_steps 20 --fp16 --train_batch_size 16 --weight_decay 1e-2 \
    --patience 20 --start_epoch 5
CUDA_VISIBLE_DEVICES=1 python train.py --name asap --stage train --model vit_base_patch16 --task Stoic --num_classes 1 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_STOIC \
    --output_dir "output/Stoic/asap/10" --data_volume '10' --num_steps 1000  --eval_batch_size 48 \
    --learning_rate 3e-3 --warmup_steps 50 --fp16 --train_batch_size 64 --weight_decay 1e-2 \
    --patience 20 --start_epoch 10
CUDA_VISIBLE_DEVICES=6 python train.py --name asap --stage train --model vit_base_patch16 --task Stoic --num_classes 1 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_STOIC \
    --output_dir "output/Stoic/asap/100" --data_volume '100' --num_steps 5000  --eval_batch_size 48 \
    --learning_rate 3e-3 --warmup_steps 300 --fp16 --train_batch_size 64 --weight_decay 1e-2 \
    --patience 20 --start_epoch 10


## RadChestCT
CUDA_VISIBLE_DEVICES=0 python train.py --name asap --stage train --model vit_base_patch16 --task RadChestCT --num_classes 16 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_RADCHESTCT \
    --output_dir "output/RadChestCT/asap/10" --data_volume '10' --num_steps 3000  --eval_batch_size 48 \
    --learning_rate 1.5e-3 --warmup_steps 160 --fp16 --train_batch_size 16 \
    --patience 20 --start_epoch 10
CUDA_VISIBLE_DEVICES=1 python train.py --name asap --stage train --model vit_base_patch16 --task RadChestCT --num_classes 16 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_RADCHESTCT \
    --output_dir "output/RadChestCT/asap/100" --data_volume '100' --num_steps 6000  --eval_batch_size 48 \
    --learning_rate 3e-3 --warmup_steps 200 --fp16 --train_batch_size 128 \
    --patience 20 --start_epoch 10


## LUNA16
CUDA_VISIBLE_DEVICES=0 python train.py --name asap --stage train --model vit_base_patch16 --task LUNA16 --num_classes 1 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_LUNA16 \
    --output_dir "output/LUNA16/asap/10" --data_volume '10' --num_steps 500  --eval_batch_size 48 \
    --learning_rate 1.5e-3 --warmup_steps 40 --fp16 --train_batch_size 31 --weight_decay 1e-2 \
    --patience 40 --start_epoch 10
CUDA_VISIBLE_DEVICES=1 python train.py --name asap --stage train --model vit_base_patch16 --task LUNA16 --num_classes 1 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_LUNA16 \
    --output_dir "output/LUNA16/asap/100" --data_volume '100' --num_steps 2000  --eval_batch_size 48 \
    --learning_rate 3e-2 --warmup_steps 200 --fp16 --train_batch_size 128 \
    --patience 40 --start_epoch 10


## INSPECT
CUDA_VISIBLE_DEVICES=6 python train.py --name asap --stage train --model vit_base_patch16 --task INSPECT --num_classes 1 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_INSPECT \
    --output_dir "output/INSPECT/asap/10" --data_volume '10' --num_steps 8000  --eval_batch_size 48 \
    --learning_rate 1.5e-2 --warmup_steps 300 --fp16 --train_batch_size 32 \
    --a_min \-100 --a_max 500 \
    --patience 10 --start_epoch 10
CUDA_VISIBLE_DEVICES=1 python train.py --name asap --stage train --model vit_base_patch16 --task INSPECT --num_classes 1 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_INSPECT \
    --output_dir "output/INSPECT/asap/100" --data_volume '100' --num_steps 20000  --eval_batch_size 48 \
    --learning_rate 1e-2 --warmup_steps 1000 --fp16 --train_batch_size 64 \
    --patience 10 --start_epoch 10


## CC-CCII
CUDA_VISIBLE_DEVICES=0 python train.py --name asap --stage train --model vit_base_patch16 --task CC-CCII --num_classes 3 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_CC_CCII \
    --output_dir "output/CC-CCII/asap/1" --data_volume '1' --num_steps 600  --eval_batch_size 48 \
    --learning_rate 1e-2 --warmup_steps 80 --fp16 --train_batch_size 24 --weight_decay 1e-2 \
    --patience 40 --start_epoch 80
CUDA_VISIBLE_DEVICES=1 python train.py --name asap --stage train --model vit_base_patch16 --task CC-CCII --num_classes 3 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_CC_CCII \
    --output_dir "output/CC-CCII/asap/10" --data_volume '10' --num_steps 2000  --eval_batch_size 48 \
    --learning_rate 1e-2 --warmup_steps 40 --fp16 --train_batch_size 128 --weight_decay 1e-2 \
    --patience 40 --start_epoch 80
CUDA_VISIBLE_DEVICES=4 python train.py --name asap --stage train --model vit_base_patch16 --task CC-CCII --num_classes 3 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_CC_CCII \
    --output_dir "output/CC-CCII/asap/100" --data_volume '100' --num_steps 20000  --eval_batch_size 48 \
    --learning_rate 1e-2 --warmup_steps 200 --fp16 --train_batch_size 128 --weight_decay 1e-2 \
    --patience 20 --start_epoch 80


