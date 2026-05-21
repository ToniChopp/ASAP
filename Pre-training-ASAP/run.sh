# export WANDB_MODE=disabled
source activate simcrop
cd /workspace/ASAP/Pre-training-ASAP/
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 main_pretrain.py \
    --num_workers 14 \
    --accum_iter 8 \
    --batch_size 14 \
    --model asap \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --max_epochs 200 \
    --epochs 200 \
    --warmup_epochs 20 \
    --lr 1.5e-4 --weight_decay 0.05 \
    --data_path ./dataset/ \
    --dataset_path ../../../CT-Rate \
    --resume ./checkpoints/checkpoint_CXRBERT_imagenetMAE.pth \
    --output_dir ../output_asap_womlp \
    --description "ASAP Pretraining based on the manustript" \
