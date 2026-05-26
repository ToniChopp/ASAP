$PATH_TO_ASAP_CHECKPOINT="Your path to the checkpoint of ASAP"
$PATH_TO_LUNA16="Your path to the LUNA16 dataset"
$PATH_TO_MSD_LUNG="Your path to the MSD Lung dataset"
$PATH_TO_COVID_19_20="Your path to the COVID-19-20 dataset"
$PATH_TO_FUMPE="Your path to the FUMPE dataset"
$PATH_TO_SEGTHOR="Your path to the SegThor dataset"
$PATH_TO_BTCV="Your path to the BTCV dataset"
$PATH_TO_ACDC="Your path to the ACDC dataset"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

## LUNA16
CUDA_VISIBLE_DEVICES=0 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/LUNA16/asap/1" --out_channels 4 \
    --optim_lr 1e-4 --num_samples 32 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset_1.json" \
    --task LUNA16 --data_dir $PATH_TO_LUNA16 \
    --warmup_epochs 100 --max_epochs 1000  --val_every 5 \
    --batch_size 1 --sw_batch_size 1 --patience 20 --start_val_epochs 100
CUDA_VISIBLE_DEVICES=1 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/LUNA16/asap/10" --out_channels 4 \
    --optim_lr 1.5e-4 --num_samples 32 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset_10.json" \
    --task LUNA16 --data_dir $PATH_TO_LUNA16 \
    --warmup_epochs 50 --max_epochs 1000  --val_every 5 \
    --batch_size 1 --sw_batch_size 1 --patience 20 --start_val_epochs 50
CUDA_VISIBLE_DEVICES=3 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/LUNA16/asap/100" --out_channels 4 \
    --optim_lr 1e-4 --num_samples 32 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset_100.json" \
    --task LUNA16 --data_dir $PATH_TO_LUNA16 \
    --warmup_epochs 100 --max_epochs 1000  --val_every 10 \
    --batch_size 1 --sw_batch_size 1 --patience 10 --start_val_epochs 100

## MSD LUNG
CUDA_VISIBLE_DEVICES=1 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/MSD/asap/10" --out_channels 2 \
    --optim_lr 1e-4 --num_samples 16 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "MSD_Task06_Lung_10.json" \
    --task MSD --data_dir $PATH_TO_MSD_LUNG \
    --warmup_epochs 100 --max_epochs 1000  --val_every 5 \
    --batch_size 2 --sw_batch_size 1  --patience 20 --start_val_epochs 100
CUDA_VISIBLE_DEVICES=0 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/MSD/asap/100" --out_channels 2 \
    --optim_lr 1e-4 --num_samples 16 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "MSD_Task06_Lung_100.json" \
    --task MSD --data_dir $PATH_TO_MSD_LUNG \
    --warmup_epochs 50 --max_epochs 1000  --val_every 5 \
    --batch_size 2 --sw_batch_size 4 --patience 20 --start_val_epochs 50

## COVID-19-20
CUDA_VISIBLE_DEVICES=6 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/Covid-19-20/asap/10" --out_channels 2 \
    --optim_lr 3e-4 --num_samples 32 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset_10.json" \
    --task COVID-19-20 --data_dir $PATH_TO_COVID_19_20 \
    --warmup_epochs 80 --max_epochs 800  --val_every 5 \
    --batch_size 1 --sw_batch_size 4 --patience 20  --start_val_epochs 80
CUDA_VISIBLE_DEVICES=7 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/Covid-19-20/asap/100" --out_channels 2 \
    --optim_lr 3e-4 --num_samples 32 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset_100.json" \
    --task COVID-19-20 --data_dir $PATH_TO_COVID_19_20 \
    --warmup_epochs 50 --max_epochs 1000  --val_every 5 \
    --batch_size 1 --sw_batch_size 4 --patience 20  --start_val_epochs 50

## FUMPE
CUDA_VISIBLE_DEVICES=2 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/FUMPE/asap" --out_channels 4 \
    --optim_lr 1e-3 --num_samples 16 --pos 3 --neg 1\
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task FUMPE --data_dir $PATH_TO_FUMPE \
    --warmup_epochs 5 --max_epochs 800  --val_every 5 \
    --batch_size 2 --sw_batch_size 4 --patience 20 --start_val_epochs 50

## SegThor
CUDA_VISIBLE_DEVICES=1 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/SegThor/asap/10" --out_channels 5 \
    --optim_lr 3e-4 --num_samples 32 --pos 3 --neg 1 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset_10.json" \
    --task SegThor --data_dir $PATH_TO_SEGTHOR \
    --warmup_epochs 5 --max_epochs 1000  --val_every 10 \
    --batch_size 1 --sw_batch_size 4 --patience 20
CUDA_VISIBLE_DEVICES=5 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/SegThor/asap" --out_channels 5 \
    --optim_lr 3e-4 --num_samples 32 --pos 3 --neg 1 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task SegThor --data_dir $PATH_TO_SEGTHOR \
    --warmup_epochs 5 --max_epochs 1000  --val_every 5 \
    --batch_size 1 --sw_batch_size 4 --patience 20 --start_val_epochs 50

## BTCV
CUDA_VISIBLE_DEVICES=4 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/BTCV/asap" --out_channels 14 \
    --optim_lr 3e-4 --num_samples 32 \
    --a_min \-175 --a_max 250 \
    --RandShiftIntensityd_prob 0.5 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset_5.json" \
    --task BTCV --data_dir $PATH_TO_BTCV \
    --warmup_epochs 100 --max_epochs 2000  --val_every 20 \
    --batch_size 1 --sw_batch_size 4 --patience 20  --start_val_epochs 100

## ACDC
CUDA_VISIBLE_DEVICES=4 python main.py \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --output_dir "./output/ACDC/asap" --out_channels 4 \
    --optim_lr 3e-4 --num_samples 16 --pos 3 --neg 1 \
    --space_x 1.0 --space_y 1.0 --space_z 1.5 --json_path "dataset.json" \
    --roi_x 192 --roi_y 192 --roi_z 32 \
    --patch_size_x 24 --patch_size_y 24 --patch_size_z 4 \
    --task ACDC --data_dir $PATH_TO_ACDC \
    --warmup_epochs 5 --max_epochs 500  --val_every 5 \
    --batch_size 1 --sw_batch_size 2 --patience 20 --start_val_epochs 50

