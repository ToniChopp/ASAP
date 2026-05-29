$PATH_TO_RIDER="Your path to the RIDER dataset"
$PATH_TO_COVID19_CT_Seg_20cases="Your path to the COVID19_CT_Seg_20cases dataset"

## RIDER
CUDA_VISIBLE_DEVICES=4 python test_external.py \
    --pretrained_path "./output/MSD/asap/10/model.pt" \
    --output_dir "./output/MSD/asap/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir $PATH_TO_RIDER \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=5 python test_external.py \
    --pretrained_path "./output/MSD/asap/100/model.pt" \
    --output_dir "./output/MSD/asap/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir $PATH_TO_RIDER \
    --batch_size 1 --sw_batch_size 1


## COVID19-CT-Seg-20Cases Lung
CUDA_VISIBLE_DEVICES=0 python test_external.py \
    --pretrained_path "./output/LUNA16/asap/1/model_final.pt" \
    --output_dir "./output/LUNA16/asap/1" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir $PATH_TO_COVID19_CT_Seg_20cases \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=1 python test_external.py \
    --pretrained_path "./output/LUNA16/asap/10/model_final.pt" \
    --output_dir "./output/LUNA16/asap/10" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir $PATH_TO_COVID19_CT_Seg_20cases \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=2 python test_external.py \
    --pretrained_path "./output/LUNA16/asap/100/model_final.pt" \
    --output_dir "./output/LUNA16/asap/100" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir $PATH_TO_COVID19_CT_Seg_20cases \
    --batch_size 1 --sw_batch_size 1




## COVID19-CT-Seg-20Cases Lung
CUDA_VISIBLE_DEVICES=6 python test_external.py \
    --pretrained_path "./output/Covid-19-20/asap/10/model.pt" \
    --output_dir "./output/Covid-19-20/asap/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir $PATH_TO_COVID19_CT_Seg_20cases \
    --batch_size 2 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=7 python test_external.py \
    --pretrained_path "./output/Covid-19-20/asap/100/model.pt" \
    --output_dir "./output/Covid-19-20/asap/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir $PATH_TO_COVID19_CT_Seg_20cases \
    --batch_size 2 --sw_batch_size 4