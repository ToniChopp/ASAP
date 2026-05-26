## RIDER
CUDA_VISIBLE_DEVICES=0 python test_external.py \
    --pretrained_path "./output/MSD/hiendmae/10/model.pt" \
    --output_dir "./output/MSD/hiendmae/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=1 python test_external.py \
    --pretrained_path "./output/MSD/hiendmae/100/model.pt" \
    --output_dir "./output/MSD/hiendmae/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1

CUDA_VISIBLE_DEVICES=2 python test_external.py \
    --pretrained_path "./output/MSD/ctclip/10/model.pt" \
    --output_dir "./output/MSD/ctclip/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=3 python test_external.py \
    --pretrained_path "./output/MSD/ctclip/100/model.pt" \
    --output_dir "./output/MSD/ctclip/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1

CUDA_VISIBLE_DEVICES=4 python test_external.py \
    --pretrained_path "./output/MSD/fvlm/10/model.pt" \
    --output_dir "./output/MSD/fvlm/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=5 python test_external.py \
    --pretrained_path "./output/MSD/fvlm/100/model.pt" \
    --output_dir "./output/MSD/fvlm/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1

CUDA_VISIBLE_DEVICES=6 python test_external.py \
    --pretrained_path "./output/MSD/HLIP/10/model.pt" \
    --output_dir "./output/MSD/HLIP/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=7 python test_external.py \
    --pretrained_path "./output/MSD/HLIP/100/model.pt" \
    --output_dir "./output/MSD/HLIP/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1

CUDA_VISIBLE_DEVICES=6 python test_external.py \
    --pretrained_path "./output/MSD/simcrop/10/model.pt" \
    --output_dir "./output/MSD/simcrop/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=7 python test_external.py \
    --pretrained_path "./output/MSD/simcrop/100/model.pt" \
    --output_dir "./output/MSD/simcrop/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1

CUDA_VISIBLE_DEVICES=4 python test_external.py \
    --pretrained_path "./output/MSD/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/10/model.pt" \
    --output_dir "./output/MSD/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=5 python test_external.py \
    --pretrained_path "./output/MSD/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/100/model.pt" \
    --output_dir "./output/MSD/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "dataset.json" \
    --task RIDER --data_dir "../../../Data/RIDER" \
    --batch_size 1 --sw_batch_size 1

CUDA_VISIBLE_DEVICES=0 python test_external.py \
    --pretrained_path "./output/MSD/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/100/model.pt" \
    --output_dir "./output/MSD/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "MSD_Task06_Lung_100.json" \
    --task MSD --data_dir "../../../Data/MSD_Task06_Lung" \
    --batch_size 1 --sw_batch_size 1


## COVID19-CT-Seg-20Cases Lung
CUDA_VISIBLE_DEVICES=0 python test_external.py \
    --pretrained_path "./output/LUNA16/hiendmae/1/model_final.pt" \
    --output_dir "./output/LUNA16/hiendmae/1" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=1 python test_external.py \
    --pretrained_path "./output/LUNA16/hiendmae/10/model_final.pt" \
    --output_dir "./output/LUNA16/hiendmae/10" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=2 python test_external.py \
    --pretrained_path "./output/LUNA16/hiendmae/100/model_final.pt" \
    --output_dir "./output/LUNA16/hiendmae/100" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1


CUDA_VISIBLE_DEVICES=0 python test_external.py \
    --pretrained_path "./output/LUNA16/ctclip/1/model_final.pt" \
    --output_dir "./output/LUNA16/ctclip/1" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=1 python test_external.py \
    --pretrained_path "./output/LUNA16/ctclip/10/model_final.pt" \
    --output_dir "./output/LUNA16/ctclip/10" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=2 python test_external.py \
    --pretrained_path "./output/LUNA16/ctclip/100/model_final.pt" \
    --output_dir "./output/LUNA16/ctclip/100" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1


CUDA_VISIBLE_DEVICES=0 python test_external.py \
    --pretrained_path "./output/LUNA16/fvlm/1/model_final.pt" \
    --output_dir "./output/LUNA16/fvlm/1" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=1 python test_external.py \
    --pretrained_path "./output/LUNA16/fvlm/10/model_final.pt" \
    --output_dir "./output/LUNA16/fvlm/10" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=2 python test_external.py \
    --pretrained_path "./output/LUNA16/fvlm/100/model_final.pt" \
    --output_dir "./output/LUNA16/fvlm/100" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1


CUDA_VISIBLE_DEVICES=0 python test_external.py \
    --pretrained_path "./output/LUNA16/HLIP/1/model_final.pt" \
    --output_dir "./output/LUNA16/HLIP/1" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=1 python test_external.py \
    --pretrained_path "./output/LUNA16/HLIP/10/model_final.pt" \
    --output_dir "./output/LUNA16/HLIP/10" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=2 python test_external.py \
    --pretrained_path "./output/LUNA16/HLIP/100/model_final.pt" \
    --output_dir "./output/LUNA16/HLIP/100" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1


CUDA_VISIBLE_DEVICES=0 python test_external.py \
    --pretrained_path "./output/LUNA16/simcrop/1/model_final.pt" \
    --output_dir "./output/LUNA16/simcrop/1" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=1 python test_external.py \
    --pretrained_path "./output/LUNA16/simcrop/10/model_final.pt" \
    --output_dir "./output/LUNA16/simcrop/10" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=2 python test_external.py \
    --pretrained_path "./output/LUNA16/simcrop/100/model_final.pt" \
    --output_dir "./output/LUNA16/simcrop/100" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1


CUDA_VISIBLE_DEVICES=0 python test_external.py \
    --pretrained_path "./output/LUNA16/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/1/model_final.pt" \
    --output_dir "./output/LUNA16/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/1" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=1 python test_external.py \
    --pretrained_path "./output/LUNA16/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/10/model_final.pt" \
    --output_dir "./output/LUNA16/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/10" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=2 python test_external.py \
    --pretrained_path "./output/LUNA16/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/100/model_final.pt" \
    --output_dir "./output/LUNA16/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/100" --out_channels 4 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Lung.json" \
    --task COVID19-20Cases_lung --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 1 --sw_batch_size 1




## COVID19-CT-Seg-20Cases Lung
CUDA_VISIBLE_DEVICES=6 python test_external.py \
    --pretrained_path "./output/Covid-19-20/hiendmae/10/model.pt" \
    --output_dir "./output/Covid-19-20/hiendmae/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=7 python test_external.py \
    --pretrained_path "./output/Covid-19-20/hiendmae/100/model.pt" \
    --output_dir "./output/Covid-19-20/hiendmae/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 4


CUDA_VISIBLE_DEVICES=6 python test_external.py \
    --pretrained_path "./output/Covid-19-20/ctclip/10/model.pt" \
    --output_dir "./output/Covid-19-20/ctclip/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=7 python test_external.py \
    --pretrained_path "./output/Covid-19-20/ctclip/100/model.pt" \
    --output_dir "./output/Covid-19-20/ctclip/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 4


CUDA_VISIBLE_DEVICES=6 python test_external.py \
    --pretrained_path "./output/Covid-19-20/fvlm/10/model.pt" \
    --output_dir "./output/Covid-19-20/fvlm/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=7 python test_external.py \
    --pretrained_path "./output/Covid-19-20/fvlm/100/model.pt" \
    --output_dir "./output/Covid-19-20/fvlm/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 4


CUDA_VISIBLE_DEVICES=6 python test_external.py \
    --pretrained_path "./output/Covid-19-20/HLIP/10/model.pt" \
    --output_dir "./output/Covid-19-20/HLIP/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=7 python test_external.py \
    --pretrained_path "./output/Covid-19-20/HLIP/100/model.pt" \
    --output_dir "./output/Covid-19-20/HLIP/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 4


CUDA_VISIBLE_DEVICES=6 python test_external.py \
    --pretrained_path "./output/Covid-19-20/simcrop/10/model.pt" \
    --output_dir "./output/Covid-19-20/simcrop/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=7 python test_external.py \
    --pretrained_path "./output/Covid-19-20/simcrop/100/model.pt" \
    --output_dir "./output/Covid-19-20/simcrop/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 4


CUDA_VISIBLE_DEVICES=6 python test_external.py \
    --pretrained_path "./output/Covid-19-20/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/10/model.pt" \
    --output_dir "./output/Covid-19-20/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/10" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 1
CUDA_VISIBLE_DEVICES=7 python test_external.py \
    --pretrained_path "./output/Covid-19-20/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/100/model.pt" \
    --output_dir "./output/Covid-19-20/simcrop_512_segmask_label_V2_warmup20_o_sigmoidk_crossksim_recon_topkscalecls_new/100" --out_channels 2 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 --json_path "Covid19_CT_Seg_Covid.json" \
    --task COVID19-20Cases_Covid --data_dir "../../../Data/COVID19_CT_Seg_20cases" \
    --batch_size 2 --sw_batch_size 4