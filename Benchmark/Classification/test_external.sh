$PATH_TO_RSPECT="Your path to the RSPECT dataset"

## RSPECT
CUDA_VISIBLE_DEVICES=1 python test_external.py --name asap --stage test --model vit_base_patch16 --task RSPECT --num_classes 1 \
    --pretrained_path "./output/INSPECT/asap/10/asap_bestauc_checkpoint.bin" \
    --dataset_path $PATH_TO_RSPECT \
    --output_dir "output/external/RSPECT/asap/10" --eval_batch_size 64 \
    --fp16 --fp16_opt_level O2
CUDA_VISIBLE_DEVICES=2 python test_external.py --name asap --stage test --model vit_base_patch16 --task RSPECT --num_classes 1 \
    --pretrained_path "./output/INSPECT/asap/10/asap_bestauc_checkpoint.bin" \
    --dataset_path $PATH_TO_RSPECT \
    --output_dir "output/external/RSPECT/asap/100" --eval_batch_size 64 \
    --fp16 --fp16_opt_level O2