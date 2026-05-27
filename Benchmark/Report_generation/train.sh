$PATH_TO_ASAP_CHECKPOINT="Your path to the checkpoint of ASAP"

CUDA_VISIBLE_DEVICES=2 python main.py \
   --cfg ./config/ctrg_vit.yml \
   --version ViT \
   --name asap \
   --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
   --batch_size 4  --lr_ed 1e-4  --lr_scheduler "cosine"




CUDA_VISIBLE_DEVICES=6 python main.py \
   --cfg ./config/CT-Rate_vit.yml \
   --version ViT \
   --name asap \
   --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
   --batch_size 8  --lr_ed 5e-5  --lr_scheduler "cosine" --epochs 30 --max_epochs 100
