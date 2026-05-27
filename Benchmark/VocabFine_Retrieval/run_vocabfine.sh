$PATH_TO_ASAP_CHECKPOINT="Your path to the checkpoint of ASAP"
$PATH_TO_CTCLIP_CHECKPOINT="Your path to the checkpoint of CT-CLIP"
$PATH_TO_CTRATE="Your path to the CTRATE dataset"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 python ct_vocabfine_train.py \
    --lr 1e-3 \
    --wd 1e-4 \
    --epochs 20 \
    --warmup_length 5894 \
    --name "asap" \
    --save ./output_vocabfine/CT-Rate/asap/ \
    --vit_pretrained $PATH_TO_ASAP_CHECKPOINT \
    --pretrained $PATH_TO_CTCLIP_CHECKPOINT \
    --data-folder .$PATH_TO_CTRATE \
    --reports-file ./dataset/CT-Rate_train_reports.csv \
    --labels ./dataset/CT-Rate_train_labels.csv \
    --batch_size 16
