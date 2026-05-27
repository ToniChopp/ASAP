for i in {1..20}
do
    CUDA_VISIBLE_DEVICES=7 python zero_shot_vocabfine.py \
        --name "asap" \
        --checkpoint_path ./output_vocabfine/CT-Rate/asap/epoch_${i}.pt
done