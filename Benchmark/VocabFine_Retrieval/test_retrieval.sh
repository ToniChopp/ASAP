for i in {1..40}
do
    CUDA_VISIBLE_DEVICES=1 python zero_shot_retrieval.py \
        --name "asap" \
        --checkpoint_path ./output_retrieval/CT-Rate/asap/epoch_${i}.pt
done