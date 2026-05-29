# ASAP Bechmark
We release implementations of 20 downstream tasks across various medical tasks (2 with AHPH-10K are in-house data), including abnormality classification, segmentation, disease prognosis prediction, report generation, vocabulary classification, cross-modal retrieval, and visual question answering.

Here, we provid the official links of benchmark datasets, you can use [preprocess.py](../preprocess/preprocess.py) to preprocess the datasets.
| Dataset                                                            | Modality      | Task                                                               |
|--------------------------------------------------------------------|---------------|--------------------------------------------------------------------|
| [CT-Rate](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)  | Chest CT      | Abnormality Cls., Report Gen., Vocabulary Cls., Volume-report Ret. |
| [STOIC-2021](https://stoic2021.grand-challenge.org/stoic-db/)      | Chest CT      | COVID Cls., Prognosis Pred.                                        |
| [Rad-ChestCT](https://cvit.duke.edu/resource/rad-chestct-dataset/) | Chest CT      | Abnormality Cls.                                                   |
| [LUNA16](https://luna16.grand-challenge.org/Data/)                 | Chest CT      | Lung Nodule Cls., Lung Seg.                                        |
| [INSPECT](https://som-shahlab.github.io/inspect-website/)          | Chest CT      | Pulmonary Embolism Cls., Prognosis Pred.                           |
| [CC-CCII](https://www.kaggle.com/datasets/fakaframe082/cc-ccii)    | Chest CT      | COVID & Pneumonia Cls.                                             |
| [MSD-Lung](https://huggingface.co/datasets/Angelou0516/msd-lung)   | Chest CT      | Lung Tumor Seg.                                                    |
| [COVID-19-20](https://covid-segmentation.grand-challenge.org/Data/)| Chest CT      | COVID Seg.                                                         |
| [SegThor](https://competitions.codalab.org/competitions/21145)     | Chest CT      | Thoracic Risk Seg.                                                 |
| [FUMPE](https://www.kaggle.com/datasets/andrewmvd/pulmonary-embolism-in-ct-images)| Chest CT      | Pulmonary Embolism Seg.                             |
| [BTCV](https://figshare.com/articles/dataset/BTCV_dataset/29077214)| Abdomen CT    | Abdomen Seg.                                                       |
| [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)           | MRI           | Heart Seg.                                                         |
| [CTRG](https://huggingface.co/datasets/Trusure/CTRG-Chest-548K_volume)| Chest CT   | Report Gen.                                                        |
| [RadGenome-Chest CT](https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT)| Chest CT   | VQA                                                     |
| [RSPECT](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pe-detection-challenge-2020)| Chest CT      | Pulmonary Embolism Cls. (**external validation**)  |
| [RIDER](https://www.cancerimagingarchive.net/analysis-result/rider-lungct-seg/)| Chest CT    | Lung Tumor Seg. (**external validation**)                |
| [COVID-19-CT-Seg](https://zenodo.org/records/3757476#.Xpz8OcgzZPY) | Chest CT      | Lung Seg., COVID Seg. (**external validation**)                    |


## Fine-tuning
Here, we take abnormality classification task as an example:
```bash
cd Classification
source activate ASAP
bash run.sh
```

A template for run.sh should be like:
```bash
$PATH_TO_ASAP_CHECKPOINT="Your path to the checkpoint of ASAP"
$PATH_TO_DATASET="Your path to the dataset"
$TASK="Your fine-tuning task for classification"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python train.py --name asap --stage train --model vit_base_patch16 --task $TASK --num_classes 18 \
    --pretrained_path $PATH_TO_ASAP_CHECKPOINT \
    --dataset_path $PATH_TO_CTRATE \
    --output_dir "output/CT-Rate/asap/1/" --data_volume '1' --num_steps 6000  --eval_batch_size 48 \
    --learning_rate 1.5e-3 --warmup_steps 150 --fp16 --train_batch_size 32 --weight_decay 1e-2 \
    --patience 10 --start_epoch 20 
```

Parameters you need to modify:
`--name`: The name of pre-trained models.
`--stage`: The stage of current script, *train* or *test* is expected
`--task`: Your fine-tuning task for classification, ["CT-Rate", "RadChestCT", "CC-CCII", "LUNA16", "INSPECT", "RSPECT", "Stoic"]
`--pretrained_path`: Your path to the checkpoint of pre-trained model
`--dataset_path`: Your path to the dataset used for the task

Validation and testing are incorperated in the training codes.

## External Validation
Here, we take abnormality classification task as an example:
```bash
cd Classification
source activate ASAP
bash test_external.sh
```

A template for test_external.sh should be like:
```bash
$PATH_TO_RSPECT="Your path to the RSPECT dataset"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=1 python test_external.py --name asap --stage test --model vit_base_patch16 --task RSPECT --num_classes 1 \
    --pretrained_path "./output/INSPECT/asap/10/simcrop_bestauc_checkpoint.bin" \
    --dataset_path $PATH_TO_RSPECT \
    --output_dir "output/external/RSPECT/asap/10" --eval_batch_size 64 \
    --fp16 --fp16_opt_level O2
```

Parameters you need to modify:
`--dataset_path`: Your path to the dataset used for the task


## Downstream Checkpoints
We will upload the fine-tuned checkpoints upon acceptance!