# Open Vocabulary Classification & Volume-Report Retrieval
Download the pre-trained weight of [CT-CLIP](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models_deprecated/CT_CLIP_zeroshot.pt)

### Training:
Modify the scripts [run_retrieval.sh](./run_retrieval.sh) and [run_vocabfine.sh](./run_vocabfine.sh):  
`--pretrained`  
`--vit_pretrained`  
`--data-folder`  
`--reports-file `  
`--labels`  

Open Vocabulary Classification training:
```bash
bash run_vocabfine.sh
```
Volume-Report Retrieval training
```bash
bash run_retrieval.sh
```

### Evaluation:
Modify the scripts [test_retrieval.sh](./test_retrieval.sh) and [test_vocabfine.sh](./test_vocabfine.sh):  
```--checkpoint_path```  

Open Vocabulary Classification evaluation: 
```bash
bash test_vocabfine.sh
```
Volume-Report Retrieval evaluation: 
```bash
bash test_retrieval.sh
```


## Acknowledgement
Some codes are borrowed from [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) and [VOCO](https://github.com/Luffy03/Large-Scale-Medical/tree/main/Downstream/monai/CT_CLIP).