# Visual Question Answering
Download the weights of [Qwen3-4B-Instruct]() and move to [model](./model) folder.

Modify the `volume_root` in [preprocess.py](./preprocess.py) to your volume dir.
### Training:
```
python preprocess.py
bash run_lora.sh
```

### Evaluation:
```
bash test_abnormality.sh
bash test_presence.sh
bash test_size.sh
bash test_location.sh
bash test_disorder.sh
```



## Aknowledgement
Some codes are borrowed from [RadGenome-ChestCT](https://github.com/xiaoman-zhang/RadGenome-ChestCT).