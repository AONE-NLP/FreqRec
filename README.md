# Overview
This is the official source code for FreqRec.

Code and implementation details of our paper `Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation`, accepted by AAAI 2026.

![]([F:\code\AAAI2025\FreqRec\FreqRec-main\photo]https://github.com/AONE-NLP/FreqRec/tree/main/photo\FreqRec.jpg)


## Dataset
In our experiments, we utilize six datasets, all stored in the `src/data` folder. 
- For the Beauty, Sports and Toys datasets, we employed the datasets downloaded from [this repository](https://github.com/Woeee/FMLP-Rec). 
- For CDs , Automotive,Grocery, we download form [this code](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon\_v2/).
- The `src/data/*_same_target.npy` files are utilized for training DuoRec and FEARec, both of which incorporate contrastive learning.

## Quick Start
### Environment Setting
```
conda env create -f freqrec_env.yaml
conda activate freqrec
```

### How to train FreqRec
- Note that pretrained model (.pt) and train log file (.log) will saved in `src/output`
- `train_name`: name for log file and checkpoint file
```python
# For Beauty dataset:
python main.py --alpha_loss 0.6 --gama 0.7 --data_name Beauty --lr 0.0005 --alpha 0.7 --num_attention_heads 1 --train_name freqrec_beauty --batch_size 512 --fft_loss_type l2 --num_hidden_layers 2 --model_type freqrec --chux p
# For Toys_and_Games dataset:
python main.py --alpha_loss 0.6 --gama 0.7 --data_name Toys_and_Games --lr 0.001 --alpha 0.7 --num_attention_heads 1 --train_name freqrec_Toys --batch_size 256 --fft_loss_type l1 --num_hidden_layers 2 --model_type freqrec --chux p
# For Sports_and_outdoors dataset:
python main.py --alpha_loss 0.7 --gama 0.8 --data_name Sports_and_outdoors --lr 0.001 --alpha 0.3 --num_attention_heads 4 --train_name freqrec_Sports --batch_size 256  --fft_loss_type l2 --num_hidden_layers 2 --model_type freqrec --chux p

```

## Citation

```

```



## Acknowledgement

This repository is based on [BSARec](https://github.com/yehjin-shin/BSARec/tree/main).

We are deeply grateful for the valuable code and efforts contributed by the following GitHub repositories. Their contributions have been immensely beneficial to our work.

[BSARec](https://github.com/yehjin-shin/BSARec/tree/main)

[DIFF](https://github.com/HyeYoung1218/DIFF)
