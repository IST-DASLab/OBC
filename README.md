# Optimal Brain Compression

This repository contains efficient implementations of ExactOBS for quantization,
unstructured-, block- and N:M pruning, introduced in the NeurIPS 2022 paper 
"Optimal Brain Compression: A Framework for Accurate Post-Training Quantization 
and Pruning".

## Files

* `trueobs.py`: efficient implementations of ExactOBS for all compression types
* `main_trueobs.py`: code to run ExactOBS 
* `post_proc.py`: post processing operations like statistics corrections
* `database.py`: generating databases for non-uniform compression
* `spdy.py`: implementation of the DP algorithm for finding non-uniform
  compression configurations; adapted from code provided by the authors of SPDY [9]
* `modelutils.py`: model utilities
* `datautils.py`: data utilities
* `quant.py`: quantization utilities

NOTE: The code as provided here only fully supports torchvision ResNet variants
(the full integration of YOLO and BERT models is omitted due to large amounts
of complex dependencies).

## Usage 

First, make sure ImageNet is located/linked to `../imagenet` (alternatively,
you can specifiy the `--datapath` argument for all commands).

### Applying OBC

```
# Quantize weights and activations
python main_trueobs.py rn18 imagenet quant --wbits 4 --abits 4 --save rn18_4w4a.pth

# Prune to the N:M pattern
python main_trueobs.py rn18 imagenet nmprune --prunen 2 --prunem 4 --save rn18_24.pth

# Generate an unstructured pruning database
mkdir models_unstr
python main_trueobs.py rn18 imagenet unstr --sparse-dir models_unstr

# Generate a 4-block pruning database
mkdir models_4block
python main_trueobs.py rn18 imagenet blocked --sparse-dir models_blocked

# Quantize a 2:4 pruned model
python main_trueobs.py rn18 imagenet quant --wbits 4 --abits 4 --load rn18_24.pth --save rn18_24_4w4a.pth 
```

# Statistics Corrections

```
# Batchnorm tuning
python postproc.py rn18 imagenet rn18_24.pth --bnt

# Statistics correction
python postproc.py rn18 imagenet rn18_24.pth --statcorr --statcorr-samples 1024
```

# Non-Uniform Compression

```
mkdir scores

# Unstructured pruning

# Setup database
mkdir models_unstr
python main_trueobs.py rn18 imagenet unstr --sparse-dir models_unstr
# Compute corresponding losses
python database.py rn18 imagenet unstr loss
# Run DP algorithm to determine per-layer compression targets 
python spdy.py rn18 imagenet 2 unstr --dp 
# Stitch profile, apply batchnorm resetting and compute validation accuracy 
python postproc.py rn18 imagenet rn18_unstr_200x_dp.txt --database unstr --bnt

# Mixed quantization + 2:4 pruning

mkdir models_nm
mkdir models_quant
mkdir models_nm_quant
python main_trueobs.py rn18 imagenet nmprune --save models_nm/rn18_24.pth
python main_trueobs.py rn18 imagenet quant --wbits 8 --abits 8 --save models_quant/rn18_8w8a.pth
python main_trueobs.py rn18 imagenet quant --wbits 4 --abits 4 --save models_quant/rn18_4w4a.pth
python main_trueobs.py rn18 imagenet quant --wbits 8 --abits 8 --load models_nm/rn18_24.pth --save models_nm_quant/rn18_24_8w8a.pth 
python main_trueobs.py rn18 imagenet quant --wbits 4 --abits 4 --load models_nm/rn18_24.pth --save models_nm_quant/rn18_24_4w4a.pth 
python database.py rn18 imagenet mixed loss
python spdy.py rn18 imagenet 8 mixed --dp
python postproc.py rn18 imagenet rn18_mixed_800x_dp.txt --database mixed --bnt
```

# BERT

Before using our BERT integration, please download our [pretrained checkpoints](https://seafile.ist.ac.at/d/c155c45712ad4bcb9341/) and move them to the `bertsquad` folder.
Then you should be able to use most features described above by passing `bertsquad` (or `bertsquad6` for smaller variants) as the model name and `squad` as the dataset name.
The code was tested with `transformers==4.21.2` and `datasets==1.17.0`.

# BibTex

```
@article{frantar2022obc,
  title={{Optimal Brain Compression:} A Framework for Accurate Post-Training Quantization and Pruning},
  author={Frantar, Elias and Singh, Sidak Pal and Alistarh, Dan},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2022}
}
```
