
# Post-Training Sparsity-Aware Quantization

This repository is the official implementation of [Post-Training Sparsity-Aware Quantization](https://arxiv.org/abs/2030.12345). 

## Requirements

Install PyTorch. Specifically, we use version 1.5.1 with CUDA 10.1.
```pytorch
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
Pick PyTorch 1.5.1 with the appropriate CUDA version from the [official PyTorch website](https://pytorch.org/).  
Then, install the other packages and our custom CUDA package:
```setup
pip install -r requirements.txt
cd cu_gemm_2x48
python ./setup install
```
The ImageNet path, as well as the seeds used to achieve the paper's results, are configured in `Config.py`.  
Throughout this work, we used Ubuntu 18.04, Python 3.6.9, and NVIDIA TITAN V GPU.  

## 8-bit Model Quantization

SPARQ operates on top 8-bit models.
To quantize the models, execute the following command:

```quantize
python ./main.py -a resnet18_imagenet --action QUANTIZE --x_bits 8 --w_bits 8
```
We support the following models: `resnet18_imagenet`, `resnet34_imagenet`, `resnet50_imagenet`, `resnet101_imagenet`, `googlenet_imagenet`, `inception_imagenet`, `densenet_imagenet`.

