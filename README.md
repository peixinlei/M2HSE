# Complementarity is the King: Multi-modal and Multi-grained Hierarchical Semantic Enhancement Network for Cross-modal Retrieval (M<sup>2</sup>HSE)

PyTorch code for M<sup>2</sup>HSE. *The local-level subenetwork* of our M<sup>2</sup>HSE is built on top of the [VSESC](https://github.com/HuiChen24/MM_SemanticConsistency).

[Xinlei Pei](https://boreas-pxl.github.io/), Zheng Liu, Shanshan Gao, and Yijun Su. "Complementarity is the King: Multi-modal and Multi-grained Hierarchical Semantic Enhancement Network for Cross-modal Retrieval", [Expert Systems with Applications](https://www.sciencedirect.com/journal/expert-systems-with-applications), Accepted. 

## Introduction

We give a demo code of the [Corel 5K](https://github.com/watersink/Corel5K) dataset, including the details of training process for *the global-level subnetwork* and *the local-level subnetwork*.

## Requirements

We recommended the following dependencies.

* Python 3.6
* [PyTorch](http://pytorch.org/) (1.3.1)
* [NumPy](http://www.numpy.org/) (1.19.2)

* Punkt Sentence Tokenizer:

```python
import nltk
nltk.download()
> d punkt
```

## Download data

The raw images and the corrsponding texts can be downloaded from [here](https://github.com/watersink/Corel5K). Note that we performed data cleaning on this dataset and the specific operations are described in the paper. 

Besides, 1) for extracting the fine-grained visual features, the raw images are divided uniformly into 3*3 blocks. 2) we adopt the AlexNet, pre-trained on ImageNet, to extract the CNN features. 3) We upload text data in the ./data/coarse-grained-data/ and ./data/fine-grained-data . Therefore, for data preparation you have the following two options :

1. Download the above raw data and extract the corresponding features according to the strategy we introduced in the paper.
2. Contact us for relevant data. (Email: peixinlei1998@gmail.com)

## Training models

+ **For training *the global-level subnetwork***:
   
  Run `train_global.py`:

    ```bash
    python train_global.py 
        --data_path ./data/coarse-grained-data
        --data_name corel5k_precomp 
        --vocab_path ./vocab 
        --logger_name ./checkpoint/M2HSE/Global/Corel5K 
        --model_name ./checkpoint/M2HSE/Global/Corel5K 
        --num_epochs 100 
        --lr_updata 50 
        --batchsize 100  
        --gamma_1 1 
        --gamma_2 .5 
        --alpha_1 .8 
        --alpha_2 .8
    ```

+ **For training *the local-level subnetwork***:
   
  Run `train_local.py`:

    ```bash
    python train_local.py 
        --data_path ./data/fine-grained-data
        --data_name corel5k_precomp 
        --vocab_path ./vocab 
        --logger_name ./checkpoint/M2HSE/Local/Corel5K 
        --model_name ./checkpoint/M2HSE/Local/Corel5K 
        --num_epochs 100 
        --lr_updata 50 
        --batchsize 100  
        --gamma_1 1 
        --gamma_2 .5 
        --beta_1 .4 
        --beta_2 .4
    ```

## Reference

Stay tuned. :)

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
