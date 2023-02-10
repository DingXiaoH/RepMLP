# RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality (PyTorch) (CVPR-2022)

Accepted to CVPR-2022!

The latest version: https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_RepMLPNet_Hierarchical_Vision_MLP_With_Re-Parameterized_Locality_CVPR_2022_paper.pdf

Compared to the old version, we no longer use RepMLP Block as a plug-in component in traditional ConvNets. Instead, we build an MLP architecture with RepMLP Block with a hierarchical design. RepMLPNet shows favorable performance, compared to the other vision MLP models including MLP-Mixer, ResMLP, gMLP, S2-MLP, etc. 

Of course, you may also use it in your model as a building block.

The overlap between the two versions is the Structural Re-parameterization method (Localtiy Injection) that equivalently merges conv into FC. The architectural designs presented in the latest version significantly differ from the old version (ResNet-50 + RepMLP). 

Citation:

    @inproceedings{ding2022repmlpnet,
    title={Repmlpnet: Hierarchical vision mlp with re-parameterized locality},
    author={Ding, Xiaohan and Chen, Honghao and Zhang, Xiangyu and Han, Jungong and Ding, Guiguang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={578--587},
    year={2022}
    }

Old version: https://arxiv.org/abs/2105.01883

## Code of RepMLP and Locality Injection

Please check ```repmlpnet.py``` for the definition of our models.

To conduct Locality Injection, just call ```locality_injection()``` of a RepMLPNet. It is implemented by calling ```local_inject``` of every ```RepMLPBlock```. We show an example in ```convert.py```.

We also show an example of checking the equivalence of Locality Injection:
```
python repmlpnet.py
```

If you want to use RepMLP as a building block in your model, just check the definition of ```RepMLPBlock``` in  ```repmlpnet.py```. For the conversion, just **call local_inject of every RepMLPBlock**:
```
        for m in your_model.modules():
            if hasattr(m, 'local_inject'):
                m.local_inject()
```


## Pre-trained models

We have released the models **reproduced with the training script in this repository**.

| name | resolution |ImageNet-1K acc | #params | FLOPs | ImageNet-1K pretrained model |
|:---:|:---:|:---:|:---:| :---:|:---:|
|RepMLPNet-T224|224x224| 76.62 | 38.3M| 2.8B | [Google Drive](https://drive.google.com/file/d/1TgPDD78G-d3h_Y_vNTuqYdSJZHmnju7b/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1vDMl06wVUvkJgC7DOeleNA?pwd=rmlp)|
|RepMLPNet-B224|224x224| 80.32 | 68.2M| 6.7B | [Google Drive](https://drive.google.com/file/d/1GwoUvxGXgzKArlCf0V9kvc9_3GtkGltp/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1MzkKTTzmOdOvtUCuRqwswA?pwd=rmlp)|
|RepMLPNet-T256|256x256| 77.44 | 58.7M| 4.2B | [Google Drive](https://drive.google.com/file/d/1mU0L76x20eAKzkHxEM7J2BvEOLWx_mux/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1pALVZES9lVnjpXf20NxW3Q?pwd=rmlp)|
|RepMLPNet-B256|256x256| 81.03 | 96.5M| 9.6B | [Google Drive](https://drive.google.com/file/d/1W-__gXE7903cX-gOSXEGxNBCJjvqFCiJ/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1GGqIOoVBAd3thaU5LoUtqQ?pwd=rmlp)|
|RepMLPNet-D256|256x256| 80.83 | 86.9M| 8.6B |[Google Drive](https://drive.google.com/file/d/1e5IYac0UHnJq2_lzbuE7J4rahMENX2ha/view?usp=sharing), [Baidu](https://pan.baidu.com/s/12PvSLGepMCCImr--D-RQvw?pwd=rmlp)|
|RepMLPNet-L256|256x256| 81.68 | 117.7M  | 11.5B |[Google Drive](https://drive.google.com/file/d/1SHhNJ6pZax9qMLm8DJZtQ_XfmcmddPYU/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1qz2JBUyYY6JEpnzFIdm-xQ?pwd=rmlp)|


## Test our models and verify the equivalency of Locality Injection

You may test our models with eight GPUs. For example,
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repmlp.py --data-path [imagenet-folder] --arch RepMLPNet-D256 --batch-size 32 --tag test --eval --resume RepMLPNet-D256-train-acc80.828.pth --opts DATA.IMG_SIZE 256
```
If you have no more than one GPUs or you are unfamiliar with PyTorch, you may use a much simpler testing script. It runs in CPU and single-GPU mode. 
```
python test.py [imagenet-folder] train RepMLPNet-D256-train-acc80.828.pth -a RepMLPNet-D256 -r 256
```
Here "train" indicates the training-time architecture.

We showcase the transformation from the training-time model into the inference-time structure with ```test.py``` since this script is short and simple.

**Use case A**: we may convert the weights of a trained RepMLPNet and save the trained weights; when we use it, we build an inference-time RepMLPNet, load the converted weights and test. For example, we save the converted weights to ```RepMLPNet-D256-deploy.pth```.
```
python convert.py RepMLPNet-D256-train-acc80.828.pth RepMLPNet-D256-deploy.pth -a RepMLPNet-D256
python test.py [imagenet-folder] deploy RepMLPNet-D256-deploy.pth -a RepMLPNet-D256 -r 256
```
Here "deploy" indicates building the inference-time architecture.

**Use case B**: we may build a training-time RepMLPNet, load the weights of the trained model, and convert by calling ```RepMLPNet.locality_injection()``` at any time before testing. You may check the equivalency by
```
python test.py [imagenet-folder] check RepMLPNet-D256-train-acc80.828.pth -a RepMLPNet-D256 -r 256
```

## Train from scratch

You may use the training script (based on the script provided by [Swin Transformer](https://github.com/microsoft/Swin-Transformer)) to reproduce our results. For example, you may run
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repmlp.py --arch RepMLPNet-B256 --batch-size 32 --tag my_experiment --opts TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.001 TRAIN.WEIGHT_DECAY 0.1 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.MOMENTUM 0.9 TRAIN.WARMUP_LR 5e-7 TRAIN.MIN_LR 0.0 TRAIN.WARMUP_EPOCHS 10 AUG.PRESET raug15 AUG.MIXUP 0.4 AUG.CUTMIX 1.0 DATA.IMG_SIZE 256
```
so that the log and models will be saved to ```output/RepMLPNet-B256/my_experiment/```.


## FAQs

**Q**: Is the inference-time model's output the _same_ as the training-time model?

**A**: Yes. You can verify that by
```
python repmlpnet.py
```

**Q**: How to use RepMLPNet for other tasks?

**A**: It is better to finetune the training-time model on your datasets. Then you should do the conversion after finetuning and before you deploy the models. For example, say you want to use RepMLPNet and UperNet for semantic segmentation, you should build a UperNet with a training-time RepMLPNet as the backbone, load pre-trained weights into the backbone, and finetune the UperNet on your segmentation dataset. Then you should convert the backbone and keep the other structures. That will be as simple as
```
        for m in your_upernet.modules():
            if hasattr(m, 'local_inject'):
                m.local_inject()
```

Finetuning with a converted RepMLPNet also makes sense, but the performance may be slightly lower.

**Q**: How to quantize a model with RepMLP?

**A1**: Post-training quantization. After training and conversion, you may quantize the converted model with any post-training quantization method. Then you may insert a BN after fc3 and finetune to recover the accuracy just like you quantize and finetune the other models. This is the recommended solution.

**A2**: Quantization-aware training. During the quantization-aware training, instead of constraining the params in a single kernel (e.g., making every param in {-127, -126, .., 126, 127} for int8) for ordinary models, you should constrain the equivalent kernel (get_equivalent_fc3() in repmlpnet.py). 

**Q**: I tried to finetune your model with multiple GPUs but got an error. Why are the names of params like "stages.0..." in the downloaded weights file but sometimes like "module.stages.0..." (shown by my_model.named_parameters()) in my model?

**A**: DistributedDataParallel may prefix "module." to the name of params and cause a mismatch when loading weights by name. The simplest solution is to load the weights (model.load_state_dict(...)) before DistributedDataParallel(model). Otherwise, you may A) insert "module." before the names like this
```
checkpoint = torch.load(...)    # This is just a name-value dict
ckpt = {('module.' + k) : v for k, v in checkpoint.items()}
model.load_state_dict(ckpt)
```
or load the weights into model.module
```
checkpoint = torch.load(...)    # This is just a name-value dict
model.module.load_state_dict(ckpt)
```

**Q**: So a RepMLP Block derives the equivalent big fc kernel before each forwarding to save computations?

**A**: No! More precisely, we do the conversion only once right after training. Then the training-time model can be discarded, and the resultant model has no conv branches. We only save and use the resultant model.


## Contact

**xiaohding@gmail.com** (The original Tsinghua mailbox dxh17@mails.tsinghua.edu.cn will expire in several months)

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

Homepage: https://dingxiaohan.xyz/

My open-sourced papers and repos: 

The **Structural Re-parameterization Universe**:

1. RepLKNet (CVPR 2022) **Powerful efficient architecture with very large kernels (31x31) and guidelines for using large kernels in model CNNs**\
[Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs](https://arxiv.org/abs/2203.06717)\
[code](https://github.com/DingXiaoH/RepLKNet-pytorch).

2. **RepOptimizer** (ICLR 2023) uses **Gradient Re-parameterization** to train powerful models efficiently. The training-time **RepOpt-VGG** is **as simple as the inference-time**. It also addresses the problem of quantization.\
[Re-parameterizing Your Optimizers rather than Architectures](https://arxiv.org/pdf/2205.15242.pdf)\
[code](https://github.com/DingXiaoH/RepOptimizers).

3. RepVGG (CVPR 2021) **A super simple and powerful VGG-style ConvNet architecture**. Up to **84.16%** ImageNet top-1 accuracy!\
[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)\
[code](https://github.com/DingXiaoH/RepVGG).

4. RepMLP (CVPR 2022) **MLP-style building block and Architecture**\
[RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/abs/2112.11081)\
[code](https://github.com/DingXiaoH/RepMLP).

5. ResRep (ICCV 2021) **State-of-the-art** channel pruning (Res50, 55\% FLOPs reduction, 76.15\% acc)\
[ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_ResRep_Lossless_CNN_Pruning_via_Decoupling_Remembering_and_Forgetting_ICCV_2021_paper.pdf)\
[code](https://github.com/DingXiaoH/ResRep).

6. ACB (ICCV 2019) is a CNN component without any inference-time costs. The first work of our Structural Re-parameterization Universe.\
[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).\
[code](https://github.com/DingXiaoH/ACNet). 

7. DBB (CVPR 2021) is a CNN component with higher performance than ACB and still no inference-time costs. Sometimes I call it ACNet v2 because "DBB" is 2 bits larger than "ACB" in ASCII (lol).\
[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)\
[code](https://github.com/DingXiaoH/DiverseBranchBlock).

**Model compression and acceleration**:

1. (CVPR 2019) Channel pruning: [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html)\
[code](https://github.com/DingXiaoH/Centripetal-SGD)

2. (ICML 2019) Channel pruning: [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html)\
[code](https://github.com/DingXiaoH/AOFP)

3. (NeurIPS 2019) Unstructured pruning: [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf)\
[code](https://github.com/DingXiaoH/GSM-SGD)

