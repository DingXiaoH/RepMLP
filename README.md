# RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition (PyTorch)

Paper: https://arxiv.org/abs/2105.01883

Citation:

    @article{ding2021repmlp,
    title={RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition},
    author={Ding, Xiaohan and Zhang, Xiangyu and Han, Jungong and Ding, Guiguang},
    journal={arXiv preprint arXiv:2105.01883},
    year={2021}
    }


# How to use the code

If you want to use RepMLP as a building block in your model, just check repmlp.py. It also shows an example of checking the equivalence between a training-time and an inference-time RepMLP. You can see that by
```
python repmlp.py
```
Just use it like this
```
from repmlp.py import *
your_model = YourModel(...)   # It has RepMLPs somewhere
train(your_model)
deploy_model = repmlp_model_convert(your_model)
test(deploy_model)
```
From ```repmlp_model_convert```, you will see that the conversion is as simple as calling **switch_to_deploy** of every RepMLP. 

The definition of the two block structures (RepMLP Bottleneck and RepMLP Light) are shown in ```repmlp_blocks.py```. The RepMLP-ResNet is defined in ```repmlp_resnet.py```.


# Use our pre-trained models

You may download our pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1eDFunxOQ67MvBBmJ4Bw01TFh2YVNRrg2?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/14tGRpKT_WohX7UBcnWH6Zg) (the access key of Baidu is "rmlp").
```
python test.py [imagenet-folder] train RepMLP-Res50-light-224_train.pth -a RepMLP-Res50-light-224
```
Here ```imagenet-folder``` should contain the "train" and "val" folders. The default input resolution is 224x224. Here "train" indicates the training-time architecture.

You may convert them into the inference-time structure and test again to check the equivalence. For example
```
python convert.py RepMLP-Res50-light-224_train.pth RepMLP-Res50-light-224_deploy.pth -a RepMLP-Res50-light-224
python test.py [imagenet-folder] deploy RepMLP-Res50-light-224_deploy.pth -a RepMLP-Res50-light-224
```
Now "deploy" indicates the inference-time structure (without Local Perceptron).



# Abstract

We propose RepMLP, a multi-layer-perceptron-style neural network building block for image recognition, which is composed of a series of fully-connected (FC) layers. Compared to convolutional layers, FC layers are more efficient, better at modeling the long-range dependencies and positional patterns, but worse at capturing the local structures, hence usually less favored for image recognition. We propose a structural re-parameterization technique that adds local prior into an FC to make it powerful for image recognition. Specifically, we construct convolutional layers inside a RepMLP during training and merge them into the FC for inference. On CIFAR, a simple pure-MLP model shows performance very close to CNN. By inserting RepMLP in traditional CNN, we improve ResNets by 1.8% accuracy on ImageNet, 2.9% for face recognition, and 2.3% mIoU on Cityscapes with lower FLOPs. Our intriguing findings highlight that combining the global representational capacity and positional perception of FC with the local prior of convolution can improve the performance of neural network with faster speed on both the tasks with translation invariance (e.g., semantic segmentation) and those with aligned images and positional patterns (e.g., face recognition).



# FAQs

**Q**: Is the inference-time model's output the _same_ as the training-time model?

**A**: Yes. You can verify that by
```
python repmlp.py
```

**Q**: How to use RepMLP for other tasks?

**A**: It is better to finetune the training-time model on your datasets. Then you should do the conversion after finetuning and before you deploy the models. For example, say you want to use RepMLP-Res50 and PSPNet for semantic segmentation, you should build a PSPNet with a training-time RepMLP-Res50 as the backbone, load pre-trained weights into the backbone, and finetune the PSPNet on your segmentation dataset. Then you should convert the backbone following the code provided in this repo and keep the other task-specific structures (the PSPNet parts, in this case). The pseudo code will be like
```
#   train_backbone = create_xxx(deploy=False)
#   train_backbone.load_state_dict(torch.load(...))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repmlp_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
```

Finetuning with a converted model also makes sense if you insert a BN after fc3, but the performance may be slightly lower.

**Q**: How to quantize a model with RepMLP?

**A1**: Post-training quantization. After training and conversion, you may quantize the converted model with any post-training quantization method. Then you may insert a BN after fc3 and finetune to recover the accuracy just like you quantize and finetune the other models. This is the recommended solution.

**A2**: Quantization-aware training. During the quantization-aware training, instead of constraining the params in a single kernel (e.g., making every param in {-127, -126, .., 126, 127} for int8) for ordinary models, you should constrain the equivalent kernel (get_equivalent_fc1_fc3_params() in repmlp.py). 

**Q**: I tried to finetune your model with multiple GPUs but got an error. Why are the names of params like "stage1.0..." in the downloaded weight file but sometimes like "module.stage1.0..." (shown by nn.Module.named_parameters()) in my model?

**A**: DistributedDataParallel may prefix "module." to the name of params and cause a mismatch when loading weights by name. The simplest solution is to load the weights (model.load_state_dict(...)) before DistributedDataParallel(model). Otherwise, you may insert "module." before the names like this
```
checkpoint = torch.load(...)    # This is just a name-value dict
ckpt = {('module.' + k) : v for k, v in checkpoint.items()}
model.load_state_dict(ckpt)
```

**Q**: So a RepMLP derives the equivalent big fc kernel before each forwarding to save computations?

**A**: No! More precisely, we do the conversion only once right after training. Then the training-time model can be discarded, and the resultant model has no conv branches. We only save and use the resultant model.


## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos: 

The **Structural Re-parameterization Universe**:

1. (preprint, 2021) **A powerful MLP-style CNN building block**\
[RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition](https://arxiv.org/abs/2105.01883)\
[code](https://github.com/DingXiaoH/RepMLP).

2. (CVPR 2021) **A super simple and powerful VGG-style ConvNet architecture**. Up to **83.55%** ImageNet top-1 accuracy!\
[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)\
[code](https://github.com/DingXiaoH/RepVGG).

3. (preprint, 2020) **State-of-the-art** channel pruning\
[Lossless CNN Channel Pruning via Decoupling Remembering and Forgetting](https://arxiv.org/abs/2007.03260)\
[code](https://github.com/DingXiaoH/ResRep).

4. ACB (ICCV 2019) is a CNN component without any inference-time costs. The first work of our Structural Re-parameterization Universe.\
[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).\
[code](https://github.com/DingXiaoH/ACNet). 

5. DBB (CVPR 2021) is a CNN component with higher performance than ACB and still no inference-time costs. Sometimes I call it ACNet v2 because "DBB" is 2 bits larger than "ACB" in ASCII (lol).\
[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)\
[code](https://github.com/DingXiaoH/DiverseBranchBlock).

**Model compression and acceleration**:

1. (CVPR 2019) Channel pruning: [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html)\
[code](https://github.com/DingXiaoH/Centripetal-SGD)

2. (ICML 2019) Channel pruning: [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html)\
[code](https://github.com/DingXiaoH/AOFP)

3. (NeurIPS 2019) Unstructured pruning: [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf)\
[code](https://github.com/DingXiaoH/GSM-SGD)
