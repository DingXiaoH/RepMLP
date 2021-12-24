# RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality (PyTorch)

The latest version: https://arxiv.org/abs/2112.11081

Compared to the old version, we no longer use RepMLP Block as a plug-in component in traditional ConvNets. Instead, we build an MLP architecture with RepMLP Block with a hierarchical design. RepMLPNet shows favorable performance, compared to the other vision MLP models including MLP-Mixer, ResMLP, gMLP, S2-MLP, etc.

The overlap between the two versions is the Structural Re-parameterization method (Localtiy Injection) that equivalently merges conv into FC. The architectural designs presented in the latest version significantly differ from the old version (ResNet-50 + RepMLP). 

Old version: https://arxiv.org/abs/2105.01883

Citation (will be updated in 2 days):

    @article{ding2021repmlp,
    title={RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition},
    author={Ding, Xiaohan and Xia, Chunlong and Zhang, Xiangyu and Chu, Xiaojie and Han, Jungong and Ding, Guiguang},
    journal={arXiv preprint arXiv:2105.01883},
    year={2021}
    }


# How to use the code

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


# Use our pre-trained models

You may download our pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1eDFunxOQ67MvBBmJ4Bw01TFh2YVNRrg2?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/14tGRpKT_WohX7UBcnWH6Zg) (the access key of Baidu is "rmlp").
```
python test.py [imagenet-folder] train RepMLPNet-B256-train-acc8111.pth -a RepMLPNet-B256 -r 256
```
Here ```imagenet-folder``` should contain the "train" and "val" folders. The default input resolution is 224x224. Here "train" indicates the training-time architecture.

You may convert the training-time model into the inference-time structure via Locality Injection and test again to verify the equivalence. For example
```
python convert.py RepMLPNet-B256-train-acc8111.pth RepMLPNet-B256-deploy-acc8111.pth -a RepMLPNet-B256
python test.py [imagenet-folder] deploy RepMLPNet-B256-deploy-acc8111.pth -a RepMLPNet-B256 -r 256
```
Now "deploy" indicates the inference-time structure (without Local Perceptron).


# Abstract

Compared to convolutional layers, fully-connected (FC) layers are better at modeling the long-range dependencies but worse at capturing the local patterns, hence usually less favored for image recognition. In this paper, we propose a methodology, Locality Injection, to incorporate local priors into an FC layer via merging the trained parameters of a parallel conv kernel into the FC kernel. Locality Injection can be viewed as a novel Structural Re-parameterization method since it equivalently converts the structures via transforming the parameters. Based on that, we propose a multi-layer-perceptron (MLP) block named RepMLP Block, which uses three FC layers to extract features, and a novel architecture named RepMLPNet. The hierarchical design distinguishes RepMLPNet from the other concurrently proposed vision MLPs. As it produces feature maps of different levels, it qualifies as a backbone model for downstream tasks like semantic segmentation. Our results reveal that 1) Locality Injection is a general methodology for MLP models; 2) RepMLPNet has favorable accuracy-efficiency trade-off compared to the other MLPs; 3) RepMLPNet is the first MLP that seamlessly transfer to Cityscapes semantic segmentation.

# Results

![image](https://user-images.githubusercontent.com/55726946/147339507-71dcdb18-95ea-420f-b80b-310e83d0c301.png)

We have released the weights of RepMLPNet-B224 and B256. The accuracies are slightly higher than those reported in the paper.

Uploading the other weights.

# TODO

Release more model weights (in several days, I think)

Training code (based on the code of Swin and DeiT)

# FAQs

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
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos: 

The **Structural Re-parameterization Universe**:

1. RepMLP (preprint, 2021) **MLP-style building block and Architecture**\
[RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/abs/2112.11081)\
[code](https://github.com/DingXiaoH/RepMLP).

2. RepVGG (CVPR 2021) **A super simple and powerful VGG-style ConvNet architecture**. Up to **84.16%** ImageNet top-1 accuracy!\
[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)\
[code](https://github.com/DingXiaoH/RepVGG).

3. ResRep (ICCV 2021) **State-of-the-art** channel pruning (Res50, 55\% FLOPs reduction, 76.15\% acc)\
[ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_ResRep_Lossless_CNN_Pruning_via_Decoupling_Remembering_and_Forgetting_ICCV_2021_paper.pdf)\
[code](https://github.com/DingXiaoH/ResRep).

4. ACB (ICCV 2019) is a CNN component without any inference-time costs. The first work of our Structural Re-parameterization Universe.\
[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).\
[code](https://github.com/DingXiaoH/ACNet). 

5. DBB (CVPR 2021) is a CNN component with higher performance than ACB and still no inference-time costs. Sometimes I call it ACNet v2 because "DBB" is 2 bits larger than "ACB" in ASCII (lol).\
[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)\
[code](https://github.com/DingXiaoH/DiverseBranchBlock).

6. COMING SOON

7. COMING SOON

**Model compression and acceleration**:

1. (CVPR 2019) Channel pruning: [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html)\
[code](https://github.com/DingXiaoH/Centripetal-SGD)

2. (ICML 2019) Channel pruning: [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html)\
[code](https://github.com/DingXiaoH/AOFP)

3. (NeurIPS 2019) Unstructured pruning: [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf)\
[code](https://github.com/DingXiaoH/GSM-SGD)
