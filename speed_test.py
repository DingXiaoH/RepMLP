import torch
from repmlp_resnet import create_RepMLPRes50_Bottleneck_224, create_RepMLPRes50_Light_224, create_RepMLPRes50_Base_224

x = torch.randn(128, 3, 224, 224).cuda()
# btnk = create_RepMLPRes50_Bottleneck_224(deploy=True).cuda()
# btnk.eval()
light = create_RepMLPRes50_Base_224(deploy=True).cuda()
light.eval()

#   warm-up
# y = btnk(x)
# y = btnk(x)
y = light(x)
y = light(x)

with torch.no_grad():
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range(5):
            # y = btnk(x)
            y = light(x)
    print(prof)

