import torch
import torch.nn as nn
import torchvision
from nets.SCU import SCU
from nets.UNet import UNet
from nets.Unet_fca import UNetfca
from nets.USwinT import USwinT
from thop import profile
from torchstat import stat
from Train_one_epoch import train_one_epoch
import Config as config

if __name__ == '__main__':
    # model = torchvision.models.AlexNet()
    config_vit = config.get_CTranS_config()
    model = USwinT(config_vit)
    input = torch.randn(2, 3, 384, 384)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)
    # 打印模型参数
    # for param in model.parameters():
    # print(param)

    # 打印模型名称与shape
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
