""" 3D brain age model"""
# mean absolute error
# adam optimizer
# learning rate 10^-4
# weight decay 10^-4

from box import Box
from torch import nn
import torch


def conv_blk(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_channel), nn.MaxPool3d(2, stride=2), nn.ReLU()
    )


class Model(nn.Module):
    def __init__(self, initialization="default", **kwargs):
        super(Model, self).__init__()
        self.initialization = initialization
        self.conv1 = conv_blk(1, 32) #1 ,32
        self.conv2 = conv_blk(32, 64)
        self.conv3 = conv_blk(64, 128)
        self.conv4 = conv_blk(128, 256)
        self.conv5 = conv_blk(256, 256)

        self.conv6 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1, stride=1),
            nn.InstanceNorm3d(64), nn.ReLU(),
            nn.AvgPool3d(kernel_size=(2, 3, 2))
        )

        self.drop = nn.Identity()  # nn.Dropout3d(p=0.5)

        self.output = nn.Conv3d(64, 1, kernel_size=1, stride=1)

        self.init_weights()

    def init_weights(self):
        if self.initialization == "custom":
            for k, m in self.named_modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out",
                        nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.output.bias, 62.68)

    def forward(self, x):
        #print("dsdsds: ", x.shape)
        x = torch.squeeze(x, dim = 0)
        # print("in forward: ", x.shape)
        x = self.conv1(x)
        # print("in forward after conv1: ", x.shape)   
        # x = torch.squeeze(x)     
        # print("in forward after conv1 squeezed: ", x.shape)   

        x = self.conv2(x)
        # print("conv2 shape: ", x.shape)
        x = self.conv3(x)
        # print("conv3 shape: ", x.shape)
        x = self.conv4(x)
        # print("conv4 shape: ", x.shape)
        x = self.conv5(x)
        # print("conv5 shape: ", x.shape)
        x = self.conv6(x)
        # print("conv6 shape: ", x.shape)
        x = self.drop(x)
        # print("drop shape: ", x.shape)
        x = self.output(x)
        x = torch.squeeze(x, dim=[1,2,3,4])
        # print("final shape: ", x.shape)
        return Box({"y_pred": x})


def get_arch(*args, **kwargs):
    return {"net": Model(**kwargs)}
