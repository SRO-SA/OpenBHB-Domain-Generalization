""" 3D brain age model"""
# mean absolute error
# adam optimizer
# learning rate 10^-4
# weight decay 10^-4

from box import Box
from torch import nn
import torch
# torch.autograd.set_detect_anomaly(True)

def conv_blk(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_channel), nn.MaxPool3d(2, stride=2), nn.ReLU()
    ).to('cuda:2')


class BrainCancer(nn.Module):
    def __init__(self, initialization="default", device = 'cuda:2', **kwargs):
        super(BrainCancer, self).__init__()
        self.initialization = initialization
        self.d_out = 1
        self.batch_size = 1
        self.device = device
        self.device2 = device[:-1]+str(int(device[-1])+1)
        self.conv1 = conv_blk(self.batch_size, 32).to(device) #1 ,32
        self.conv2 = conv_blk(32, 64).to(device)
        self.conv3 = conv_blk(64, 128).to(device)
        self.conv4 = conv_blk(128, 256).to(device)
        self.conv5 = conv_blk(256, 256).to(device)
        
        self.batch_norm=False
        self.conv6 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1, stride=1),
            nn.InstanceNorm3d(64), nn.ReLU(),
            nn.AvgPool3d(kernel_size=(2, 3, 2))
        ).to(self.device)
            
        self.drop = nn.Identity().to(device)  # nn.Dropout3d(p=0.5)
        
        self.output = nn.Conv3d(64, self.batch_size, kernel_size=1, stride=1).to(device)

        self.init_weights()
        
        # self.layer1 = nn.Sequential(Squeezer().to(device), self.conv1, self.conv2, self.conv3).to('cpu').to(self.device)
        # self.layer2 = nn.Sequential(self.conv4, self.conv5, self.conv6).to('cpu').to(self.device2)        
        # self.conv_features = SequentialOnMultipleDivices(self.layer1,
        #                                                     self.layer2,
        #                                                     self.device,
        #                                                     self.device2)

        self.conv_features = nn.Sequential(Squeezer().to(device), self.conv1, self.conv2, self.conv3,
                                        self.conv4, self.conv5, self.conv6)
        
        self.dense_features = self.drop
        
        self.featurizer = nn.Sequential(self.conv_features, self.dense_features)
        
        self.classifier =  nn.Sequential(self.drop, self.output)
        self.double()


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
        print("data type of x in model (beign)", x.dtype)
        # print("dsdsds: ", x.shape) # dsdsds:  torch.Size([32, 1, 1, 121, 145, 121])
        # dsdsds:  torch.Size([1, 1, 1, 121, 145, 121]) if batch_size == 1
        if torch.equal(torch.Tensor(list(x.shape)), torch.Tensor([1, 1, 1, 1, 121, 145, 121])):
            print("in if")
            x = torch.squeeze(x, dim = 0)
        x = x.to(self.device)
        # x = x.view(-1, 1, 121, 145, 121).to(self.device)
        # print(self.device2)
        # print("in forward: ", x.shape)
        # print("y_pred in model: ", x)
        x = self.featurizer(x)
        x = self.classifier(x)
        
        # x = self.conv1(x)
        # # print("in forward after conv1: ", x.shape)   
        # # x = torch.squeeze(x)     
        # # print("in forward after conv1 squeezed: ", x.shape)   
        # # print("y_pred in model: ", x)
        # x = self.conv2(x)
        # # print("conv2 shape: ", x.shape)
        # x = self.conv3(x)
        # # print('before: ', x.count_nonzero() == 0)
        # x = x.to('cpu').to(self.device2)
        # # print('after change: ', x.count_nonzero() == 0)
        # # print("conv3 shape: ", x.shape)
        # x = self.conv4(x)
        # # print("after conv4: ", x.count_nonzero() == 0)
        # # print("y_pred in model: ", x)
        # # print("conv4 shape: ", x.shape)
        # x = self.conv5(x)
        # # print("conv5 shape: ", x.shape)
        # x = self.conv6(x)
        # # print("conv6 shape: ", x.shape)
        # x = self.drop(x.to('cpu').to(self.device))
        # # print("y_pred in model: ", x)
        # # print("drop shape: ", x.shape)
        # x = self.output(x)
        # print("y_pred in model: ", x)
        x = torch.squeeze(x, dim=[1,2,3])
        # print("y_pred in model: ", x)
        # print("final shape: ", x.shape)
        # print("data type of x in model ", x.dtype)
        return x


def get_arch(*args, **kwargs):
    return {"net": BrainCancer(**kwargs)}


class Squeezer(torch.nn.Module):
    def forward(self, x):
        # print("shape of x in squeezer ", x.shape)
        x = x.double()
        x = x.view(-1, 1, 121, 145, 121)
        # print("shape of x in squeezer ", x.shape)
        # x = torch.squeeze(x, dim=0)
        return x
