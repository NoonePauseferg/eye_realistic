#SimGAN
#https://arxiv.org/pdf/1612.07828.pdf
#https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html
#https://github.com/ajdillhoff/simgan-pytorch/blob/master/model/models.py

# X_train.shape,     Y_train.shape,    X_test.shape,      Y_test.shap
#(40000, 35, 55, 1) (3200, 35, 55, 3) (10000, 35, 55, 1) (800, 35, 55, 3)


import torch
from torch import nn

class ResnetBlock_(nn.Module):
    def __init__(self, 
                input_features, 
                n_feature_maps,
                kernel_size,
                stride,
                ind):

        super().__init__()
        padding=2**ind*(kernel_size-1)//2
        self.conv1 = nn.Conv2d(input_features,
                               n_feature_maps, 
                               kernel_size, 
                               padding=padding, 
                               stride=stride,
                               dilation=2**ind)

        self.conv2 = nn.Conv2d(n_feature_maps, 
                               n_feature_maps, 
                               kernel_size, 
                               padding=padding, 
                               stride=stride,
                               dilation=2**ind)

        self.l_relu = nn.LeakyReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(n_feature_maps)
        self.norm2 = nn.BatchNorm2d(n_feature_maps)

    def forward(self, x):
        prev = x
        x = self.l_relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.l_relu(prev + x)
        return x

class Refiner_(nn.Module):
    def __init__(self,
                 num_blocks,
                 in_features=1,
                 num_features=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, num_features, kernel_size=3, padding=1)
        self.l_relu = nn.LeakyReLU(inplace=True)

        blocks = [ResnetBlock_(input_features=num_features, 
                               n_feature_maps=num_features, 
                               kernel_size=3, 
                               stride=1, 
                               ind=i) for i in range(num_blocks)]
        
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features, in_features, kernel_size=1)

    def forward(self, x):
        x = self.l_relu(self.conv1(x))
        x = self.blocks(x)
        x = self.conv2(x)
        return x

class Discriminator_(nn.Module):
    def __init__(self, in_features=1):
        super().__init__()
        self.l_relu = nn.LeakyReLU(inplace=True)
        self.conv0 = nn.Conv2d(3, 1, kernel_size=1)
        self.conv1 = nn.Conv2d(in_features, 96, kernel_size=3, stride=2)
        self.norm1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 64, kernel_size=3, stride=2)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool =  nn.MaxPool2d(3, stride=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.norm4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        #receptive field same to refiner

    def forward(self, x):
        if x.shape[1] != 1:
            x = self.conv0(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.l_relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.l_relu(x)

        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.l_relu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.l_relu(x)
        x = self.conv5(x)
        x = self.l_relu(x)
        return x.reshape(x.shape[0], 2, -1)




if __name__ == "__main__":
    rf = Refiner_(4)
    ds = Discriminator_()
    input_batch = torch.Tensor(256, 1, 35, 55).normal_()
    # rf(input_batch)
    ds(input_batch)
