# ResNet18 pipeline implementation

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import BatchNorm2d
from torch.nn import AdaptiveAvgPool2d
from torch.nn import Dropout
import torch.nn as nn
from torch import rand
from torch import Tensor
from torch import device
from torch import cuda
from typing import Type
from torch import flatten

# The most important part of a Residual Neural Network are the residual blocks, so let's define them:
# Reference for the residual block logic https://arxiv.org/pdf/1512.03385v1.pdf
class ResBlock(Module):
    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                stride: int = 1, 
                expansion: int = 1, 
                downsample: Module = None)-> None:
        super(ResBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        
        self.conv1 = Conv2d(in_channels,
                            out_channels,
                            kernel_size = 3,
                            stride = stride,
                            padding = 1, 
                            bias = False)
        self.bn1 = BatchNorm2d(out_channels)                    
        self.relu = ReLU(inplace = True)

        self.conv2 = Conv2d(out_channels,
                            out_channels*expansion,
                            kernel_size=3,
                            padding = 1,
                            bias = False)
        self.bn2 = BatchNorm2d(out_channels*expansion)    

    def forward(self,x: Tensor) -> Tensor:
        identity = x

        out = self.relu((self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out    


class ResNet(Module):
    def __init__(self, 
                img_channels: int,
                num_layers: int, 
                block: Type[ResBlock],
                #dropout_rate: float,
                num_classes: int
                ) -> None: 

        # The number of classes correspond to the number of zernike polynomials 
        # used to generate training data  
            
        # Call the parent constructor
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of Residual Blocks 
            # used to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
            
        # Initialize Layer 0 (conv1 + Max pooling)
        self.layer0 = Sequential(
            Conv2d(in_channels = img_channels, 
                out_channels = self.in_channels, 
                kernel_size= 7, 
                stride = 2, 
                padding = 3,
                bias = False),
            BatchNorm2d(self.in_channels),
            ReLU(inplace= True),
            MaxPool2d(kernel_size=3, 
                      stride = 2, 
                      padding= 1)
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
    
        # Global Average Pooling - > gap + Fully Connected layer -> fc
        self.gap = AdaptiveAvgPool2d((1,1))
        self.fc1 = Linear(512*self.expansion, 256)
        self.fc2 = Linear(256, num_classes)

        # Dropout layer
        #self.drop = Dropout(dropout_rate)

    def _make_layer(
        self,
        block: Type[ResBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = Sequential(
                Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size = 1,
                    stride = stride,
                    bias = False
                ),
                BatchNorm2d(out_channels*self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            ))
        self.in_channels = out_channels*self.expansion

        for i in range(1,blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion = self.expansion
            ))     
        return Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)

        x = self.gap(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        #x = self.drop(x)
        x = self.fc2(x)

        return x

if __name__ == '__main__':
    device1 = device('cuda:0' if cuda.is_available() else 'cpu')
    tensor = rand([1, 1, 369, 369]).to(device1)
    model = ResNet(img_channels=3, num_layers=18, block=ResBlock, num_classes=4).to(device1)
    folder = 'C:\\Users\\thali\\Resnet18\\outputs'
    nn.save(model.state_dict(), folder)
    #, dropout_rate= 0.2
    print(model)
    
    # Total parameters and trainable parameters.da
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    output = model(tensor)
    folder = 'C:\\Users\\thali\\Resnet18\\outputs'
    nn.save(output.state_dict(), folder)
