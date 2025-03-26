import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        # First convolutional layer in the block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Batch normalization
        self.relu = nn.ReLU(inplace=True)  # ReLU activation
        # Second convolutional layer in the block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # Batch normalization
        
        self.skip_connection = nn.Sequential()
        # If the input and output dimensions do not match, adjust them with a convolution
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += self.skip_connection(x)  # Add the skip connection
        output = self.relu(output)
        return output

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization
        self.relu = nn.ReLU(inplace=True)  # ReLU activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Max pooling
        
        # Residual layers
        self.layer1 = self._make_layer(Block, 64, 2, stride=1)
        self.layer2 = self._make_layer(Block, 128, 2, stride=2)
        self.layer3 = self._make_layer(Block, 256, 2, stride=2)
        self.layer4 = self._make_layer(Block, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive average pooling
        self.fc = nn.Linear(512, num_classes)  # Fully connected layer for output

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.maxpool(output)
        
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)  # Flatten the output
        return  self.fc(output)
