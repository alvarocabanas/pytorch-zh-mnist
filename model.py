import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ConstantPad2d(1, 0.)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        self.mlp = nn.Sequential(
            nn.Linear(4608, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.LogSoftmax(dim=1),
        )
    
    def forward(self, x):
        x = self.pad(x)
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))
        # Obtain the parameters of the tensor in terms of:
        # batch size, number of channels, "height", "width"
        bsz, nch, height, width = x.shape

        print(f"Output shape: {x.shape}")
        x = x.reshape(bsz, -1)
        print(f"Output shape: {x.shape}")
        return self.mlp(x)
