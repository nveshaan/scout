import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

class encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        return x

class latent(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        pass

class decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3)
        self.deconv3 = nn.ConvTranspose2d(16, 3, 3)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.deconv1(input)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.relu(x)

        return x

class auto_encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc = encoder()
        # self.lat = latent()
        self.dec = decoder()

    def forward(self, input):
        x = tf.resize(input, (256, 256))
        x = tf.normalize(x)

        x = self.enc(x)
        # x = self.lat(x)
        x = self.dec(x)
        return x