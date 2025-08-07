import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  #    16 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 #    16 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), #    32 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 #    32 x 8 x 8
            
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, 8)),

            nn.Upsample((16, 16)),                                              #   32 x 16 x 16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),     #   16 x 32 x 32
            nn.ReLU(),
            nn.Upsample((64, 64)),                                              #   16 x 64 x 64
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),      #   3 x 128 x 128
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x