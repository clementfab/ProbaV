import torch
import torch.nn as nn
from math import sqrt
    
#Super resolution model based on a FSRCNN architecture (http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN)
class SRModel(torch.nn.Module):
    def __init__(self, d=64, s=16, m=8):
        super(SRModel, self).__init__()

        self.first_part = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU())


        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        # Non-linear Mapping
        for _ in range(m):
            self.layers.append(
                nn.Sequential(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1),
                              nn.PReLU()))
        # # Expanding
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.mid_part = torch.nn.Sequential(*self.layers)


        # Upscaling through deconvolution
        self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=1, kernel_size=9, stride=3, padding=3)


    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, sqrt(2 / m.out_channels / m.kernel_size[0] / m.kernel_size[0]))  # MSRA
                if m.bias is not None:
                    m.bias.data.zero_()