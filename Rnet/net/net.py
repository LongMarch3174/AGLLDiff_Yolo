import torch
import torch.nn as nn
class res_block(nn.Module):
    def __init__(self, in_ch):
        super(res_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=0),
            nn.ReflectionPad2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=0),
            nn.ReflectionPad2d(1)
        )
        self.nonlinear = nn.ReLU(inplace=True)
    def forward(self, x):
        res=self.conv1(x)
        output=res+x
        output=self.nonlinear(output)
        return output


class L_net(nn.Module):
    def __init__(self, num=8):
        super(L_net, self).__init__()

        self.L_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            res_block(num),
            res_block(num),
            res_block(num),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0)
        )
        

    def forward(self, input):
        return torch.sigmoid(self.L_net(input))


class R_net(nn.Module):
    def __init__(self, num=16):
        super(R_net, self).__init__()

        self.R_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            res_block(num),
            res_block(num),
            res_block(num),
            res_block(num),
            res_block(num),
            res_block(num),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0)

        )
    def forward(self, input):
        return torch.sigmoid(self.R_net(input))


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.L_net = L_net(num=32)
        self.R_net = R_net(num=32)
    def forward(self, input):
        L = self.L_net(input)
        R = self.R_net(input)
        return L, R

