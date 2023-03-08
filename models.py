import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class ReptileModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class OmniglotModel(ReptileModel):
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes):
        ReptileModel.__init__(self)

        self.num_classes = num_classes

        self.conv = nn.Sequential(
            # 28 x 28 - 1
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 14 x 14 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 7 x 7 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 4 x 4 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 2 x 2 - 64
        )

        self.classifier = nn.Sequential(
            # 2 x 2 x 64 = 256
            nn.Linear(256, num_classes),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        out = x.view(-1, 1, 28, 28)
        out = self.conv(out)
        out = out.view(len(out), -1)
        out = self.classifier(out)
        return out

    def predict(self, prob):
        __, argmax = prob.max(1)
        return argmax

    def clone(self):
        clone = OmniglotModel(self.num_classes)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone


class Generator(ReptileModel):
    def __init__(self, ims=(28, 28)):
        super(ReptileModel, self).__init__()

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LayerNorm([128, 16, 16], 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.LayerNorm([64, 32, 32], 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def clone(self):
        clone = Generator().cuda()
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone


class Discriminator(ReptileModel):

    def __init__(self, ims=None):
        super(ReptileModel, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True, HW=8):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.LayerNorm([out_filters, HW, HW], 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32, HW=8),
            *discriminator_block(32, 64, HW=4),
            *discriminator_block(64, 128, HW=2),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

    def clone(self):
        clone = Discriminator().cuda()
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

if __name__ == '__main__':
    G = Generator()
    D = Discriminator()
    z = Variable(torch.zeros(5, 100))
    x = G(z)
    y = D(x)
    print( 'x', x.size())
    print( 'y', y.size())

