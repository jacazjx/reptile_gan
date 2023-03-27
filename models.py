import copy

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import SerializationTool


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

class Resize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x.view(-1, *self.size)


class Generator(ReptileModel):
    def __init__(self, num_classes, latent_dim=100, img_channels=1, hidden_dim=16, LayerNorm=False, Condition=False):
        super(ReptileModel, self).__init__()
        self.num_classes = num_classes
        self.condition = Condition
        self.embedding = nn.Embedding(num_classes, latent_dim) if Condition else None
        self.noise_dim = latent_dim
        self.img_channels = img_channels
        self.hidden_dim = hidden_dim
        self.generator = nn.Sequential(
            nn.Linear(self.noise_dim, self.hidden_dim * 4 * 4 * 4, bias=False),
            nn.LayerNorm(self.hidden_dim * 4 * 4 * 4) if LayerNorm else nn.BatchNorm1d(self.hidden_dim * 4 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            Resize((self.hidden_dim * 4, 4, 4)),
            nn.ConvTranspose2d(self.hidden_dim * 4, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.GroupNorm(num_groups=self.hidden_dim // 4,
                         num_channels=self.hidden_dim * 2) if LayerNorm else nn.BatchNorm2d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=self.hidden_dim // 8,
                         num_channels=self.hidden_dim) if LayerNorm else nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.hidden_dim, self.img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels=None):
        if self.condition:
            embedding = self.embedding(labels)
            x = z + embedding
        else:
            x = z
        return self.generator(x)

    def clone(self):
        clone = Generator(self.num_classes, img_channels=self.img_channels, hidden_dim=self.hidden_dim,
                          LayerNorm=self.LayerNorm, Condition=self.Condition).cuda()
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone


class Discriminator(ReptileModel):

    def __init__(self, num_classes, img_channels=1, hidden_dim=32, LayerNorm=False, Condition=False,
                 FeatureExtraction=False):
        super(ReptileModel, self).__init__()
        self.img_channels = img_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.LayerNorm = LayerNorm
        self.FeatureExtraction = FeatureExtraction
        self.Condition = Condition
        self.embedding = nn.Embedding(num_classes, 32 ** 2) if Condition else None
        self.feature_extraction = nn.Flatten() if FeatureExtraction else None
        self.discriminator_front = nn.Sequential(
            nn.Conv2d(self.img_channels, self.hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([self.hidden_dim, 16, 16]) if LayerNorm else nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([self.hidden_dim * 2, 8, 8]) if LayerNorm else nn.BatchNorm2d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([self.hidden_dim * 4, 4, 4]) if LayerNorm else nn.BatchNorm2d(self.hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.discriminator_backbone = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten()
        )

    def forward(self, img, labels=None):
        if self.Condition:
            embedding = self.embedding(labels).view(labels.shape[0], 1, 32, 32)
            x = img + embedding
        else:
            x = img
        hidden = self.discriminator_front(x)
        out = self.discriminator_backbone(hidden)

        return out if self.FeatureExtraction else out, self.feature_extractor(hidden)

    def clone(self):
        clone = Discriminator(self.num_classes, img_channels=self.img_channels, hidden_dim=self.hidden_dim,
                              LayerNorm=self.LayerNorm, Condition=self.Condition,
                              FeatureExtraction=self.FeatureExtraction).cuda()
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone


# class Generator(ReptileModel):
#     def __init__(self, num_classes, input_size=100,  image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
#         super(Generator, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_classes = num_classes
#         self.blocks = blocks
#         self.height = height
#         self.length = length
#         self.mult = 2**blocks
#         self.embedding = nn.Embedding(num_classes, input_size)
#         self.initial_linear = nn.Linear(input_size, hidden_size * self.mult * height//self.mult * length//self.mult)
#         self.initial_activ = nn.PReLU(hidden_size * self.mult * height//self.mult * length//self.mult)
#         self.initial_norm = nn.LayerNorm(hidden_size * self.mult * height//self.mult * length//self.mult)
#
#         self.convs = nn.ModuleList([nn.Conv2d(hidden_size * 2 **(blocks - i), hidden_size * 2**(blocks - i - 1), (5, 5), padding=(2, 2)) for i in range(blocks)])
#         self.activ = nn.ModuleList([nn.PReLU(hidden_size * 2**(blocks - i - 1)) for i in range(blocks)])
#         self.norm = nn.ModuleList([nn.LayerNorm(
#             [hidden_size * 2 ** (blocks - i - 1), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
#                        range(blocks)])
#
#         self.final_conv = nn.Conv2d(hidden_size, image_channels, (5, 5), padding=(2, 2))
#         self.final_activ = nn.Tanh()
#
#     def forward(self, inputs, labels):
#         embedding = self.embedding(labels)
#         x = torch.mul(inputs, embedding)
#         x = self.initial_linear(x)
#         x = self.initial_activ(x)
#         x = self.initial_norm(x)
#         x = x.view(x.shape[0], self.hidden_size * self.mult, self.height//self.mult, self.length//self.mult)
#
#         for i in range(self.blocks):
#             x = self.convs[i](x)
#             x = self.activ[i](x)
#             x = self.norm[i](x)
#             x = F.upsample(x, scale_factor=2)
#
#         x = self.final_conv(x)
#         x = self.final_activ(x)
#         return x
#
#     def clone(self):
#         clone = Generator(self.num_classes).cuda()
#         clone.load_state_dict(self.state_dict())
#         if self.is_cuda():
#             clone.cuda()
#         return clone

# class Discriminator(ReptileModel):
#     def __init__(self, num_classes, image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
#         super(Discriminator, self).__init__()
#         self.hidden_size = hidden_size
#         self.blocks = blocks
#         self.height = height
#         self.length = length
#         self.num_classes = num_classes
#         self.embedding = nn.Embedding(num_classes, height * length)
#         self.initial_conv = nn.Conv2d(image_channels+1, hidden_size, (5, 5), padding=(2, 2))
#         self.initial_norm = nn.LayerNorm([hidden_size, height, length])
#         self.initial_activ = nn.PReLU(hidden_size)
#
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** (i + 1), (5, 5), padding=(2, 2)) for
#              i in range(blocks)])
#         self.norm = nn.ModuleList([nn.LayerNorm(
#             [hidden_size * 2 ** (i + 1), height // (2 ** i), length // (2 ** i)]) for i
#                                    in range(blocks)])
#         self.activ = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (i + 1)) for i in range(blocks)])
#
#         self.final_linear = nn.Linear(hidden_size * 2 ** blocks * height//(2**blocks) * length//(2**blocks), 1)
#
#     def forward(self, inputs, labels):
#         embedding = self.embedding(labels).view(labels.shape[0], 1, self.height, self.length)
#         x = torch.cat([inputs, embedding], dim=1)
#         x = self.initial_conv(x)
#         x = self.initial_norm(x)
#         x = self.initial_activ(x)
#         h = 0
#         for i in range(self.blocks):
#             x = self.convs[i](x)
#             if i == self.blocks // 2:
#                 h = x
#             x = self.norm[i](x)
#             x = self.activ[i](x)
#             x = F.avg_pool2d(x, kernel_size=(2, 2))
#
#         x = x.view(x.shape[0], -1)
#         x = self.final_linear(x)
#         return x, h
#
#     def clone(self):
#         clone = Discriminator(self.num_classes).cuda()
#         clone.load_state_dict(self.state_dict())
#         if self.is_cuda():
#             clone.cuda()
#         return clone

if __name__ == '__main__':
    G = Generator(10)
    D1 = Discriminator(10)
    D2 = Discriminator(10)
    z = Variable(torch.zeros(5, 100))
    b = torch.arange(5)
    r = torch.tensor([0 for i in range(5)], dtype=torch.float32).view(-1, 1)

    x = G(z, b)
    y1 = D1(x, b)
    y2 = D2(x, b)
    loss = nn.BCEWithLogitsLoss()
    loss_p = (loss(y1, r) + loss(y2, r)) * 0.5
    loss_p.backward(retain_graph=True)
    print('g1', SerializationTool.serialize_model(G, "grad").norm())
    G.zero_grad()
    D1.zero_grad()
    D2.zero_grad()
    del y1, y2

    xx = x.clone().detach()
    xx1 = copy.deepcopy(xx).requires_grad_(True)
    xx2 = copy.deepcopy(xx).requires_grad_(True)
    y1 = D1(xx1, b)
    loss_md_1 = loss(y1, r)
    loss_md_1.backward()

    y2 = D1(xx2, b)
    loss_md_2 = loss(y2, r)
    loss_md_2.backward()

    x.backward(xx1.grad, retain_graph=True)
    t1 = SerializationTool.serialize_model(G, "grad")
    G.zero_grad()
    x.backward(xx2.grad, retain_graph=True)
    t2 = SerializationTool.serialize_model(G, "grad")
    print('g2', ((t1 + t2) * 0.5).norm())
