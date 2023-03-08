import torch
import torch.nn as nn
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        self.embedding = nn.Embedding(num_classes, latent_dim)
        self.fc = nn.Linear(latent_dim, 128 * (img_size // 4) ** 2)
        self.bn = nn.BatchNorm2d(128)
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        embedding = self.embedding(labels)
        x = torch.mul(z, embedding)
        x = self.fc(x)
        x = x.view(x.shape[0], 128, (self.img_size // 4), (self.img_size // 4))
        x = self.bn(x)
        x = self.conv_transpose(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Discriminator, self).__init__()

        self.num_classes = num_classes
        self.img_size = img_size

        self.embedding = nn.Embedding(num_classes, img_size ** 2)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        embedding = self.embedding(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([img, embedding], dim=1)
        x = self.conv(x)
        return x.view(-1, 1).squeeze(1)

if __name__ == '__main__':
    G = Generator(100, 64, 32)
    D = Discriminator(64, 32)
    z = Variable(torch.zeros(5, 100))
    l = torch.arange(5)
    x = G(z, l)
    y = D(x, l)
    print( 'x', x.size())
    print( 'y', y.size())
