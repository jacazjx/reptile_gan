import torch
import torch.nn as nn
from torch.autograd import Variable
from models import ReptileModel

class Resize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x.view(-1, *self.size)
class Generator(ReptileModel):
    def __init__(self, num_classes, latent_dim=100, img_channels=1, hidden_dim=32, LayerNorm=True):
        super(ReptileModel, self).__init__()
        self.num_classes = num_classes

        self.embedding = nn.Embedding(num_classes, latent_dim)
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
            nn.GroupNorm(num_groups=4, num_channels=self.hidden_dim * 2) if LayerNorm else nn.BatchNorm2d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=2, num_channels=self.hidden_dim) if LayerNorm else nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.hidden_dim, self.img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        embedding = self.embedding(labels)
        x = z * embedding
        # out = self.l1(x)
        # out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # img = self.conv_blocks(out)
        return self.generator(x)

    def clone(self):
        clone = Generator(self.num_classes).cuda()
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone


class Discriminator(ReptileModel):

    def __init__(self, num_classes, img_channels=1, hidden_dim=32, LayerNorm=True):
        super(ReptileModel, self).__init__()
        self.img_channels = img_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, 32 ** 2)
        self.discriminator_front = nn.Sequential(
            nn.Conv2d(self.img_channels, self.hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.LayerNorm([self.hidden_dim, 32, 32]),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([self.hidden_dim, 16, 16]) if LayerNorm else nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([self.hidden_dim * 2, 8, 8]) if LayerNorm else nn.BatchNorm2d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(self.hidden_dim * 2 * 8 * 8, self.hidden_dim * 8 * 8)
        )

        self.discriminator_backbone = nn.Sequential(
            nn.Linear(self.hidden_dim * 8 * 8, self.hidden_dim * 4 * 4),
            nn.LayerNorm(self.hidden_dim * 4 * 4) if LayerNorm else nn.BatchNorm1d(self.hidden_dim * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_dim * 4 * 4, 1),
        )

        # self.feature_extractor = nn.Linear(1024 + 32 ** 2, 256, bias=False)

    def forward(self, img, labels):
        embedding = self.embedding(labels).view(labels.shape[0], 1, 32, 32)
        x = img * embedding
        hidden = self.discriminator_front(x)
        out = self.discriminator_backbone(hidden)
        return out.view(-1, 1)#, self.feature_extractor(torch.cat([hidden, embedding.view(img.shape[0], -1)], dim=-1))

    def clone(self):
        clone = Discriminator(self.num_classes).cuda()
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone



if __name__ == '__main__':
    G = Generator(100, 64, 32)
    D = Discriminator(64, 32)
    z = Variable(torch.zeros(5, 100))
    l = torch.arange(5)
    x = G(z, l)
    y = D(x, l)
    print( 'x', x.size())
    print( 'y', y.size())
