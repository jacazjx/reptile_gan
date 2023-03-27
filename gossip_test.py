import os
import torch
import torchvision
from matplotlib import pyplot as plt

from models import Generator, Discriminator
log_dir = "./checkpoint/1+lsgan+mnist+kd+20+20/"
id_string = "Client_0"
generator = Generator(10, LayerNorm=False)
discriminator = Discriminator(10, LayerNorm=False)

def load_checkpoint():
    checkpoint = torch.load(f"{log_dir}{id_string}")
    generator.load_state_dict(checkpoint["model"]["G"])
    discriminator.load_state_dict(checkpoint["model"]["D"])
    eps = checkpoint["eps"]
load_checkpoint()
generator.eval()
labels = torch.arange(10).view(-1, 1).expand(10, 10).reshape(100)
z = torch.randn(100, 100)

x_g = generator(z, labels)

torchvision.utils.save_image(x_g.detach(), f"gossip.png",
                                 nrow=10, normalize=True)


