import argparse
import math
import os

from torch.autograd import Variable

from datasets.EMNIST import EMNIST
import numpy as np
import torch
import torchvision
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F
from utils import SerializationTool, make_infinite, wassertein_loss, l2_regularization, calcu_fid, \
    calc_gradient_penalty, kl_loss
from logger import Logger

Tensor = torch.cuda.FloatTensor

# Parsing
parser = argparse.ArgumentParser('Train reptile on omniglot')

# Mode
parser.add_argument('--logdir', default="log", type=str, help='Folder to store everything/load')

# - Training params
parser.add_argument('-T', default=100000, type=int, help='num of communications')
parser.add_argument('-N', default=1, type=int, help='num of client')
parser.add_argument('--model', default="lsgan", type=str, choices=["lsgan", "cgan"], help='num of client')
parser.add_argument('--dataset', default="mnist", type=str, choices=["emnist", "mnist"], help='num of client')
parser.add_argument('--shareway', default="kd", type=str, choices=["kd", "fl", "idea"],
                    help='the method to share between clients/twins')
parser.add_argument('--num_tasks', default=5, type=int, help='number of sample tasks')
parser.add_argument('--meta_epochs', default=20, type=int, help='number of meta iterations')
parser.add_argument('--test_iterations', default=50, type=int, help='number of base iterations')
parser.add_argument('--batch', default=20, type=int, help='minibatch size in base task')
parser.add_argument('--meta-lr', default=1e-4, type=float, help='meta learning rate')
parser.add_argument('--lr', default=0.0002, type=float, help='base learning rate')
parser.add_argument('--m', default=1, type=int, help='base learning rate')
# - General params
parser.add_argument('--validation', default=0.1, type=float, help='Percentage of validation')
parser.add_argument('--validate_every', default=500, type=int, help='Meta-evaluation every ... base-tasks')
parser.add_argument('--cuda', default=0, type=int, help='Use cuda')
parser.add_argument('--check_every', default=1000, type=int, help='Checkpoint every')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()
id_string = f"gossipgan_{args.N}+{args.model}+{args.dataset}+{args.shareway}+{args.meta_epochs}+{args.batch}"
log_dir = f"./checkpoint/{id_string}/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = Logger(log_dir)


# writer = SummaryWriter("log")

class AbstractTrainer(object):
    def __init__(self, args, idx):
        self.idx = idx
        self.args = args
        self.id_string = f"{idx}"
        num_class = 62 if self.args.dataset == "emnist" else 10
        from models import Generator, Discriminator
        self.generator = Generator(num_class, LayerNorm=False).to(device)
        self.discriminator = Discriminator(num_class, LayerNorm=False).to(device)
        self.fake_targets = torch.tensor([0] * args.batch, dtype=torch.float, device=device).view(-1, 1)
        self.real_targets = torch.tensor([1] * args.batch, dtype=torch.float, device=device).view(-1, 1)
        self.opti_d = optim.Adam(params=self.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.opti_g = optim.Adam(params=self.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.loss = nn.BCEWithLogitsLoss()

    def train(self, *args, **kwargs):
        # if self.args.meta_arg:
        self.meta_training_loop()
        # else:
        #     self.base_training_loop()

    def meta_training_loop(self):
        pass

    def base_training_loop(self):
        pass

    def validate_run(self):
        pass

    def load_checkpoint(self):
        if not os.path.exists(f"{log_dir}{self.id_string}"):
            self.eps = 0
            return
        checkpoint = torch.load(f"{log_dir}{self.id_string}")
        self.generator.load_state_dict(checkpoint["model"]["G"])
        self.discriminator.load_state_dict(checkpoint["model"]["D"])
        self.opti_g.load_state_dict(checkpoint["opti"]["G"])
        self.opti_d.load_state_dict(checkpoint["opti"]["D"])
        self.eps = checkpoint["eps"]

    def checkpoint_step(self):
        checkpoint = {
            "model": {"G": self.generator.state_dict(), "D": self.discriminator.state_dict()},
            "opti": {"G": self.opti_g.state_dict(), "D": self.opti_d.state_dict()},
            "eps": self.eps
        }

        torch.save(checkpoint, f"{log_dir}{self.id_string}")
        torch.save(checkpoint, f"{log_dir}{self.id_string}_{self.eps}")

class Client(AbstractTrainer):
    def __init__(self, dataset, args, idx):
        super().__init__(args, idx)
        self.id_string = "Client_" + self.id_string
        self.dataset = dataset
        self.cache = []
        self.loader = make_infinite(DataLoader(dataset.get_dataset(), batch_size=self.args.batch, shuffle=True))
        self.load_checkpoint()

    def inner_loop(self, data):
        x_r, lbl = data
        fake = Variable(Tensor(x_r.shape[0], 1).fill_(0.0), requires_grad=False)
        real = Variable(Tensor(x_r.shape[0], 1).fill_(1.0), requires_grad=False)
        # update d
        z = torch.randn((x_r.shape[0], 100)).to(device)
        lbl = lbl.to(device)
        x_g = self.generator(z, lbl).detach()
        x_r = x_r.type(Tensor)

        pre_r = self.discriminator(x_r, lbl)
        pre_f = self.discriminator(x_g, lbl)

        # gp = calc_gradient_penalty(d, x_r, x_g, device=device)
        loss_adv = 0.5 * (self.loss(pre_r, real) + self.loss(pre_f, fake))
        loss_d = loss_adv
        self.opti_d.zero_grad()
        loss_d.backward()
        grad_d = SerializationTool.serialize_model(self.discriminator, "grad").norm().mean()
        self.opti_d.step()

        # update g
        real = Variable(Tensor(self.args.batch, 1).fill_(1.0), requires_grad=False)
        z = torch.randn((self.args.batch, 100)).to(device)
        x_g = self.generator(z, lbl)
        pre_f = self.discriminator(x_g, lbl)
        loss_g = self.loss(pre_f, real)
        self.opti_g.zero_grad()
        loss_g.backward()
        grad_g = SerializationTool.serialize_model(self.generator, "grad").mean()
        self.opti_g.step()
        return loss_d.item(), loss_g.item(), grad_d.item(), grad_g.item()

    def validate_run(self):
        real_img = make_infinite(DataLoader(self.dataset.get_random_test_task(100), batch_size=100, shuffle=True))
        x_r, l_r = next(real_img)
        z = torch.tensor(np.random.normal(size=(100, 100)), dtype=torch.float, device=device)
        self.generator.eval()
        self.discriminator.eval()
        x_g = self.generator(z, l_r.to(device)).cpu()
        # fid = calcu_fid(x_r, x_g)
        torchvision.utils.save_image(x_r[np.random.choice(range(len(x_r)), 25, replace=False)],
                                     f"{log_dir}ground_truth.png",
                                     nrow=5, normalize=True)
        torchvision.utils.save_image(x_g[np.random.choice(range(len(x_g)), 25, replace=False)],
                                     f"{log_dir}{self.id_string}.png",
                                     nrow=5, normalize=True)
        # logger.info(f"EPOCHS: [{self.eps}/{self.args.T}]; {self.id_string}; "
        #             f"Real Predition: {self.discriminator(x_r.type(Tensor), l_r.to(device)).mean(dim=0).cpu().item()}; "
        #             f"Fake Predition: {self.discriminator(x_g.type(Tensor), l_r.to(device)).mean(dim=0).cpu().item()}")

    def train(self, t):
        self.eps = t
        super().train()

    def meta_training_loop(self):
        self.excu_cache()
        self.generator.train()
        self.discriminator.train()
        d_loss, g_loss, d_grad, g_grad = 0, 0, 0, 0
        for i in range(self.args.meta_epochs):
            data = next(self.loader)
            while len(data[0]) != self.args.batch:
                data = next(self.loader)
            loss = self.inner_loop(data)
            d_loss += loss[0]
            g_loss += loss[1]
            d_grad += loss[2]
            g_grad += loss[3]
        logger.info(f"EPOCHS: [{self.eps}/{self.args.T}]; [{self.id_string}]; "
                    f"D_Loss: {d_loss / 200}; G_loss: {g_loss / 200}; "
                    f"D_Grad: {d_grad / 200}; G_grad: {g_grad / 200}")

    def excu_cache(self):
        if len(self.cache) > 0:
            generator_sum = 0
            discriminator_sum = 0
            for state_dict in self.cache:
                generator_sum += state_dict[0]
                discriminator_sum += state_dict[1]
            generator_sum += SerializationTool.serialize_model(self.generator)
            discriminator_sum += SerializationTool.serialize_model(self.discriminator)
            SerializationTool.deserialize_model(self.generator, generator_sum / (len(self.cache) + 1))
            SerializationTool.deserialize_model(self.discriminator, discriminator_sum / (len(self.cache) + 1))
            self.cache = []

    def state_dict(self):
        return (SerializationTool.serialize_model(self.generator),
                SerializationTool.serialize_model(self.discriminator))


import pickle as pkl


def main_loop():
    # read the dataset
    datasets = EMNIST("data", args.N, True, dataset=args.dataset)
    # initialized the clients and twins
    clients = [Client(dataset, args, i) for i, dataset in enumerate(datasets)]
    pkl.dump(datasets.datasets_index, open(f'{log_dir}data_distribution.pkl', 'wb'))

    def client_train_step(t):
        for i in range(args.N):
            clients[i].train(t)

    def share_para():
        num_choose = max(0, int(math.log2(args.N)))
        for i in range(args.N):
            chosen_clients = np.random.choice([j for j in range(args.N) if j != i], size=num_choose, replace=False)
            for j in chosen_clients:
                clients[i].cache.append(clients[j].state_dict())

    def checkpoint_step():
        for i in range(args.N):
            clients[i].checkpoint_step()
            clients[i].validate_run()

    for t in range(clients[0].eps, args.T):
        if t % args.check_every == 0:
            checkpoint_step()
        client_train_step(t)
        share_para()


if __name__ == "__main__":
    main_loop()
