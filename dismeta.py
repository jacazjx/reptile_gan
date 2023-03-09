import argparse
import math

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch import optim
from tqdm import tqdm

from utils import SerializationTool, make_infinite
from logger import Logger
# Parsing
parser = argparse.ArgumentParser('Train reptile on omniglot')

# Mode
parser.add_argument('--logdir', default="log", type=str, help='Folder to store everything/load')

# - Training params
parser.add_argument('-T', default=10000, type=int, help='num of communications')
parser.add_argument('-N', default=1, type=int, help='num of client')
parser.add_argument('--model', default="lsgan", type=str, choices= ["lsgan", "cgan"],help='num of client')
parser.add_argument('--dataset', default="femnist", type=str, choices= ["femnist", "mnist"],help='num of client')
parser.add_argument('--meta', dest="meta_arg", action="store_true", default=False, help="whether use meta learning")
parser.add_argument('--shareway', default="kd", type=str, choices= ["kd", "fl", "idea"],help='the method to share between clients/twins')
parser.add_argument('--meta_epochs', default=10, type=int, help='number of meta iterations')
parser.add_argument('--test_iterations', default=50, type=int, help='number of base iterations')
parser.add_argument('--batch', default=10, type=int, help='minibatch size in base task')
parser.add_argument('--meta-lr', default=1e-3, type=float, help='meta learning rate')
parser.add_argument('--lr', default=1e-4, type=float, help='base learning rate')
parser.add_argument('--m', default=1, type=int, help='base learning rate')

# - General params
parser.add_argument('--validation', default=0.1, type=float, help='Percentage of validation')
parser.add_argument('--validate_every', default=100, type=int, help='Meta-evaluation every ... base-tasks')
parser.add_argument('--cuda', default=0, type=int, help='Use cuda')
parser.add_argument('--check_every', default=10000, type=int, help='Checkpoint every')
parser.add_argument('--checkpoint', default='', help='Path to checkpoint. This works only if starting fresh (i.e., no checkpoints in logdir)')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args = parser.parse_args()
id_string = f"{args.N}+{args.model}+{args.dataset}+meta_{args.meta_arg}+{args.shareway}+{args.meta_epochs}+{args.batch}"
logger = Logger("log", id_string)

class AbstractTrainer(object):
    def __init__(self, args, idx):
        self.idx = idx
        self.args = args
        self.id_string = f"{idx}"
        from models import Generator, Discriminator
        if args.model == 'lsgan':
            self.generator = Generator().to(device)
            self.discriminator = Discriminator().to(device)
        self.d_targets = torch.tensor([1] * args.batch + [-1] * args.batch, dtype=torch.float,
                                                  device=device).view(-1, 1)
        self.g_targets = torch.tensor([1] * args.batch, dtype=torch.float, device=device).view(-1, 1)
        self.opti_d = optim.Adam(params=self.discriminator.parameters(), lr=args.lr)
        self.opti_g = optim.Adam(params=self.generator.parameters(), lr=args.lr)
        self.loss = torch.nn.BCEWithLogitsLoss()


    def train(self, *args, **kwargs):
        if self.args.meta_arg:
            self.meta_training_loop()
        else:
            self.base_training_loop()

    def meta_training_loop(self):
        pass

    def base_training_loop(self):
        pass

class twin(AbstractTrainer):
    def __init__(self, args, num_samples, idx):
        super().__init__(args, idx)
        self.id_string = "Twin_" + self.id_string
        self.num_samples = num_samples
        self.meta_g = self.generator.clone()
        self.meta_opti_g = optim.SGD(self.meta_g.parameters(),  lr=args.meta_lr)
        self.cache = []
        self.share_cache = None
        self.A = torch.zeros(args.N).fill_(1 / (args.N - 1)) if args.N > 1 else None
        if self.A is not None:
            self.A[idx] = 0

    def train(self, t):
        self.share_cache = None
        super().train()
        if t % int((self.num_samples * self.args.m) / self.args.batch) == 0 and t != 0 and self.A is not None:
            num_choose = max(int(math.log(self.args.N, 2)), 1)
            chosen = list(WeightedRandomSampler(self.A, num_choose, replacement=False))
            if chosen.count(self.idx) > 0:
                chosen.remove(self.idx)
            return chosen
        return []

    def share_data(self):
        if self.share_cache is not None:
            return self.share_cache

        if self.args.shareway == "kd":
            z = torch.tensor(np.random.normal(size=(self.args.batch, 100)), dtype=torch.float, device=device)
            x = self.generator(z)
            y, kn = self.discriminator(x, True)
            self.share_cache = (z, x, kn, y)

        elif self.args.shareway == "idea":
            pass

        elif self.args.shareway == "fl":
            pass

        return self.share_cache

    def inner_loop(self):
        z = torch.tensor(np.random.normal(size=(self.args.batch, 100)), dtype=torch.float, device=device)
        x_g = self.meta_g(z)
        x = x_g
        loss = self.loss(self.discriminator(x), self.g_targets)
        loss.backward()
        self.meta_opti_g.step()
        return loss.item()

    def meta_training_loop(self):
        self.generator.train()
        self.discriminator.eval()
        self.meta_g.load_state_dict(self.generator.state_dict())
        g_loss = []
        for i in range(self.args.meta_epochs):
            g_loss.append(self.inner_loop())
        g_loss = np.array(g_loss).mean()
        logger.info(f"{self.id_string}_Loss_{g_loss}")

        self.generator.point_grad_to(self.meta_g)
        self.opti_g.step()
        return g_loss
    def fine_tuning(self):
        zs = []
        xs = []
        fs = []
        ys = []
        while len(self.cache) > 0:
            kd = self.cache.pop()
            zs.append(kd[0])
            xs.append(kd[1])
            fs.append(kd[2])
            ys.append(kd[3])
        weight = torch.zeros(len(zs))

    def sync(self, para):
        SerializationTool.deserialize_model(self.discriminator, para)

    def state_dict(self):
        return SerializationTool.serialize_model(self.generator)

class client(AbstractTrainer):
    def __init__(self, dataset, args, idx):
        super().__init__(args, idx)
        self.id_string = "Client_" + self.id_string
        self.dataset = dataset
        self.meta_d = self.discriminator.clone()
        self.meta_opti_d = optim.SGD(self.meta_d.parameters(), lr=args.meta_lr)
        self.cache = []

    def inner_loop(self, x_r):
        z = torch.tensor(np.random.normal(size=(self.args.batch, 100)), dtype=torch.float, device=device)
        x_g = self.generator(z)
        x_r = x_r[0].to(device).view(x_g.shape)
        x = torch.cat([x_r, x_g])
        loss = self.loss(self.meta_d(x), self.d_targets)
        loss.backward()
        self.meta_opti_d.step()
        return loss.item()

    def validate_run(self):
        pass

    def meta_training_loop(self):
        self.generator.eval()
        self.discriminator.train()
        batch = self.dataset.get_random_task(5, self.args.batch)
        batch = make_infinite(DataLoader(batch, batch_size=self.args.batch, shuffle=False))

        self.meta_d.load_state_dict(self.discriminator.state_dict())
        d_loss = []
        for i in range(self.args.meta_epochs):
            d_loss.append(self.inner_loop(next(batch)))

        d_loss = np.array(d_loss).mean()
        logger.info(f"{self.id_string}_Loss_{d_loss}")
        self.discriminator.point_grad_to(self.meta_d)
        self.opti_d.step()

        return d_loss

    def checkpoint_step(self):
        pass

    def sync(self, para):
        SerializationTool.deserialize_model(self.generator, para)

    def state_dict(self):
        return SerializationTool.serialize_model(self.discriminator)

def main_loop():
    # read the dataset
    if args.dataset == "femnist":
        from datasets.FEMNIST import FEMNIST
        datasets = FEMNIST("data", args.N)
    # initialized the clients and twins
    clients = [client(dataset, args, i) for i, dataset in enumerate(datasets)]
    twins = [twin(args, clients[i].dataset.num_samples, i) for i in range(args.N)]

    def client_train_step(t):
        for i in range(args.N):
            clients[i].train()

    def twin_train_step(t):
        for i in range(args.N):
            chosen = twins[i].train(t)
            for ch in chosen:
                twins[ch].cache.append(twins[i].share_data())

        for i in range(args.N):
            twins[i].fine_tuning()

    def sync_para(srcs, dests):
        for src, dest in zip(srcs, dests):
            src.sync(dest.state_dict())


    for t in tqdm(range(args.T), leave=False):
        client_train_step(t)
        sync_para(clients, twins)
        twin_train_step(t)
        sync_para(twins, clients)





if __name__ == "__main__":
    main_loop()