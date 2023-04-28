import argparse
import math
import os
from collections import Counter
import random

import matplotlib.pyplot as plt
import numpy as np
from datasets.EMNIST import EMNIST
import numpy as np
import torch
import torchvision
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F
from utils import SerializationTool, make_infinite, wassertein_loss, l2_regularization, calcu_fid, \
    calc_gradient_penalty, kl_loss, get_optimizer
from logger import Logger
import pickle as pkl
import concurrent.futures
import traceback

# Parsing
parser = argparse.ArgumentParser('Train reptile on omniglot')

# Mode
parser.add_argument('--logdir', default="log", type=str, help='Folder to store everything/load')

# - Training params
parser.add_argument('-T', default=10000, type=int, help='num of communications')
parser.add_argument('-N', default=1, type=int, help='num of client')
parser.add_argument('--model', default="twingan", type=str, choices=["fegan", "mdgan", "gossipgan", "twingan"],
                    help='num of client')
parser.add_argument('--dataset', default="mnist", type=str, choices=["emnist", "mnist"], help='num of client')
parser.add_argument('--meta', dest="meta_arg", action="store_true", default=False, help="whether use meta learning")
parser.add_argument('--niid', dest="noniid", action="store_true", default=False, help="whether use non-iid")
parser.add_argument('--ln', dest="layer_norm", action="store_true", default=False, help="whether use layer_norm")
parser.add_argument('--feature', dest="feature_extra", action="store_true", default=False,
                    help="whether use feature extra")
parser.add_argument('--condition', dest="cond", action="store_true", default=False, help="whether use condition")
parser.add_argument('--shareway', default="kd", type=str, choices=["kd", "fl", "idea"],
                    help='the method to share between clients/twins')
parser.add_argument('--num_tasks', default=5, type=int, help='number of sample tasks')
parser.add_argument('--meta_epochs', default=20, type=int, help='number of meta iterations')
parser.add_argument('--test_iterations', default=50, type=int, help='number of base iterations')
parser.add_argument('--batch', default=20, type=int, help='minibatch size in base task')
parser.add_argument('--meta-lr', default=1e-4, type=float, help='meta learning rate')
parser.add_argument('--lr', default=0.0002, type=float, help='base learning rate')
parser.add_argument('--m', default=1, type=int, help='')
parser.add_argument('--Lambda', default=0.01, type=float, help='regularition')
parser.add_argument('--S', default=20, type=int, help='when to share')
parser.add_argument('--R', default=5000, type=int, help='when to change the strategy')

# - General params
parser.add_argument('--validation', default=0.1, type=float, help='Percentage of validation')
parser.add_argument('--validate_every', default=100, type=int, help='Meta-evaluation every ... base-tasks')
parser.add_argument('--cuda', default=0, type=int, help='Use cuda')
parser.add_argument('--check_every', default=500, type=int, help='Checkpoint every')
parser.add_argument('--checkpoint', default='checkpoint',
                    help='Path to checkpoint. This works only if starting fresh (i.e., no checkpoints in logdir)')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()
id_string = f"[ModelName={args.model}]_[NumClient={args.N}]_[Dataset={args.dataset}]_[NumTask={args.num_tasks}]_" \
            f"[IsNonIID={args.noniid}]_[IsCondition={args.cond}]_[IsLayerNrom{args.layer_norm}]_[IsFeatureExtra={args.feature_extra}]_" \
            f"[ShareWay={args.shareway}]_[NumInnerLoop={args.meta_epochs}]_[Batch={args.batch}]"
log_dir = f"./{args.checkpoint}/{id_string}/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = Logger(log_dir)
logger.info(args)

num_classes = {
    "emnist": 62,
    "mnist": 10,
    "cifar10": 10,
    "mini_imagenet": 100,
}
num_class = num_classes[args.dataset]

np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)


class AbstractTrainer(object):
    def __init__(self, args, idx):
        self.idx = idx
        self.args = args
        self.id_string = f"{idx}"
        from models import Generator, Discriminator
        self.generator = Generator(num_class, LayerNorm=self.args.layer_norm, Condition=self.args.cond).to(device)
        self.discriminator = Discriminator(num_class, LayerNorm=self.args.layer_norm, Condition=self.args.cond,
                                           FeatureExtraction=self.args.feature_extra).to(device)
        self.fake_targets = torch.tensor([0] * args.batch, dtype=torch.float, device=device).view(-1, 1)
        self.real_targets = torch.tensor([1] * args.batch, dtype=torch.float, device=device).view(-1, 1)
        self.opti_d = optim.Adam(params=self.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.opti_g = optim.Adam(params=self.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.loss = nn.BCEWithLogitsLoss()

    def train(self, *args, **kwargs):
        if self.args.meta_arg:
            self.meta_training_loop()
        else:
            self.base_training_loop()

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
        # torch.save(checkpoint, f"{log_dir}{self.id_string}_{self.eps}")


class twin(AbstractTrainer):
    def __init__(self, args, num_samples, idx):
        super().__init__(args, idx)
        self.collections = None
        self.counter = torch.zeros(num_class)
        self.id_string = "Twin_" + self.id_string
        self.cache = []
        self.share_cache = None
        self.state_g = None
        self.decay = 0.5
        self.load_checkpoint()

    def train(self, t):
        self.eps = t
        self.share_cache = None
        super().train()

    def choose_neighbors(self, t):
        if args.N == 1:
            return []
        num_choose = max(int(math.log2(self.args.N)), 1)
        rd = random.random()
        if rd > self.decay:
            chosen = torch.topk(A[self.idx][:], k=num_choose)[1]
        else:
            _clients = [i for i in range(self.args.N)]
            _clients.remove(self.idx)
            chosen = np.random.choice(_clients, size=num_choose, replace=False)
        self.chosen = chosen
        self.decay -= 0.001
        return chosen

    def share_data(self):

        if self.args.shareway == "kd":
            self.discriminator.eval()
            self.generator.eval()
            # 拿出自己最拿手的项目
            task = list(WeightedRandomSampler(self.counter / self.counter.sum(), 1, replacement=False))
            l = torch.tensor(task*5).to(device)
            z = torch.randn((l.shape[0], 100)).to(device)
            x = self.generator(z, l)
            y = SerializationTool.serialize_model(self.discriminator)
            self.share_cache = (z, l, x, y)

        elif self.args.shareway == "fl":
            pass

        return self.share_cache

    def inner_loop(self, g, d, task, opti):
        z = torch.randn((5, 100)).to(device)
        l = torch.tensor(task).to(device)
        x_g = g(z, l)
        y_g = d(x_g, l)
        # loss = -torch.mean(y_g)
        loss = -torch.mean(torch.sigmoid(y_g).log())
        opti.zero_grad()
        loss.backward()
        opti.step()
        return loss.item()

    def meta_training_loop(self):
        self.generator.train()
        self.discriminator.eval()

        g_loss = 0
        for _ in range(self.args.meta_epochs):
            meta_g = self.generator.clone()
            meta_opti_g = get_optimizer(meta_g, self.state_g)
            # self.collections = np.random.choice(self.num_samples, self.args.batch, replace=False)
            np.random.shuffle(self.collections)
            for i in range(self.args.meta_epochs):
                g_loss += self.inner_loop(meta_g, self.discriminator, np.repeat(self.collections[i], 5), meta_opti_g)
            self.state_g = meta_opti_g.state_dict()
            self.generator.point_grad_to(meta_g)
            self.opti_g.step()
        self.counter[self.collections] += 1 # 记录学了什么

        # logger.info(
        #     f"EPOCHS:{self.eps}; {self.id_string}_Loss: {g_loss / (len(self.collections) * self.args.meta_epochs)}")
        return g_loss

    def fine_tuning(self):
        if len(self.cache) == 0:
            return
        try:
            global A
            self.generator.train()
            self.discriminator.eval()
            chs = []
            for ch, kd in self.cache:
                _z, l, _x, d = kd[0].clone().detach(), kd[1].clone().detach(), kd[2].clone().detach(), kd[3]
                _D = self.discriminator.clone()
                SerializationTool.deserialize_model(_D, d)
                _D.eval()
                sim = []

                # fine tuning
                meta_g = self.generator.clone()
                meta_opti_g = get_optimizer(meta_g, self.state_g)
                for i in range(self.args.meta_epochs):
                    z = torch.randn(_z.size()).to(device)
                    x = meta_g(z, l)
                    _y, _f = _D(_x, l, True)
                    y, f = _D(x, l, True)
                    Loss_mse = torch.mean(F.mse_loss(x, _x, reduction="none").view(x.shape[0], -1).sum(dim=-1) /
                                          F.mse_loss(z, _z, reduction="none").view(x.shape[0], -1).sum(dim=-1))

                    Loss_wd = F.l1_loss(f, _f, reduction="mean")
                    # Loss_adv = -torch.mean(y)
                    Loss_adv = -torch.mean(torch.sigmoid(y).log())

                    loss = Loss_adv + 0.5 * Loss_wd + 0.5 * Loss_mse

                    meta_opti_g.zero_grad()
                    loss.backward()
                    meta_opti_g.step()
                    sim.append(torch.sigmoid(torch.mean(y - _y) ** -1).cpu().item())

                self.generator.point_grad_to(meta_g)
                self.opti_g.step()

                A[self.idx][ch] += (sum(sim) / len(sim))
                # 继续记录学了什么
                self.counter[l[0]] += 1
                chs.append(ch)
            # A[self.idx][:] = A[self.idx][:] / A[self.idx][:].norm()
            logger.info(f"[{self.id_string}] recv from {chs}; Knowledge Vector: {np.array(A[self.idx][:])}")
            self.share_cache = None
            self.cache.clear()
        except RuntimeError:
            print(traceback.format_exc())

    def sync(self, para):
        SerializationTool.deserialize_model(self.discriminator, para[0])
        self.collections = para[1]

    def state_dict(self):
        return SerializationTool.serialize_model(self.generator)


class client(AbstractTrainer):
    def __init__(self, dataset, args, idx):
        super().__init__(args, idx)
        self.id_string = "Client_" + self.id_string
        self.dataset = dataset
        self.cache = []
        self.loader = DataLoader(dataset.get_dataset(), batch_size=self.args.batch // self.args.meta_epochs,
                                 shuffle=True)
        self.is_a_epoch = False
        self.batches = 0
        self.collections = None
        self.meta_state = None
        self.load_checkpoint()

    def inner_loop(self, g, d, data, opti):
        x_r, lbl = data
        x_r = x_r.to(device)
        lbl = lbl.to(device)
        z = torch.randn((x_r.shape[0], 100)).to(device)

        x_g = g(z, lbl)
        pre_f, pre_r = d(x_g, lbl), d(x_r, lbl)

        # loss_adv = -torch.mean(pre_r) + torch.mean(pre_f)
        loss_adv = self.loss(pre_f, torch.zeros(pre_f.shape[0]).view(-1, 1).to(device)) + self.loss(pre_r, torch.ones(
            pre_r.shape[0]).view(-1, 1).to(device))
        loss = loss_adv
        opti.zero_grad()
        loss.backward()
        opti.step()
        return torch.mean(pre_r).item(), torch.mean(pre_f).item()

    def validate_run(self):
        l = torch.arange(num_class).view(-1, 1).expand(num_class, 10).reshape(num_class * 10).to(device)
        z = torch.randn((num_class * 10, 100)).to(device)
        self.generator.eval()
        self.discriminator.eval()
        x_g = self.generator(z, l).cpu()

        torchvision.utils.save_image(x_g, f"{log_dir}{self.id_string}.png", nrow=10)

        test_data = DataLoader(self.dataset.get_random_test_task(100), batch_size=100)
        for x_r, l in test_data:
            z = torch.randn((100, 100)).to(device)
            x_g = self.generator(z, l.to(device))
            fid = calcu_fid(x_r, x_g)
            logger.info(f"EPOCHS:{self.eps}; {self.id_string}; FID:{fid}")

    def train(self, t):
        self.eps = t
        super().train()

    def meta_training_loop(self):
        self.generator.eval()
        self.discriminator.train()

        d_loss = []
        g_loss = []
        tasks = self.dataset.get_random_tasks(self.args.num_tasks) # 这一轮学了什么
        # for i, data in enumerate(self.loader):
        #     if i == self.args.meta_epochs:
        #         break
        #     self.batches += 1
        #     loss = self.inner_loop(self.generator, self.discriminator, data, self.opti_d)
        #     d_loss.append(loss[0])
        #     g_loss.append(loss[1])

        for _ in range(self.args.meta_epochs):
            tasks = self.dataset.get_random_tasks(self.args.num_tasks)
            meta_d = self.discriminator.clone()
            meta_opti_d = get_optimizer(meta_d, self.meta_state)
            # 专精这一项
            for i in range(self.args.meta_epochs):
                samples = iter(DataLoader(self.dataset.get_n_task(tasks, 5), batch_size=5, shuffle=True))
                loss = self.inner_loop(self.generator,  meta_d, next(samples), meta_opti_d)
                if i == self.args.meta_epochs - 1:
                    d_loss.append(loss[0])
                    g_loss.append(loss[1])
            self.discriminator.point_grad_to(meta_d) # 计算梯度
            self.meta_state = meta_opti_d.state_dict()
            self.opti_d.step() # 更新参数
        d_loss = np.array(d_loss).mean()
        g_loss = np.array(g_loss).mean()
        if self.eps % 10 == 0:
            logger.info(f"EPOCHS: [{self.eps}/{self.args.T}]; {self.id_string}; Loss D: {d_loss}; Loss G: {g_loss}")
        self.collections = tasks
        return d_loss

    def sync(self, para):
        SerializationTool.deserialize_model(self.generator, para)

    def state_dict(self):
        return SerializationTool.serialize_model(self.discriminator), self.collections


def main_loop():
    global A
    A = torch.zeros((args.N, args.N), requires_grad=False)  # SimilarityMatrix
    # A = torch.zeros((args.N, num_class), requires_grad=False)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.N)
    # read the dataset
    datasets = EMNIST("data", args.N, iid=not args.noniid, dataset=args.dataset)
    # initialized the clients and twins
    clients = [client(dataset, args, i) for i, dataset in enumerate(datasets)]
    twins = [twin(args, datasets.datasets_index[i], i) for i in range(args.N)]

    counter = []
    for target in datasets.datasets_index:
        count = torch.zeros(num_class)
        for c in range(num_class):
            count[c] += target.count(c)
        counter.append(count / count.sum())
    pkl.dump(counter, open(f'{log_dir}data_distribution.pkl', 'wb'))

    def train_step(trainer, t):
        futures = [executor.submit(trainer[i].train, t) for i in range(args.N)]
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

    for i in range(args.N):
        for j in range(args.N):
            if i != j:
                A[i][j] = F.kl_div(torch.log_softmax(counter[i], dim=-1), torch.softmax(counter[j], dim=-1)) ** -1
        norms = A[i][:] / A[i][:].norm()
        A[i][:] = torch.softmax(norms, dim=-1)
        A[i][i] = 0

    def share(t):
        chosens = []
        for i in range(args.N):
            chosen = twins[i].choose_neighbors(t)
            chosens.append(chosen)
            for ch in chosen:
                if ch == i:
                    continue
                twins[ch].cache.append((i, twins[i].share_data()))
            clients[i].is_a_epoch = False
            twins[i].share_cache = None
        futures = [executor.submit(twins[i].fine_tuning) for i in range(args.N)]
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

    def sync_para(srcs, dests):
        for src, dest in zip(srcs, dests):
            dest.sync(src.state_dict())

    def checkpoint_step(t):
        for i in range(args.N):
            clients[i].checkpoint_step()
            clients[i].validate_run()

        # Convert matrix A to a numpy array
        A_np = np.array(A)
        # Create heatmap plot
        plt.imshow(A_np, cmap='hot', interpolation='nearest')
        plt.colorbar()  # add colorbar
        plt.xlabel('Client Index')  # add x-axis label
        plt.ylabel('Client Index')  # add y-axis label
        plt.title(f"Heatmap of Similarity Matrix")  # add title
        # Save plot to log folder
        plt.savefig(f"{log_dir}heatmap_{t}.png")
        plt.close()

    for t in tqdm(range(clients[0].eps, args.T + 1), leave=False):
        train_step(clients, t)
        sync_para(clients, twins)
        train_step(twins, t)
        if t % args.S == 0 and t != 0:
            share(t)
        sync_para(twins, clients)
        if t % args.check_every == 0:
            checkpoint_step(t)

if __name__ == "__main__":
    main_loop()
