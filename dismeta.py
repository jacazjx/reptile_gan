import argparse
import math
import os
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
parser.add_argument('--m', default=1, type=int, help='base learning rate')

# - General params
parser.add_argument('--validation', default=0.1, type=float, help='Percentage of validation')
parser.add_argument('--validate_every', default=100, type=int, help='Meta-evaluation every ... base-tasks')
parser.add_argument('--cuda', default=0, type=int, help='Use cuda')
parser.add_argument('--check_every', default=500, type=int, help='Checkpoint every')
parser.add_argument('--checkpoint', default='checkpoint',
                    help='Path to checkpoint. This works only if starting fresh (i.e., no checkpoints in logdir)')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()
id_string = f"[ModelName:{args.model}]_[NumClient:{args.N}]_[Dataset:{args.dataset}]_[NumTask:{args.num_tasks}]_" \
            f"[IsNonIID:{args.noniid}]_[IsCondition:{args.cond}]_[IsFeatureExtra:{args.feature_extra}]_" \
            f"[ShareWay:{args.shareway}]_[NumInnerLoop:{args.meta_epochs}]_[Batch:{args.batch}]"
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
        self.generator = Generator(num_class, LayerNorm=self.args.meta_arg, Condition=self.args.cond).to(device)
        self.discriminator = Discriminator(num_class, LayerNorm=self.args.meta_arg, Condition=self.args.cond,
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
        torch.save(checkpoint, f"{log_dir}{self.id_string}_{self.eps}")


class twin(AbstractTrainer):
    def __init__(self, args, num_samples, idx):
        super().__init__(args, idx)
        self.collections = None
        self.id_string = "Twin_" + self.id_string
        self.num_samples = num_samples
        self.cache = []
        self.share_cache = None
        self.A = torch.zeros(args.N).fill_(1 / (args.N - 1)) if args.N > 1 else None
        if self.A is not None:
            self.A[idx] = 0
        self.load_checkpoint()

    def train(self, t):
        self.eps = t
        self.share_cache = None
        super().train()

    def choose_neighbors(self):
        if self.eps % int((self.num_samples * self.args.m) / self.args.batch) == 0 and self.eps != 0 and self.A is not None:
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

    def inner_loop(self, g, d, task, opti):
        z = torch.tensor(np.random.normal(size=(self.args.batch, 100)), dtype=torch.float, device=device)
        l = torch.tensor([task] * self.args.batch).to(device)
        x_g = g(z, l)
        loss = self.loss(d(x_g, l), self.real_targets)
        loss.backward()
        opti.step()
        return loss.item()

    def meta_training_loop(self):
        self.generator.train()
        self.discriminator.eval()
        # self.meta_g.load_state_dict(self.generator.state_dict())
        g_loss = 0
        for t in self.collections:
            g_loss += self.inner_loop(self.generator, self.discriminator, t, self.opti_g)
        logger.info(f"EPOCHS:{self.eps}; {self.id_string}_Loss: {g_loss/len(self.collections)}")
        # writer.add_scalar(f"{self.id_string} G Loss", g_loss, self.eps)
        # self.generator.point_grad_to(self.meta_g)
        # self.opti_g.step()
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
        self.loader = make_infinite(DataLoader(dataset.get_dataset(), batch_size=self.args.batch, shuffle=True))
        self.is_a_epoch = False
        self.collections = None
        self.load_checkpoint()

    def inner_loop(self, g, d, data, opti):
        x_r, lbl = data
        x_r = x_r.to(device)
        lbl = lbl.to(device)
        z = torch.tensor(np.random.normal(size=(self.args.batch, 100)), dtype=torch.float, device=device)

        x_g = g(z, lbl)
        pre_f, pre_r = d(x_g, lbl), d(x_r, lbl)

        loss_adv = 0.5 * (self.loss(pre_r, self.real_targets) + self.loss(pre_f, self.fake_targets))
        loss = loss_adv  # + loss_kl * 0.001
        opti.zero_grad()
        loss.backward()
        opti.step()
        return loss.item()

    def validate_run(self):
        x_t = self.dataset.get_random_test_task(100)
        for d, l in DataLoader(x_t, batch_size=100):
            x_t = d
        z = torch.tensor(np.random.normal(size=(100, 100)), dtype=torch.float, device=device)
        self.generator.eval()
        self.discriminator.eval()
        x_g = self.generator(z).cpu()
        fid = calcu_fid(x_t, x_g)
        torchvision.utils.save_image(x_t[np.random.choice(range(len(x_t)), 25, replace=False)],
                                     f"{log_dir}ground_truth.png",
                                     nrow=5)
        torchvision.utils.save_image(x_g[np.random.choice(range(len(x_g)), 25, replace=False)],
                                     f"{log_dir}{self.id_string}.png",
                                     nrow=5)
        logger.info(f"EPOCHS: {self.eps}; {self.id_string} FID: {fid}; "
                    f"Real Predition: {self.discriminator(x_t.to(device)).mean(dim=0).cpu().item()}; "
                    f"Fake Predition: {self.discriminator(x_g.to(device)).mean(dim=0).cpu().item()}")

    def train(self, t):
        self.eps = t
        super().train()

    def meta_training_loop(self):
        self.generator.eval()
        self.discriminator.train()

        d_loss = []
        tasks = self.dataset.get_random_tasks(self.args.num_tasks)
        grads = []
        for t in tasks:
            meta_g = self.discriminator.clone()
            opti_d = get_optimizer(meta_g, self.opti_d.state_dict())
            minibatch = DataLoader(self.dataset.get_one_task(t, self.args.batch), batch_size=(self.args.batch // self.args.meta_epochs))
            for x, l in minibatch:
                loss = self.inner_loop(self.generator, meta_g, (x, l), opti_d)
                d_loss.append(loss)
            self.discriminator.point_grad_to(meta_g)
            self.opti_d.step()
            # grads.append(SerializationTool.serialize_model(meta_g))
        # grads = SerializationTool.serialize_model(self.discriminator) - torch.stack(grads).mean(dim=0).view(-1)
        # SerializationTool.deserialize_model(self.discriminator, grads, position="grad")
        # self.opti_d.step()
        d_loss = np.array(d_loss).mean()
        logger.info(f"EPOCHS:{self.eps}; {self.id_string}_Loss: {d_loss}")
        self.collections = tasks
        return d_loss

    def sync(self, para):
        SerializationTool.deserialize_model(self.generator, para)

    def state_dict(self):
        return SerializationTool.serialize_model(self.discriminator), self.collections


def main_loop():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.N)
    # read the dataset
    datasets = EMNIST("data", args.N, dataset=args.dataset)
    # initialized the clients and twins
    clients = [client(dataset, args, i) for i, dataset in enumerate(datasets)]
    twins = [twin(args, clients[i].dataset.num_samples, i) for i in range(args.N)]
    pkl.dump(datasets.datasets_index, open(f'{log_dir}data_distribution.pkl', 'wb'))

    def train_step(trainer, t):
        futures = [executor.submit(trainer[i].train, t) for i in range(args.N)]
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)


    def share():
        for i in range(args.N):
            chosen = twins[i].choose_neighbors()
            for ch in chosen:
                twins[ch].cache.append(twins[i].share_data())

        futures = [executor.submit(twins[i].fine_tuning,) for i in range(args.N)]
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

    def sync_para(srcs, dests):
        for src, dest in zip(srcs, dests):
            src.sync(dest.state_dict())

    def checkpoint_step():
        for i in range(args.N):
            twins[i].checkpoint_step()
            clients[i].checkpoint_step()
            clients[i].validate_run()

    for t in tqdm(range(clients[0].eps, args.T + 1), leave=False):
        train_step(clients, t)
        sync_para(clients, twins)
        train_step(twins, t)
        sync_para(twins, clients)
        share()
        if t % args.check_every == 0:
            checkpoint_step()


if __name__ == "__main__":
    main_loop()
