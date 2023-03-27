import argparse
import math
import os
from copy import deepcopy
from queue import Queue

import numpy as np
import scipy
import torch
import torchvision
from scipy import stats
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets.EMNIST import EMNIST
from logger import Logger
from utils import SerializationTool, make_infinite
from dismeta import AbstractTrainer, num_class, args, logger, id_string, log_dir, device
Tensor = torch.cuda.FloatTensor


np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)



class FLServer(AbstractTrainer):
    def __init__(self, args):
        super().__init__(args, 0)
        self.id_string = "Server"
        self.cache = []
        self.load_checkpoint()

    def aggregate(self, weights):
        if len(self.cache) > 0:
            g_paras = torch.stack([c[0] for c in self.cache], 0)
            d_paras = torch.stack([c[1] for c in self.cache], 0)
            g_aggr = (g_paras * weights.view(-1, 1)).sum(dim=0)
            d_aggr = (d_paras * weights.view(-1, 1)).sum(dim=0)
            self.cache = []
            return g_aggr, d_aggr
        else:
            return 0, 0


class MDServer(AbstractTrainer):
    def __init__(self, args):
        super().__init__(args, 0)
        self.id_string = "Server"
        self.cache = []
        self.xs = None
        self.load_checkpoint()

    def generate(self):
        self.generator.train()
        pkg = []
        k = max(2, int(math.log2(self.args.N)))
        z = torch.randn((self.args.batch * k, 100)).to(device)
        lbl = torch.from_numpy(np.random.choice(num_class, self.args.batch * k, replace=True)).to(device)
        xs = torch.chunk(self.generator(z, lbl), k)
        ls = torch.chunk(lbl, k)
        for i in range(k):
            pkg.append((xs[i], ls[i]))
        self.xs = xs
        return pkg

    def validate_run(self):
        self.generator.eval()
        z = torch.randn((100, 100)).to(device)
        lbl = torch.from_numpy(np.random.choice(num_class, 100, replace=True)).to(device)
        x_g = self.generator(z, lbl)
        torchvision.utils.save_image(x_g, f"{log_dir}{self.id_string}.png", nrow=10,  normalize=True)

    def aggregate(self):
        if len(self.cache) > 0:
            loss = sum(self.cache) / self.args.N
            self.opti_g.zero_grad()
            loss.backward()
            self.opti_g.step()
            self.cache.clear()
        return self.generate()

class MDClient(AbstractTrainer):
    def __init__(self, dataset, args, idx):
        super().__init__(args, idx)
        self.id_string = "Client_" + self.id_string
        self.dataset = dataset
        self.cache = []
        self.loader = iter(DataLoader(dataset.get_dataset(), batch_size=self.args.batch, shuffle=True))
        self.is_a_epoch = False
        self.load_checkpoint()

    def inner_loop(self, data, fake):
        x_d, l_d = fake
        x_r, l_r = data
        fake = Variable(Tensor(x_d.shape[0], 1).fill_(0.0), requires_grad=False)
        real = Variable(Tensor(x_r.shape[0], 1).fill_(1.0), requires_grad=False)
        # update d
        x_r = x_r.type(Tensor)

        pre_r = self.discriminator(x_r, l_r.to(device))
        pre_f = self.discriminator(deepcopy(x_d.clone().detach()), l_d.clone().detach())

        loss_adv = 0.5 * (self.loss(pre_r, real) + self.loss(pre_f, fake))
        loss_d = loss_adv
        self.opti_d.zero_grad()
        loss_d.backward()
        grad_d = SerializationTool.serialize_model(self.discriminator, "grad").norm().mean()
        self.opti_d.step()

        return loss_d.item(), grad_d.item()

    def load_fake(self, x_g, x_d):
        self.cache.append(x_d)
        self.cache.append(x_g)

    def load_dict(self, para):
        SerializationTool.deserialize_model(self.discriminator, para)

    def state_dict(self):
        return SerializationTool.serialize_model(self.discriminator)

    def get_grad(self):
        return self.cache.pop()

    def train(self, t):
        self.eps = t
        super().train()

    def base_training_loop(self):
        self.discriminator.train()
        d_loss, g_loss, d_grad, g_grad = 0, 0, 0, 0
        fake = self.cache.pop()
        for i in range(self.args.meta_epochs):
            try:
                data = next(self.loader)
            except:
                self.is_a_epoch = True
                self.loader = iter(DataLoader(self.dataset.get_dataset(), batch_size=self.args.batch, shuffle=True))
                data = next(self.loader)
            loss = self.inner_loop(data, fake)
            d_loss += loss[0]
            d_grad += loss[1]


        # update g
        x_g, l_g = self.cache.pop()
        real = Variable(Tensor(x_g.shape[0], 1).fill_(1.0), requires_grad=False)
        pre_f = self.discriminator(x_g.clone(), l_g.clone())
        loss_g = self.loss(pre_f, real)
        self.cache.append(loss_g)
        logger.info(f"EPOCHS: [{self.eps}/{self.args.T}]; [{self.id_string}]; "
                    f"D_Loss: {d_loss / self.args.meta_epochs}; G_loss: {loss_g.item()}; "
                    f"D_Grad: {d_grad / self.args.meta_epochs};")


class FLClient(AbstractTrainer):
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
        z = torch.randn((100, 100)).to(device)
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

    def base_training_loop(self):
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
                    f"D_Loss: {d_loss / self.args.meta_epochs}; G_loss: {g_loss / self.args.meta_epochs}; "
                    f"D_Grad: {d_grad / self.args.meta_epochs}; G_grad: {g_grad / self.args.meta_epochs}")

    def load_dict(self, para):
        SerializationTool.deserialize_model(self.generator, para[0])
        SerializationTool.deserialize_model(self.discriminator, para[1])

    def state_dict(self):
        return (SerializationTool.serialize_model(self.generator),
                SerializationTool.serialize_model(self.discriminator))


import pickle as pkl

all_groups = []

def init_groups(size, cls_freq_wrk):
    """
	Initialization of all distributed groups for the whole training process. We do this in advance so as not to hurt the performance of training.
	The server initializes the group and send it to all workers so that everybody can agree on the working group at some round.
	Args
		size		The total number of machines in the current setup
		cls_freq_wrk	The frequency of samples of each class at each worker. This is used when the "sample" option is chosen. Otherwise, random sampling is applied and this parameter is not used.
    """
    global all_groups
    done = False
    gp_size = max(1, int(0.2 * (size)))
    # If opt.sample is set, use the smart sampling, i.e., based on frequency of samples of each class at each worker. Otherwise, use random sampling

    # 2D array that records if class i exists at worker j or not
    wrk_cls = [[False for i in range(num_class)] for j in range(size)]
    cls_q = [Queue(maxsize=size) for _ in range(num_class)]
    for i, cls_list in enumerate(cls_freq_wrk):
        wrk_cls[i] = [True if freq != 0 else False for freq in cls_list]
    for worker, class_list in enumerate(reversed(wrk_cls)):
        for cls, exist in enumerate(class_list):
            if exist:
                cls_q[cls].put(size - worker - 1)
    # This array counts the number of samples (per class) taken for training so far. The algorithm will try to make the numbers in this array as equal as possible
    taken_count = [0 for i in range(num_class)]
    while not done:
        visited = [False for i in range(size)]  # makes sure that we take any worker only once in the group
        g = []
        for _ in range(gp_size):
            # Choose class (that is minimum represnted so far)...using "taken_count" array
            cls = np.where(taken_count == np.amin(taken_count))[0][0]
            if int(cls) in [1, 3, 7, 9]:
                pass
            assert cls >= 0 and cls <= len(taken_count)
            # Choose a worker to represnt that class...using wrk_cls and visited array
            done_q = False
            count = 0
            while not done_q:
                wrkr = cls_q[cls].get()
                assert wrk_cls[wrkr][cls]
                if not visited[wrkr] and wrk_cls[wrkr][cls]:
                    # Update the state: taken_count and visited
                    g.append(wrkr)
                    taken_count += cls_freq_wrk[wrkr]
                    visited[wrkr] = True
                    done_q = True
                cls_q[cls].put(wrkr)
                count += 1
                if count == size:  # Such an optimal assignment does not exist
                    done_q = True
        all_groups.append(g)
        if len(all_groups) >= args.T:
            done = True
    return all_groups

import concurrent.futures

def main_loop():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.N)
    # read the dataset
    datasets = EMNIST("data", args.N, not args.noniid, dataset=args.dataset)
    # initialized the clients and twins
    global s_k, clients
    with open(f'{log_dir}data_distribution.npy', 'wb') as f:
        pkl.dump(datasets.datasets_index, f)
    print("Generate the list of sample clients")
    if args.model == "fegan":
        server = FLServer(args)
        clients = [FLClient(dataset, args, i) for i, dataset in enumerate(datasets)]
        datasets_targets = datasets.datasets_index
        y = torch.zeros(num_class)
        cls_freq_wrks = []
        s_k = []
        for target in datasets_targets:
            cls_freq_wrk = torch.zeros(num_class)
            for c in range(num_class):
                cls_freq_wrk[c] += (target == c).sum()
            y += cls_freq_wrk
            cls_freq_wrks.append(cls_freq_wrk)

        for i in range(args.N):
            s_k.append(stats.entropy(cls_freq_wrks[i] / cls_freq_wrks[i].sum(dim=-1), y / y.sum(dim=-1)) *
                       (cls_freq_wrks[i].sum(dim=-1) / y.sum(dim=-1)))
        init_groups(args.N, np.stack(cls_freq_wrks))
    else:
        clients = [MDClient(dataset, args, i) for i, dataset in enumerate(datasets)]
        server = MDServer(args)

    def client_train_step(t):
        if args.model == "fegan":
            futures = [executor.submit(clients[i].train, t) for i in all_groups[t]]
        else:
            futures = [executor.submit(clients[i].train, t) for i in range(args.N)]
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)



    def aggregate(t, init=False):
        if args.model == 'fegan':
            if init:
                chosen_clients = [j for j in range(args.N)]
            else:
                chosen_clients = all_groups[t-1]
            weights = []
            for j in chosen_clients:
                server.cache.append(clients[j].state_dict())
                weights.append(1 / args.N if init else s_k[j])
            weights = torch.exp(torch.tensor(weights))
            weights /= weights.sum()
            aggr_para = server.aggregate(weights)
            for j in chosen_clients:
                clients[j].load_dict(aggr_para)
        else:
            for j in range(args.N):
                if len(clients[j].cache) > 0:
                    server.cache.append(clients[j].get_grad())
            pkg = server.aggregate()
            k = len(pkg)
            for j in range(args.N):
                clients[j].load_fake(pkg[j % k], pkg[(j + 1) % k])


    def share_para():
        for i in range(args.N):
            if clients[i].is_a_epoch:
                clients[i].is_a_epoch = False
                chosen_client = np.random.choice([j for j in range(args.N)], size=1, replace=False)[0]
                temp = clients[i].state_dict()
                clients[i].load_dict(clients[chosen_client].state_dict())
                clients[chosen_client].load_dict(temp)

    def checkpoint_step():
        for i in range(args.N):
            clients[i].checkpoint_step()
            clients[i].validate_run()
        if args.model == "mdgan":
            server.checkpoint_step()
            server.validate_run()

    start = clients[0].eps
    print("start to simulate...")
    if args.model == "fegan":
        for t in range(start, args.T+1):
            if t % args.check_every == 0:
                print(f"Validate [{t}/{args.T}]")
                checkpoint_step()
            aggregate(t, True if t == 0 else False)
            client_train_step(t)

    else:
        for t in range(start, args.T+1):
            if t % args.check_every == 0:
                checkpoint_step()
            aggregate(t, True if t == 0 else False)
            client_train_step(t)
            share_para()

if __name__ == "__main__":
    main_loop()
