import argparse

import concurrent.futures
import math
import os

from matplotlib import pyplot as plt

from MMD import MMD_loss
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
    calc_gradient_penalty, kl_loss, get_optimizer
from dismeta import AbstractTrainer, num_class, args, logger, id_string, log_dir, device

Tensor = torch.cuda.FloatTensor

np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)


class Client(AbstractTrainer):
    def __init__(self, dataset, args, idx):
        super().__init__(args, idx)
        self.state_g = None
        self.meta_state = None
        self.id_string = "Client_" + self.id_string
        self.dataset = dataset
        self.cache = []
        self.loader = make_infinite(DataLoader(dataset.get_dataset(), batch_size=self.args.batch, shuffle=True))
        self.state = None
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
        mmd = MMD_loss().cuda()
        torchvision.utils.save_image(x_r[np.random.choice(range(len(x_r)), 25, replace=False)],
                                     f"{log_dir}ground_truth.png",
                                     nrow=5, normalize=True)
        torchvision.utils.save_image(x_g[np.random.choice(range(len(x_g)), 25, replace=False)],
                                     f"{log_dir}{self.id_string}.png",
                                     nrow=5, normalize=True)

        logger.info(
            f"EPOCHS: [{self.eps}/{self.args.T}]; {self.id_string}; MMD_LOSS: {mmd(x_r.cuda().view(100, -1), x_g.cuda().view(100, -1))}; "
            f"Real Predition: {self.discriminator(x_r.type(Tensor), l_r.to(device)).mean(dim=0).cpu().item()}; "
            f"Fake Predition: {self.discriminator(x_g.type(Tensor), l_r.to(device)).mean(dim=0).cpu().item()}")

    def train(self, t):
        self.eps = t
        super().train()

    def meta_training_loop(self):
        self.excu_cache()
        self.generator.train()
        self.discriminator.train()
        tasks = self.dataset.get_random_tasks(self.args.num_tasks)  # 这一轮学了什么

        # for i, data in enumerate(self.loader):
        #     if i == self.args.meta_epochs:
        #         break
        #     self.batches += 1
        #     loss = self.inner_loop(self.generator, self.discriminator, data, self.opti_d)
        #     d_loss.append(loss[0])
        #     g_loss.append(loss[1])
        def inner_loop_d(g, d, data, opti):
            x_r, lbl = data
            x_r = x_r.to(device)
            lbl = lbl.to(device)
            z = torch.randn((x_r.shape[0], 100)).to(device)

            x_g = g(z, lbl)
            pre_f, pre_r = d(x_g, lbl), d(x_r, lbl)

            # loss_adv = -torch.mean(pre_r) + torch.mean(pre_f)
            loss_adv = self.loss(pre_f, torch.zeros(pre_f.shape[0]).view(-1, 1).to(device)) + \
                       self.loss(pre_r, torch.ones(pre_r.shape[0]).view(-1, 1).to(device))
            loss = loss_adv
            opti.zero_grad()
            loss.backward()
            opti.step()
            return torch.mean(torch.sigmoid(pre_r).log()).item()

        def inner_loop_g(g, d, task, opti):
            z = torch.randn((self.args.batch, 100)).to(device)
            l = torch.tensor(task).to(device)
            x_g = g(z, l)
            y_g = d(x_g, l)
            # loss = -torch.mean(y_g)
            real = torch.ones(x_g.shape[0]).view(-1, 1).to(device)
            loss = self.loss(y_g, real)
            opti.zero_grad()
            loss.backward()
            opti.step()
            return loss.item()

        d_loss = []
        g_loss = []

        for _ in range(self.args.meta_epochs):
            tasks = self.dataset.get_random_tasks(self.args.num_tasks)
            # meta_d = self.discriminator.clone()
            # meta_opti_d = get_optimizer(meta_d, self.meta_state)
            # samples = make_infinite(DataLoader(self.dataset.get_n_task(tasks, self.args.batch), batch_size=self.args.batch, shuffle=True))
            # for i in range(self.args.num_tasks):
            #     loss = inner_loop_d(self.generator, meta_d, next(samples), meta_opti_d)
            #     if i == self.args.meta_epochs - 1:
            #         d_loss.append(loss)
            # self.discriminator.point_grad_to(meta_d)  # 计算梯度
            # self.meta_state = meta_opti_d.state_dict()
            # self.opti_d.step()  # 更新参数
            loss = inner_loop_d(self.generator, self.discriminator, next(self.loader), self.opti_d)
            d_loss.append(loss)
            meta_g = self.generator.clone()
            meta_opti_g = get_optimizer(meta_g, self.state_g)
            for i in range(self.args.num_tasks):
                loss = inner_loop_g(meta_g, self.discriminator, np.repeat(tasks[i], self.args.batch), meta_opti_g)
                if i == self.args.num_tasks - 1:
                    g_loss.append(loss)
            self.state_g = meta_opti_g.state_dict()
            self.generator.point_grad_to(meta_g)
            self.opti_g.step()
        d_loss = np.array(d_loss).mean()
        g_loss = np.array(g_loss).mean()

        logger.info(f"EPOCHS: [{self.eps}/{self.args.T}]; [{self.id_string}]; "
                    f"D_Loss: {d_loss}; G_loss: {g_loss}; ")

    def base_training_loop(self):
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
                    f"D_Loss: {d_loss / self.args.meta_epochs}; G_loss: {g_loss / self.args.meta_epochs}; "
                    f"D_Grad: {d_grad / self.args.meta_epochs}; G_grad: {g_grad / self.args.meta_epochs}")

    def excu_cache(self):
        if self.args.aggr:
            global A
            if len(self.cache) > 0:
                weight = []
                l = torch.tensor(self.dataset.get_random_tasks(self.args.num_tasks)).cuda()
                z = torch.randn((self.args.num_tasks, 100)).cuda()
                x_g = self.generator(z, l)
                weight.append(torch.mean(torch.sigmoid(self.discriminator(x_g, l))))
                clone_d = self.discriminator.clone()
                for index, state_dict in self.cache:
                    SerializationTool.deserialize_model(clone_d, state_dict[1])
                    weight.append(torch.mean(torch.sigmoid(clone_d(x_g, l))))
                weight = torch.softmax(torch.tensor(weight), dim=-1)

                generator_sum = SerializationTool.serialize_model(self.generator) * weight[0]
                discriminator_sum = SerializationTool.serialize_model(self.discriminator) * weight[0]
                for state in zip(self.cache, iter(weight[1:])):
                    index, state_dict = state[0]
                    w = state[1]
                    generator_sum += state_dict[0] * w
                    discriminator_sum += state_dict[1] * w
                    A[self.idx][index] += (w - weight[0])
                SerializationTool.deserialize_model(self.generator, generator_sum)
                SerializationTool.deserialize_model(self.discriminator, discriminator_sum)
                self.cache = []
        else:
            self.set_state_dict(self.cache.pop())
            self.cache = []

    def set_state_dict(self, state_dict):
        SerializationTool.deserialize_model(self.generator, state_dict[0])
        SerializationTool.deserialize_model(self.discriminator, state_dict[1])

    def state_dict(self):
        return (SerializationTool.serialize_model(self.generator),
                SerializationTool.serialize_model(self.discriminator))


import pickle as pkl

A = []


def main_loop():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.N)
    # read the dataset
    datasets = EMNIST("data", args.N, True, dataset=args.dataset)
    # initialized the clients and twins
    clients = [Client(dataset, args, i) for i, dataset in enumerate(datasets)]
    pkl.dump(datasets.datasets_index, open(f'{log_dir}data_distribution.pkl', 'wb'))
    global A
    A = torch.ones((args.N, args.N)) - torch.eye(args.N)*1e9
    A /= (args.N - 1)

    def client_train_step(t):
        futures = [executor.submit(clients[i].train, t) for i in range(args.N)]
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

    def share_para():
        if args.aggr:
            num_choose = max(1, int(math.log2(args.N)))
            for i in range(args.N):
                chosen_clients = torch.topk(A[i], k=num_choose)[1]
                for j in chosen_clients:
                    clients[i].cache.append((j, clients[j].state_dict()))
        else:
            chosen = []
            for i in range(args.N):
                chosen_clients = np.random.choice([j for j in range(args.N) if j not in chosen], size=1, replace=False)
                for j in chosen_clients:
                    clients[j].cache.append(clients[j].state_dict())
                    chosen.append(j)

    def checkpoint_step(t):
        for i in range(args.N):
            clients[i].checkpoint_step()
            clients[i].validate_run()
            # Convert matrix A to a numpy array
        A_np = np.array(A)
        for i in range(args.N):
            A_np[i][i]=0
        # Create heatmap plot
        plt.imshow(A_np, cmap='hot', interpolation='nearest')
        plt.colorbar()  # add colorbar
        plt.xlabel('Client Index')  # add x-axis label
        plt.ylabel('Client Index')  # add y-axis label
        plt.title(f"Heatmap of Similarity Matrix")  # add title
        # Save plot to log folder
        plt.savefig(f"{log_dir}heatmap_{t}.png")
        plt.close()

    for t in range(clients[0].eps, args.T):
        if t % args.check_every == 0:
            checkpoint_step(t)
        client_train_step(t)
        share_para()


if __name__ == "__main__":
    main_loop()
