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
from dismeta import AbstractTrainer, num_class, args, logger, id_string, log_dir, device
Tensor = torch.cuda.FloatTensor


np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)



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
