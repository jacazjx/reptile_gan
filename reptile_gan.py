import torch.optim
import torchvision
from torch import nn
import torch.nn.functional as F
from datasets.EMNIST import EMNIST
from datasets.FEMNIST import FEMNIST
from datasets.MNIST import *
from torch.utils.data import DataLoader
from logger import Logger
from models import Generator, Discriminator
from torch.autograd import Variable
from inception import InceptionV3
from fid_score import calculate_frechet_distance, calculate_activation_statistics
from utils import wassertein_loss, calc_gradient_penalty, cal_dist_euclidean, get_mApping, cal_projection, kl_loss, \
    SerializationTool

Tensor = torch.cuda.FloatTensor

outer_loops = 150000
inner_loop = 10
meta_lr = 0.2
state_g = None
state_d = None
N = 1
K = 10
cross_entropy = nn.BCEWithLogitsLoss()
fid_tool = InceptionV3()


def get_loss(prediction, labels):
    return cross_entropy(prediction, labels) # cross_entropy()


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def do_learning(net_g, net_d, optimizer_g, optimizer_d, train_iter, inner_loop):
    net_d.train()
    net_g.train()

    for iteration in range(inner_loop):
        # Sample minibatch
        x_r, lbs = next(train_iter)
        x_r = Variable(x_r.type(Tensor))
        z = Variable(Tensor(np.random.normal(0, 1, (x_r.shape[0], 100))))
        fake = Variable(Tensor(x_r.shape[0], 1).fill_(-1.0), requires_grad=False)
        real = Variable(Tensor(x_r.shape[0], 1).fill_(1.0), requires_grad=False)

        x_g = net_g(z)
        loss_d = (get_loss(net_d(x_g.detach()), fake) + get_loss(net_d(x_r), real)) / 2 + calc_gradient_penalty(net_g, x_r, x_g, device="cuda")
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        y_g = net_d(x_g)
        loss_g = get_loss(y_g, real)
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

# def do_eva

def get_optimizer(net, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer

def calcu_fid(src, dest, model):
    mu_s, si_s = calculate_activation_statistics(src.cpu(), model, batch_size=src.shape[0])
    mu_d, si_d = calculate_activation_statistics(dest.cpu(), model, batch_size=dest.shape[0])
    fid = calculate_frechet_distance(mu_s, si_s, mu_d, si_d)
    return fid

def save_fig(images, lbl, name):
    # 转换像素值范围
    # images = (images + 1) / 2

    # 把 25 张图片打包成一个张量
    # images = images.view(25, 1, 28, 28)

    for i, l in enumerate(lbl):
        if i % 5 == 0:
            name += "_"
        name += get_mApping(l)

    torchvision.utils.save_image(images, f"./checkpoint/{outer_loop}_{name}.png",
                                 nrow=5, normalize=True)

def discriminator_loss(real_hidden, fake_hidden):
    c_r = real_hidden.mean(dim=0)
    c_f = fake_hidden.mean(dim=0)

    return -1 * cal_dist_euclidean(c_r, c_f).mean()


def generator_loss(h_r, h_f):
    c_f = h_f.mean(dim=0)
    return cal_dist_euclidean(c_f, h_r).mean()


# Create tensorboard logger
logger = Logger("log")

dset = EMNIST("data", 10, True, dataset="emnist")
dset = dset[0]

van_dset = make_infinite(DataLoader(dset.get_dataset(), batch_size=20, shuffle=True))
van_g = Generator(62, LayerNorm=True, Condition=True).cuda()
van_d = Discriminator(62, LayerNorm=True, Condition=True).cuda()
meta_optimizer_d = torch.optim.Adam(van_d.parameters(), lr=meta_lr)
van_optimizer_d = torch.optim.Adam(van_d.parameters(), lr=0.0002, betas=(0.5, 0.999))
van_optimizer_g = torch.optim.Adam(van_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
state_g = None

for outer_loop in range(outer_loops):
    
    batch_size = 5
    cordinator = []
    tasks = dset.get_random_tasks(5)
    meta_para = []
    loss_d = 0
    for t in tasks:  # take tasks
        # data = iter(DataLoader(dset.get_n_task(tasks, batch_size), batch_size, shuffle=True))
        for i in range(5):  # inner_loop
            x_r, l_r = next(van_dset)
            l_r = Variable(l_r.to("cuda"))
            x_r = Variable(x_r.type(Tensor))
            z = torch.randn((x_r.shape[0], 100)).to("cuda")
            x_g = van_g(z, l_r)
            fake = Variable(Tensor(x_r.shape[0], 1).fill_(0.0), requires_grad=False)
            real = Variable(Tensor(x_r.shape[0], 1).fill_(1.0), requires_grad=False)

            # p_f, p_r = van_d(x_g.detach()), van_d(x_r)
            p_f, p_r = van_d(x_g.detach(), l_r), van_d(x_r, l_r)
            loss_adv = (get_loss(p_f, fake) + get_loss(p_r, real)) * 0.5
            # loss_gp = calc_gradient_penalty(van_d, x_r, x_g.detach())
            # loss_kl = kl_loss(h_f.mean(dim=0), h_r.mean(dim=0)).mean()       
            #loss_ot = ot_distance(h_r.mean(dim=0), h_f.mean(dim=0))
            loss = loss_adv# - loss_ot * 0.001
            loss_d += loss.item()
            # loss = loss_adv - loss_kl * 0.001
            van_optimizer_d.zero_grad()
            loss.backward()
            van_optimizer_d.step()
        # grad = van_para - SerializationTool.serialize_model(van_d)
        # SerializationTool.deserialize_model(van_d, grad, position="grad")
        # meta_optimizer_d.step()
        # cordinator.append((l_r[0], h_r.mean(dim=0).detach()))
    loss_g = 0
    for t in np.random.choice(10, 5, replace=False): # tasks
        meta_g = van_g.clone()
        meta_opti_g = get_optimizer(meta_g, state_g)
        for i in range(5):
            # lbls, h_r = cordinator[i]
            l_g = torch.zeros(4, dtype=torch.int32).fill_(t).to("cuda")
            real = Variable(Tensor(4, 1).fill_(1.0), requires_grad=False)
            z = torch.randn((4, 100)).to("cuda")
            x_g = meta_g(z, l_g)
            p_f = van_d(x_g, l_g)
            loss = get_loss(p_f, real) #+ loss_ot * 0.001
            loss_g += loss.item()
            meta_opti_g.zero_grad()
            loss.backward()
            meta_opti_g.step()
        van_g.point_grad_to(meta_g)
        van_optimizer_g.step()
        state_g = meta_opti_g.state_dict()
    print(f"[{outer_loop}/{outer_loops}] [D_loss: {loss_d / 25}]; [G_loss: {loss_g/25}]")
    # lbls = torch.from_numpy(dset.get_random_tasks(batch_size)).to("cuda")
    # # lbls = lbls.repeat(1, batch_size).squeeze(0).to("cuda")
    # z = Variable(Tensor(np.random.normal(0, 1, (batch_size, 100))))
    # real = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
    # van_optimizer_g.zero_grad()
    # x_g = van_g(z, lbls)
    # p_f = van_d(x_g)
    # loss_g = get_loss(p_f, real)  # + kl_loss(h_f.mean(dim=0), h_r).mean() * 0.001
    # loss_g.backward()
    # van_optimizer_g.step()

    if outer_loop % 100 == 0:
        # meta_g.eval()
        van_g.eval()

        real_img = make_infinite(DataLoader(dset.get_random_test_task(100), batch_size=100, shuffle=True))
        real_img, lbl = next(real_img)
        z = Tensor(np.random.normal(0, 1, (100, 100)))
        # meta_x = meta_g(z)
        # loss = meta_d(meta_x)
        # save_fig(meta_x.cpu(), "meta")
        van_x = van_g(z, lbl.cuda())
        tasks = dset.get_random_tasks(5)
        tasks = torch.tensor(tasks).view(-1, 1).expand(5, 5).reshape(25)
        save_fig(van_g(Tensor(np.random.normal(0, 1, (25, 100))), tasks.cuda()).cpu(), tasks, "van")
        # save_fig(real_img, "ground_truth")

        # fid_meta = calcu_fid(real_img, meta_x, fid_tool)
        # fid_van = calcu_fid(real_img, van_x, fid_tool)
        # logger.info(f"round{outer_loop}, VAN FID:{fid_van}")

        # test_model(meta_g, meta_d, state_g, state_d)
        # test_model(van_g, van_d, van_optimizer_g.state_dict(), van_optimizer_d.state_dict(), "VAN Test")


        # meta_g.train()
        van_g.train()