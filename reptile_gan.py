import torch.optim
import torchvision
from torch import nn
from datasets.MNIST import *
from torch.utils.data import DataLoader
from logger import Logger
from models import Generator, Discriminator
from torch.autograd import Variable
from inception import InceptionV3
from fid_score import calculate_frechet_distance, calculate_activation_statistics
Tensor = torch.cuda.FloatTensor

outer_loops = 150000
inner_loop = 10
meta_lr = 1
state_g = None
state_d = None
N = 1
K = 10
cross_entropy = nn.MSELoss().cuda()
fid_tool = InceptionV3()


def get_loss(prediction, labels):
    return cross_entropy(prediction, labels)


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
        fake = Variable(Tensor(x_r.shape[0], 1).fill_(0.0), requires_grad=False)
        real = Variable(Tensor(x_r.shape[0], 1).fill_(1.0), requires_grad=False)

        x_g = net_g(z)
        loss_d = (get_loss(net_d(x_g.detach()), fake) + get_loss(net_d(x_r), real)) / 2
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

def save_fig(images, name):
    # 转换像素值范围
    # images = (images + 1) / 2

    # 把 25 张图片打包成一个张量
    # images = images.view(25, 1, 28, 28)

    torchvision.utils.save_image(images[np.random.choice(range(len(images)), 25, replace=False)], f"./checkpoint/{outer_loop}_{name}.png",
                                 nrow=5, normalize=True)


def test_model(meta_g, meta_d, opti_g, opti_d, name="Meta Test"):
    copy_g = meta_g.clone()
    copy_d = meta_d.clone()

    opti_g = get_optimizer(copy_g, opti_g)
    opti_d = get_optimizer(copy_d, opti_d)

    test_data = make_infinite(DataLoader(dset.get_random_test_task(5), batch_size=1, shuffle=True))
    # Meta Training
    do_learning(copy_g, copy_d, opti_g, opti_d, test_data, 50)

    # Meta Testing
    test_data = make_infinite(DataLoader(dset.get_random_test_task(25), batch_size=25, shuffle=True))
    z = Tensor(np.random.normal(0, 1, (25, 100)))
    x_r = next(test_data)[0].type(Tensor)
    x_g = copy_g(z)

    fid = calcu_fid(x_r, x_g, fid_tool)

    save_fig(x_g, name)
    del copy_g, copy_d, opti_g, opti_d, test_data, z, x_r, x_g
    logger.info(f"{name} FID: {fid}")



# Create tensorboard logger
logger = Logger("log")

dset = MNISTTasks(root="./data")
meta_g = Generator().cuda()
meta_d = Discriminator().cuda()

meta_optimizer_g = torch.optim.SGD(meta_g.parameters(), lr=meta_lr)
meta_optimizer_d = torch.optim.SGD(meta_d.parameters(), lr=meta_lr)

van_dset = make_infinite(DataLoader(dset.copy(), batch_size=10, shuffle=True))
van_g = Generator().cuda()
van_d = Discriminator().cuda()
van_optimizer_d = torch.optim.Adam(van_d.parameters(), lr=0.0002, betas=(0.5, 0.999))
van_optimizer_g = torch.optim.Adam(van_g.parameters(), lr=0.0002, betas=(0.5, 0.999))

for outer_loop in range(outer_loops):

    # # Meta Learning
    meta_lr = 0.2 * (1. - outer_loop / float(outer_loops))
    set_learning_rate(meta_optimizer_g, meta_lr)
    set_learning_rate(meta_optimizer_d, meta_lr)

    # Clone model
    net_g = meta_g.clone()
    optimizer_g = get_optimizer(net_g, state_g)
    net_d = meta_d.clone()
    optimizer_d = get_optimizer(net_d, state_d)

    task = make_infinite(DataLoader(dset.get_random_train_task(N * K), batch_size=10, shuffle=True))

    do_learning(net_g, net_d, optimizer_g, optimizer_d, task, inner_loop)
    state_g = optimizer_g.state_dict()  # save optimizer state
    state_d = optimizer_d.state_dict()  # save optimizer state

    # Update slow net
    meta_g.point_grad_to(net_g)
    meta_optimizer_g.step()

    meta_d.point_grad_to(net_d)
    meta_optimizer_d.step()
    # -----------------------------------------------------------------------------------------------

# van_g = meta_g.clone()
# van_optimizer_g = get_optimizer(van_g, state_g)
# van_d = meta_d.clone()
# van_optimizer_d = get_optimizer(van_d, state_d)

# for outer_loop in range(1000, 10000):
    # VANILLA LEARNING
    x_r, lbs = next(van_dset)
    x_r = Variable(x_r.type(Tensor))
    z = Variable(Tensor(np.random.normal(0, 1, (x_r.shape[0], 100))))
    fake = Variable(Tensor(x_r.shape[0], 1).fill_(0.0), requires_grad=False)
    real = Variable(Tensor(x_r.shape[0], 1).fill_(1.0), requires_grad=False)

    van_optimizer_g.zero_grad()
    x_g = van_g(z)
    loss_g = get_loss(van_d(x_g), real)
    loss_g.backward()
    van_optimizer_g.step()


    loss_d = (get_loss(van_d(x_g.detach()), fake) + get_loss(van_d(x_r), real)) * 0.5
    van_optimizer_d.zero_grad()
    loss_d.backward()
    van_optimizer_d.step()



    if outer_loop % 100 == 0 and outer_loops != 0:
        meta_g.eval()
        van_g.eval()

        real_img = make_infinite(DataLoader(dset.copy(), batch_size=500, shuffle=True))
        real_img, _ = next(real_img)
        z = Tensor(np.random.normal(0, 1, (500, 100)))
        meta_x = meta_g(z)
        loss = meta_d(meta_x)
        # save_fig(meta_x.cpu(), "meta")
        van_x = van_g(z)
        # save_fig(van_x.cpu(), "van")


        # fid_meta = calcu_fid(real_img, meta_x, fid_tool)
        # fid_van = calcu_fid(real_img, van_x, fid_tool)
        # logger.info(f"round{outer_loop}, META FID:{fid_meta}, VAN FID:{fid_van}")

        test_model(meta_g, meta_d, state_g, state_d)
        test_model(van_g, van_d, van_optimizer_g.state_dict(), van_optimizer_d.state_dict(), "VAN Test")


        meta_g.train()
        van_g.train()