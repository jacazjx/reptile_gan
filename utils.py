import os
import re


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable

from fid_score import calculate_activation_statistics, calculate_frechet_distance
from inception import InceptionV3


def calc_gradient_penalty(discriminator, real_batch, fake_batch, device="cuda"):
    epsilon = torch.rand(real_batch.shape[0], 1, device=device)
    interpolates = epsilon.view(-1, 1, 1, 1) * real_batch + (1 - epsilon).view(-1, 1, 1, 1) * fake_batch
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates, _ = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

def cal_hidden_distance(src, dest):
    pass

def cal_projection(u, v):
    inner_product = torch.dot(u, v)

    # 计算 v 的模长的平方
    v_norm_squared = torch.norm(v) ** 2

    # 计算投影向量
    p = (inner_product / v_norm_squared) * v

    return torch.norm(p)


def cal_dist_euclidean(u, v):

    return torch.linalg.norm(u - v)


def cal_dist_manhattan(u, v):
    res = []
    for t1, t2 in zip(u, v):
        res.append(torch.sum(torch.abs(t1 - t2)))
    return torch.stack(res, dim=0)


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x

def normalize_data(data):
    data *= 2
    data -= 1
    return data


def unnormalize_data(data):
    data += 1
    data /= 2
    return data

def split_tensor(tensor):
    for t in tensor:
        yield t


# Those two functions are taken from torchvision code because they are not available on pip as of 0.2.0
def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def wassertein_loss(inputs, targets):
    return torch.mean(inputs * targets)


def kl_loss(src, dest):
    if not src.dim() == dest.dim():
        raise ValueError("The dim of two tensor is not same")

    if src.dim() == 1:
        log_probs = torch.nn.functional.log_softmax(src, dim=-1)
        probs = torch.softmax(dest, dim=-1)
        kl_distances = F.kl_div(log_probs, probs, reduction='sum')
        return kl_distances
    else:
        kl = []
        for s, d in zip(src, dest):
            log_probs = torch.nn.functional.log_softmax(s, dim=-1)
            probs = torch.softmax(d, dim=-1)
            kl.append(F.kl_div(log_probs, probs, reduction='sum'))
        return torch.tensor(kl)


def find_latest_file(folder):
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        return max(files)[1]
    else:
        return None


fid_tool = InceptionV3()


def calcu_fid(src, dest):
    mu_s, si_s = calculate_activation_statistics(src.cpu(), fid_tool, batch_size=src.shape[0])
    mu_d, si_d = calculate_activation_statistics(dest.cpu(), fid_tool, batch_size=dest.shape[0])
    fid = calculate_frechet_distance(mu_s, si_s, mu_d, si_d)
    return fid

def get_optimizer(net, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer

class SerializationTool(object):
    @staticmethod
    def serialize_model_gradients(model: torch.nn.Module) -> torch.Tensor:
        gradients = [param.grad.data.view(-1) for param in model.parameters()]
        m_gradients = torch.cat(gradients)
        m_gradients = m_gradients.cpu()
        return m_gradients

    @staticmethod
    def serialize_model(model: torch.nn.Module, position="para") -> torch.Tensor:
        """Unfold model parameters

        Unfold every layer of model, concate all of tensors into one.
        Return a `torch.Tensor` with shape (size, ).

        Args:
            model (torch.nn.Module): model to serialize.
        """
        if position == "para":
            parameters = [param.data.view(-1) for param in model.parameters()]
            m_parameters = torch.cat(parameters)
            m_parameters = m_parameters.cpu()
        elif position == "grad":
            parameters = [param.grad.data.view(-1) for param in model.parameters()]
            m_parameters = torch.cat(parameters)
            m_parameters = m_parameters.cpu()
        else:
            raise ValueError(
                "Invalid deserialize position {}, require \"para\" or \"grad\" "
                .format(position))
        return m_parameters

    @staticmethod
    def deserialize_model(model: torch.nn.Module,
                          serialized_parameters: torch.Tensor,
                          mode="copy",
                          position="para"):
        """Assigns serialized parameters to model.parameters.
        This is done by iterating through ``model.parameters()`` and assigning the relevant params in ``grad_update``.
        NOTE: this function manipulates ``model.parameters``.

        Args:
            model (torch.nn.Module): model to deserialize.
            serialized_parameters (torch.Tensor): serialized model parameters.
            mode (str): deserialize mode. "copy" or "add".
            position (str): deserialize mode. "para" or "grad".
        """

        current_index = 0  # keep track of where to read from grad_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if mode == "copy":
                if position == "para":
                    parameter.data.copy_(
                        serialized_parameters[current_index:current_index + numel].view(size))
                elif position == "grad":
                    if parameter.grad is None:
                        if model.is_cuda():
                            parameter.grad = Variable(torch.zeros(size)).cuda()
                        else:
                            parameter.grad = Variable(torch.zeros(size))
                    parameter.grad.data.copy_(
                        serialized_parameters[current_index:current_index + numel].view(size))
                else:
                    raise ValueError(
                        "Invalid deserialize position {}, require \"para\" or \"grad\" "
                        .format(position))
            elif mode == "add":
                if position == "para":
                    parameter.data.add_(
                        serialized_parameters[current_index:current_index + numel].view(size))
                elif position == "grad":
                    parameter.grad.data.add_(
                        serialized_parameters[current_index:current_index + numel].view(size))
                else:
                    raise ValueError(
                        "Invalid deserialize position {}, require \"para\" or \"grad\" "
                        .format(position))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                    .format(mode))
            current_index += numel


def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)

def get_mApping(num, with_type='byclass'):
    """
    根据 mapping，由传入的 num 计算 UTF8 字符
    """
    if with_type == 'byclass':
        if num <= 9:
            return chr(num + 48)  # 数字
        elif num <= 35:
            return chr(num + 55)  # 大写字母
        else:
            return chr(num + 61)  # 小写字母
    elif with_type == 'letters':
        return chr(num + 64) + " / " + chr(num + 96)  # 大写/小写字母
    elif with_type == 'digits':
        return chr(num + 96)
    else:
        return num


if __name__ == '__main__':
    u = torch.randn([5, 128])
    v = torch.randn([5, 128])
    print(kl_loss(u, u))
    print("投影", F.cosine_similarity(u, v), F.cosine_similarity(u, u))
    print("曼哈顿距离", cal_dist_manhattan(u, u))
