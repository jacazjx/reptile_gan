import os
import re
import torch
import torch.nn.functional as F
def cal_projection(u, v):
    res = []
    for t1, t2 in zip(u, v):
        # 计算内积
        inner_product = torch.dot(t1, t2)

        # 计算 v 的模长的平方
        v_norm_squared = torch.norm(t2) ** 2

        # 计算投影向量
        p = (inner_product / v_norm_squared) * t2

        res.append(torch.norm(p))

    return torch.stack(res, dim=0)

def cal_dist_euclidean(u, v):
    res = []
    for t1, t2 in zip(u, v):
        res.append(torch.linalg.norm(t1 - t2))
    return torch.stack(res, dim=0)

def cal_dist_manhattan(u, v):
    res = []
    for t1, t2 in zip(u, v):
        res.append(torch.sum(torch.abs(t1 - t2)))
    return torch.stack(res, dim=0)

def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x

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
    log_probs = torch.nn.functional.log_softmax(src, dim=-1)
    probs = torch.softmax(dest, dim=-1)

    kl_distances = F.kl_div(log_probs, probs, reduction='none')
    return kl_distances


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


class SerializationTool(object):
    @staticmethod
    def serialize_model_gradients(model: torch.nn.Module) -> torch.Tensor:
        gradients = [param.grad.data.view(-1) for param in model.parameters()]
        m_gradients = torch.cat(gradients)
        m_gradients = m_gradients.cpu()
        return m_gradients

    @staticmethod
    def serialize_model(model: torch.nn.Module) -> torch.Tensor:
        """Unfold model parameters

        Unfold every layer of model, concate all of tensors into one.
        Return a `torch.Tensor` with shape (size, ).

        Args:
            model (torch.nn.Module): model to serialize.
        """

        parameters = [param.data.view(-1) for param in model.parameters()]
        m_parameters = torch.cat(parameters)
        m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def deserialize_model(model: torch.nn.Module,
                          serialized_parameters: torch.Tensor,
                          mode="copy"):
        """Assigns serialized parameters to model.parameters.
        This is done by iterating through ``model.parameters()`` and assigning the relevant params in ``grad_update``.
        NOTE: this function manipulates ``model.parameters``.

        Args:
            model (torch.nn.Module): model to deserialize.
            serialized_parameters (torch.Tensor): serialized model parameters.
            mode (str): deserialize mode. "copy" or "add".
        """

        current_index = 0  # keep track of where to read from grad_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if mode == "copy":
                parameter.data.copy_(
                    serialized_parameters[current_index:current_index +
                                                        numel].view(size))
            elif mode == "add":
                parameter.data.add_(
                    serialized_parameters[current_index:current_index +
                                                        numel].view(size))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                    .format(mode))
            current_index += numel


if __name__ == '__main__':
    a = torch.randn((5, 128))
    b = torch.randn((5, 128))

    print(kl_loss(a, b))
    s = 0
    for aa, bb in zip(a, b):
        s += kl_loss(aa.view(1, aa.shape[0]), bb.view(1, bb.shape[0]))
    print(s)

    u = torch.randn([5, 128])
    v = torch.randn([5, 128])

    print("投影", cal_projection(u, v))
    print("欧几里得距离", cal_dist_euclidean(u, v))
    print("曼哈顿距离", cal_dist_manhattan(u, v))





