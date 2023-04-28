import torch
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from datasets.EMNIST import EMNIST
from models import Generator
import torch.nn.functional as F
import numpy as np
from utils import SerializationTool
from MMD import MMD_loss
dir_data = "data"
classifier = resnet50(pretrained=True).cuda()
classifier.eval()
#emnist
# dir_model = "[ModelName=fegan]_[NumClient=10]_[Dataset=emnist]_[NumTask=5]_[IsNonIID=True]_[IsCondition=False]_[IsLayerNrom=False]_[IsFeatureExtra=False]_[ShareWay=kd]_[NumInnerLoop=20]_[Batch=20]"
# dir_model = "[ModelName=mdgan]_[NumClient=10]_[Dataset=emnist]_[NumTask=5]_[IsNonIID=True]_[IsCondition=False]_[IsLayerNrom=False]_[IsFeatureExtra=False]_[ShareWay=kd]_[NumInnerLoop=1]_[Batch=20]"
# dir_model = "[ModelName=twingan]_[NumClient=10]_[Dataset=emnist]_[NumTask=5]_[IsNonIID=True]_[IsCondition=True]_[IsLayerNrom=True]_[IsFeatureExtra=False]_[ShareWay=kd]_[NumInnerLoop=5]_[Batch=20]"
#mnist
# dir_model = "[ModelName=mdgan]_[NumClient=10]_[Dataset=mnist]_[NumTask=5]_[IsNonIID=False]_[IsCondition=False]_[IsLayerNrom=False]_[IsFeatureExtra=False]_[ShareWay=kd]_[NumInnerLoop=1]_[Batch=20]"
# dir_model = "[ModelName=fegan]_[NumClient=10]_[Dataset=mnist]_[NumTask=5]_[IsNonIID=False]_[IsCondition=False]_[IsLayerNrom=False]_[IsFeatureExtra=False]_[ShareWay=kd]_[NumInnerLoop=5]_[Batch=20]"
# dir_model = "[ModelName=twingan]_[NumClient=10]_[Dataset=mnist]_[NumTask=5]_[IsNonIID=False]_[IsCondition=True]_[IsLayerNrom=True]_[IsFeatureExtra=False]_[ShareWay=kd]_[NumInnerLoop=5]_[Batch=20]"

mmd_loss = MMD_loss().cuda()

paras = {}
para_json = dir_model
for p in para_json.split("_"):
    ps = p[1:-1].split("=")
    if str.isdigit(ps[1]):
        paras[ps[0]] = int(ps[1])
    elif ps[1] == "True":
        paras[ps[0]] = True
    elif ps[1] == "False":
        paras[ps[0]] = False
    else:
        paras[ps[0]] = ps[1]

niid = paras["IsNonIID"]
dataset_name = paras['Dataset']
datasets = EMNIST(dir_data, paras["NumClient"], iid=not niid, dataset=dataset_name)
num_classes = {
    "emnist": 62,
    "mnist": 10,
}
num_clients = paras['NumClient']
# twin_gan
G = Generator(num_classes[paras["Dataset"]], 100, LayerNorm=paras["IsLayerNrom"], Condition=paras["IsCondition"]).cuda()
all_kl = []
if paras["ModelName"] == "twingan":
    for n in range(num_clients):
        G.load_state_dict(torch.load("checkpoint/" + dir_model + f"/Client_{n}")["model"]["G"])
        G.eval()
        dataset = datasets[n]
        loader = DataLoader(dataset.get_dataset(), batch_size=100)
        sum = 0
        corr = 0
        wrong = 0
        kl_mean = []
        with torch.no_grad():
            for d, l in loader:
                if d.shape[0] < 100:
                    break
                d = d.expand(-1, 3, -1, -1).cuda()
                l = l.cuda()
                sum += d.shape[0]
                z = torch.randn((100, 100)).cuda()
                d_f = G(z, l).expand(-1, 3, -1, -1)
                pre_r = classifier(d)
                pre_f = classifier(d_f)
                kl = F.kl_div(torch.log_softmax(pre_f, dim=-1), torch.softmax(pre_r, dim=-1), reduction="batchmean")
                print(kl)
                # 把KL的值存放进文件中
                kl_mean.append(kl.cpu().numpy())
                # z = torch.randn((100, 100)).cuda()
                # l = l.cuda()
                # d_f = G(z, l).view(100, -1)
                # d = d.cuda().view(100, -1)
                # loss = mmd_loss(d_f, d)
                # print(loss)
                # kl_mean.append(loss.cpu().numpy())

        all_kl.append(np.mean(kl_mean))
    np.save(f"checkpoint/twin_gan_result.npy", np.array(all_kl), )
elif paras["ModelName"] == "mdgan":
    weight = torch.load("checkpoint/" + dir_model + "/Server")["model"]["G"]
    weight = torch.cat([w.view(-1) for w in weight.values()])
    SerializationTool.deserialize_model(G, weight)
    G.eval()
    for n in range(num_clients):
        dataset = datasets[n]
        loader = DataLoader(dataset.get_dataset(), batch_size=100)
        sum = 0
        corr = 0
        wrong = 0
        kl_mean = []
        with torch.no_grad():
            for d, l in loader:
                if d.shape[0] < 100:
                    break
                d = d.expand(-1, 3, -1, -1).cuda()
                l = l.cuda()
                sum += d.shape[0]
                z = torch.randn((100, 100)).cuda()
                d_f = G(z, l).expand(-1, 3, -1, -1)
                pre_r = classifier(d)
                pre_f = classifier(d_f)
                kl = F.kl_div(torch.log_softmax(pre_f, dim=-1), torch.softmax(pre_r, dim=-1), reduction="batchmean")
                print(kl)
                # 把KL的值存放进文件中
                kl_mean.append(kl.cpu().numpy())
                # z = torch.randn((100, 100)).cuda()
                # l = l.cuda()
                # d_f = G(z, l).view(100, -1)
                # d = d.cuda().view(100, -1)
                # loss = mmd_loss(d_f, d)
                # print(loss)
                # kl_mean.append(loss.cpu().numpy())
        all_kl.append(np.mean(kl_mean))
    np.save(f"checkpoint/md_gan_result.npy", np.array(all_kl), )

# fegan
# dir_model = "checkpoint/[ModelName=fegan]_[NumClient=10]_[Dataset=emnist]_[NumTask=5]_[IsNonIID=True]_[IsCondition=False]_[IsLayerNromFalse]_[IsFeatureExtra=False]_[ShareWay=kd]_[NumInnerLoop=20]_[Batch=20]"
# G = Generator(10, 100, hidden_dim=16, LayerNorm=False, Condition=False).cuda()
elif paras["ModelName"] == "fegan":
    for n in range(num_clients):
        dataset = datasets[n]
        weight = torch.load("checkpoint/" + dir_model + f"/Client_{n}")["model"]["G"]
        weight = torch.cat([w.view(-1) for w in weight.values()])
        SerializationTool.deserialize_model(G, weight)
        G.eval()
        loader = DataLoader(dataset.get_dataset(), batch_size=100)
        sum = 0
        corr = 0
        wrong = 0
        kl_mean = []
        with torch.no_grad():
            for d, l in loader:
                if d.shape[0] < 100:
                    break
                d = d.expand(-1, 3, -1, -1).cuda()
                l = l.cuda()
                sum += d.shape[0]
                z = torch.randn((100, 100)).cuda()
                d_f = G(z, l).expand(-1, 3, -1, -1)
                pre_r = classifier(d)
                pre_f = classifier(d_f)
                kl = F.kl_div(torch.log_softmax(pre_f, dim=-1), torch.softmax(pre_r, dim=-1), reduction="batchmean")
                print(kl)
                # 把KL的值存放进文件中
                kl_mean.append(kl.cpu().numpy())
                # z = torch.randn((100, 100)).cuda()
                # l = l.cuda()
                # d_f = G(z, l).view(100, -1)
                # d = d.cuda().view(100, -1)
                # loss = mmd_loss(d_f, d)
                # print(loss)
                # kl_mean.append(loss.cpu().numpy())
        all_kl.append(np.mean(kl_mean))
    np.save(f"checkpoint/fe_gan_result.npy", np.array(all_kl), )


# dir_model = "checkpoint/Gossip 算法[ModelName=twingan]_[NumClient=10]_[Dataset=mnist]_[NumTask=5]_[IsNonIID=False]_[IsCondition=True]_[IsFeatureExtra=False]_[ShareWay=kd]_[NumInnerLoop=5]_[Batch=20]"
# G = Generator(10, 100, hidden_dim=16, LayerNorm=False, Condition=True).cuda()
# for n in range(num_clients):
#     dataset = datasets[n]
#     weight = torch.load(dir_model+f"/Client_{n}")["model"]["G"]
#     weight = torch.cat([w.view(-1) for w in weight.values()])
#     SerializationTool.deserialize_model(G, weight)
#     G.eval()
#     loader = DataLoader(dataset.get_dataset(), batch_size=100)
#     sum = 0
#     corr = 0
#     wrong = 0
#     kl_mean = []
#     with torch.no_grad():
#         for d, l in loader:
#             if d.shape[0] < 100:
#                 break
#             d = d.expand(-1, 3, -1, -1).cuda()
#             l = l.cuda()
#             sum += d.shape[0]
#             z = torch.randn((100, 100)).cuda()
#             d_f = G(z, l).expand(-1, 3, -1, -1)
#             pre_r = classifier(d)
#             pre_f = classifier(d_f)
#             kl = F.kl_div(torch.log_softmax(pre_f, dim=-1), torch.softmax(pre_r, dim=-1), reduction="batchmean")
#             print(kl)
#             # 把KL的值存放进文件中
#             kl_mean.append(kl.cpu().numpy())
#     all_kl.append(np.mean(kl_mean))
# np.save(f"checkpoint/gossip_result.npy", np.array(all_kl), )