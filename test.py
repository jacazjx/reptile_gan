
import torch
from torch.utils.data import WeightedRandomSampler
import pickle as pkl
# import numpy as np
# twingan = np.load("checkpoint/twin_gan_result.npy")
# # gossipgan = np.load("checkpoint/gossip_result.npy")
# fegan = np.load("checkpoint/fe_gan_result.npy")
# mdgan = np.load("checkpoint/md_gan_result.npy")
#
#
# # print("{} {}".format(np.around(np.mean(gossipgan), 2), np.std(gossipgan)/100))
#
# print("{} {}".format(np.around(np.mean(mdgan), 2), np.std(mdgan)/100))
# print("{} {}".format(np.around(np.mean(fegan), 2), np.std(fegan)/100))
# print("{} {}".format(np.around(np.mean(twingan), 2), np.std(twingan)/100))


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
config = {
    "font.family":'serif',
    "font.size": 10,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
# 设置总时间
total_time = 300

# 设置时间间隔
time_interval_1 = 2
time_interval_2 = 5
time_interval_3 = 20

# 计算通信时刻
communication_times_1 = list(range(0, total_time + 1, time_interval_1))
communication_times_2 = list(range(0, total_time + 1, time_interval_2))
communication_times_3 = list(range(0, total_time + 1, time_interval_3))

# 绘制图形
fig, axs = plt.subplots(3)
axs[0].eventplot(communication_times_1)
axs[0].set_ylabel('MD-GAN')
axs[1].eventplot(communication_times_2)
axs[1].set_ylabel('TwinGAN')
axs[2].eventplot(communication_times_3)
axs[2].set_ylabel('FeGAN')

# plt.title("三种算法的通信频率比较")
plt.savefig("图片5-7.jpg", dpi=300)



