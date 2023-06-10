import os
from typing import Any

import PIL
import numpy as np
import torch
import torchvision
from torch.utils import data
from PIL import Image
from datasets.MNIST import MNISTTasks
from torchvision import datasets, transforms
from fedlab.utils.dataset import BasicPartitioner
import pickle

channel = 1

class Partitioner(BasicPartitioner):
    num_features = 784

    def __init__(self, targets, num_clients, dataset="emnist", partition: str = 'iid',
                 dir_alpha: Any = None,
                 major_classes_num: int = None,
                 verbose: bool = True,
                 seed: Any = None):
        if dataset == "emnist":
            self.num_classes = 62
        elif dataset == "mnist":
            self.num_classes = 10
        super(Partitioner, self).__init__(targets=targets, num_clients=num_clients,
                                                partition=partition,
                                                dir_alpha=dir_alpha,
                                                major_classes_num=major_classes_num,
                                                verbose=verbose,
                                                seed=seed)

class EMNIST(object):
    max_clients = 200
    def __init__(self, root, num_clients, iid=True, dataset="emnist"):
        curPath = os.path.abspath(os.path.dirname(__file__))
        rootPath = curPath[:curPath.find("reptile_gan\\") + len("reptile_gan\\")]
        root = rootPath + root
        if dataset == "emnist":
            self.dataset = datasets.EMNIST(root, split='byclass', train=True, download=True)
            self.max_clients = 2000
        elif dataset == "mnist":
            self.max_clients = 200
            self.dataset = datasets.MNIST(root, train=True, download=True)
        self.split_train_test(iid, num_clients, dataset)

    def split_train_test(self, iid, num_clients, dataset):
        data = self.dataset.data
        target = self.dataset.targets
        partial_clients = None
        if iid:
            partial_clients = Partitioner(target, self.max_clients, dataset, "iid", seed=2023)
        else:
            partial_clients = Partitioner(target, self.max_clients, dataset, "noniid-labeldir", dir_alpha=0.3, seed=2023)

        self.clients = []
        self.datasets_index = []

        for i in np.random.choice(self.max_clients, num_clients, replace=False):
            sets = partial_clients.client_dict[i]
            train_index = int(len(sets) * 0.9)
            train_set = {"x": data[sets[:train_index]], "y": target[sets[:train_index]]}
            test_set = {"x": data[sets[train_index:]], "y": target[sets[train_index:]]}
            self.clients.append(SingleDataset(train_set, train_index, test_set))
            self.datasets_index.append(list(target[sets[:train_index]]))


    def __getitem__(self, item):
        return self.clients[item]


class SingleDataset:
    def __init__(self, train, num_samples, test):
        self.samples = train['x'].numpy()
        self.labels = train['y'].numpy()
        self.num_samples = num_samples
        self.test_samples = test['x'].numpy()
        self.test_labels = test['y'].numpy()

    def __len__(self):
        return len(self.samples)

    def get_dataset(self):
        return FewShot(self.samples, self.labels)

    def get_random_tasks(self, num_tasks):
        num_class = len(np.unique(self.labels))
        class_indices = np.random.choice(np.unique(self.labels), size=num_tasks, replace=False if num_tasks < num_class else True)
        return class_indices

    def get_random_test_task(self, n=5):
        idx = np.random.choice(range(len(self.test_samples)), n, replace=True)
        return FewShot(self.test_samples[idx], self.test_labels[idx])

    def get_random_train_task(self, n=5):
        idx = np.random.choice(range(self.samples.shape[0]), n, replace=False)
        return FewShot(self.samples[idx], self.labels[idx])

    def get_n_task(self, n_way, k_shot):
        task_data, task_targets = [], []

        for way in n_way:
            # Select samples for this class
            class_data = self.samples[self.labels == way]
            indices = np.random.choice(len(class_data), size=k_shot, replace=True)
            task_data.append(class_data[indices])
            task_targets.extend([way] * k_shot)

        # Combine data and targets for this task
        task_data = np.concatenate(task_data, axis=0)
        task_targets = np.asarray(task_targets)

        return FewShot(task_data, task_targets)

    def get_one_task(self, way, k_shot):
        task_data, task_targets = [], []

        # Select samples for this class
        class_data = self.samples[self.labels == way]
        indices = np.random.choice(len(class_data), size=k_shot, replace=True)
        task_data.append(class_data[indices])
        task_targets.extend([way] * k_shot)

        # Combine data and targets for this task
        task_data = np.concatenate(task_data, axis=0)
        task_targets = np.asarray(task_targets)

        return FewShot(task_data, task_targets)


class FewShot(data.Dataset):
    def __init__(self, samples, targets):
        self.targets = targets
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.lbl_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = int(self.targets[idx])

        image = Image.fromarray(sample, mode='L')
        image = self.transform(image)
        target = torch.tensor(target).long()
        return image, target


if __name__ == "__main__":
    data = EMNIST("data", 10, iid=True, dataset="mnist")
    pass
