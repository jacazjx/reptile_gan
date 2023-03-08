import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils import data
from PIL import Image
np.random.seed(2023)


class FEMNIST:
    def __init__(self, root, num_clients=10, iid="iid"):
        self.path = "./" + root + f"/FEMNIST/{iid}/"
        self.num_clients = num_clients
        self.client = []
        self.num_samples = None
        self.train_set = None
        self.test_set = None
        self.read_dataset()

    def read_dataset(self):
        num = np.random.choice(350, np.clip(self.num_clients, 1, 349), replace=False)
        train_files = os.listdir(self.path + "train")
        train_files = [f for f in train_files if f.endswith('.json')]
        test_files = os.listdir(self.path + "test")
        test_files = [f for f in test_files if f.endswith('.json')]

        self.train_set = []
        self.test_set = []
        self.num_samples = []
        for n in num:
            outter_index = n // 10
            inner_index = n % 10
            with open(self.path + "train/" + train_files[outter_index]) as f:
                d = json.load(f)
            self.num_samples.append(d["num_samples"][inner_index])
            self.train_set.append(d["user_data"][d["users"][inner_index]])

            with open(self.path + "test/" + test_files[outter_index]) as f:
                d = json.load(f)
            self.test_set.append(d["user_data"][d["users"][inner_index]])


    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, item):
        return SingleDataset(self.train_set[item], self.num_samples[item], self.test_set[item])


class SingleDataset:
    def __init__(self, train, num_samples, test):
        self.samples = train['x']
        self.labels = train['y']
        self.num_samples = num_samples
        self.test = test

    def get_random_test_task(self, n=5):
        idx = np.random.choice(range(self.test['x'].shape[0]), n, replace=False)
        return FewShot(self.test['x'][idx], self.test['y'][idx])

    def get_random_train_task(self, n=5):
        idx = np.random.choice(range(self.samples.shape[0]), n, replace=False)
        return FewShot(self.samples[idx], self.labels[idx])

    def get_random_task(self, n_way, k_shot):
        class_indices = np.random.choice(np.unique(self.labels), size=n_way, replace=False)
        task_data, task_targets = [], []

        for class_idx in class_indices:
            # Select samples for this class
            class_data = self.samples[self.labels == class_idx]
            indices = np.random.choice(len(class_data), size=k_shot, replace=False)
            task_data.append(class_data[indices])
            task_targets.extend([class_idx] * k_shot)

        # Combine data and targets for this task
        task_data = np.concatenate(task_data, axis=0)
        task_targets = np.asarray(task_targets)

        # Convert data and targets to tensors
        task_data = torch.from_numpy(task_data).float()
        task_targets = torch.from_numpy(task_targets).long()

        return FewShot(task_data, task_targets)


class FewShot(data.Dataset):
    def __init__(self, samples, targets):
        self.targets = targets
        self.samples = samples
        self.transform = transforms.Compose([transforms.Resize(32),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]

        image = Image.fromarray(sample.numpy(), mode='L')

        image = self.transform(image)
        return image, target


if __name__ == '__main__':
    dataset = FEMNIST("data", 10, "iid")
    user_1 = dataset[0]

    task = user_1.get_random_task(10, 10)
    for i in torch.utils.data.DataLoader(task, batch_size=10):
        plt.show(i)

