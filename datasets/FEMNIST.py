import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import datasets, transforms
from torch.utils import data
from utils import normalize_data, unnormalize_data, get_mApping
np.random.seed(2023)


class FEMNIST:
    def __init__(self, root, num_clients=10, iid="iid"):
        curPath = os.path.abspath(os.path.dirname(__file__))
        rootPath = curPath[:curPath.find("reptile_gan\\") + len("reptile_gan\\")]
        self.path = rootPath + root + f"/FEMNIST/{iid}/"
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
        self.samples = np.array(train['x'])
        self.labels = np.array(train['y'])
        self.num_samples = num_samples
        self.test_samples = np.array(test["x"])
        self.test_labels = np.array(test["y"])

    def get_random_tasks(self, num_tasks):
        class_indices = np.random.choice(np.unique(self.labels), size=num_tasks, replace=False)
        return class_indices

    def get_dataset(self):
        return FewShot(torch.from_numpy(self.samples), torch.from_numpy(self.labels))

    def get_random_test_task(self, n=5):
        idx = np.random.choice(range(len(self.test_samples)), n, replace=True)
        return FewShot(torch.from_numpy(self.test_samples[idx]), torch.from_numpy(self.test_labels[idx]))

    def get_random_train_task(self, n=5):
        idx = np.random.choice(range(self.samples.shape[0]), n, replace=False)
        return FewShot(torch.from_numpy(self.samples[idx]), torch.from_numpy(self.labels[idx]))

    def get_random_task(self, way, k_shot):
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
        self.transform = transforms.Compose([transforms.Resize(64),
                                             transforms.ToTensor(),
                                             transforms.Normalize(0.5, 0.5),
                                             ])
        self.resize = torchvision.transforms.Resize([64, 64])
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]
        image = Image.fromarray(sample.reshape(28, 28) * 255.)
        # image = self.resize(sample.reshape(1, 1, 28, 28))
        image = self.transform(image)

        return image, torch.tensor(target)


if __name__ == '__main__':
    dataset = FEMNIST("data", 1, "iid")
    user_1 = dataset[0]

    task = user_1.get_random_task(10, 10)
    from torchvision.transforms import ToPILImage
    show = ToPILImage()
    for img, _ in torch.utils.data.DataLoader(task, batch_size=10):
        show(img).show()

