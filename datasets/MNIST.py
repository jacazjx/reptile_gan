import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils import data
from torch.utils.data import DataLoader
from PIL import Image
import copy
# class FewShot(data):
#     '''
#     Dataset for K-shot N-way classification
#     '''
#     def __init__(self, sample, target, trans=False):
#         self.data = sample
#         self.targets = target
#         self.transform = transforms.Compose([transforms.Resize(32),
#                                             transforms.ToTensor(),
#                                             transforms.Normalize([0.5], [0.5])
#                                             ])
#
#         self.T = trans
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def __getitem__(self, index):
#         # if not self.T:
#         #     return self.data[index], self.targets[index]
#         # idx = idx % self.data.shape[0]
#         img, target = self.data[index], int(self.targets[index])
#
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         if not isinstance(img, torch.Tensor):
#             img = torch.from_numpy(img).float()
#
#         img = Image.fromarray(img.numpy(), mode="L")
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         # target = torch.tensor(target).long()
#
#         return img, target


class MNISTTasks:
    def __init__(self, root):
        # Download and load MNIST dataset
        transform = transforms.Compose([transforms.Resize(32),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5])
                                            ])
        self.dataset = datasets.MNIST(root, train=True, download=True, transform=transform)

        self.data = self.dataset.data
        self.targets = self.dataset.targets
        indexes = np.random.choice(range(self.data.shape[0]), 500, replace=False)
        self.data = self.data[indexes]
        self.targets = self.targets[indexes]

        self.split_train_test()
        del self.data, self.targets

    def split_train_test(self):
        self.train_data = self.data[self.targets != 9]
        self.train_targets = self.targets[self.targets != 9]

        self.test_data = self.data[self.targets == 9]
        self.test_targets = self.targets[self.targets == 9]


    def __len__(self):
        return self.data.shape[0]

    def copy(self):
        dataset = copy.deepcopy(self.dataset)
        dataset.data = self.train_data
        dataset.targets = self.train_targets
        return dataset

    def get_random_test_task(self, n=5):
        # return the random test task.
        # The last 10% are training data.
        idx = np.random.choice(range(self.test_data.shape[0]), n, replace=False)
        few_shot = copy.deepcopy(self.dataset)
        few_shot.data = self.test_data[idx]
        few_shot.targets = self.test_targets[idx]
        return few_shot

    def get_random_train_task(self, n=5):
        # return the random train task.
        # The last 10% are training data.
        idx = np.random.choice(range(self.train_data.shape[0]), n, replace=False)
        few_shot = copy.deepcopy(self.dataset)
        few_shot.data = self.train_data[idx]
        few_shot.targets = self.train_targets[idx]
        return few_shot


    def get_random_task(self, k_shot):
        class_indices = np.random.choice(np.unique(self.train_targets), size=1, replace=False)
        task_data, task_targets = [], []

        for class_idx in class_indices:
            # Select samples for this class
            class_data = self.train_data[self.train_targets == class_idx]
            indices = np.random.choice(len(class_data), size=k_shot, replace=False)
            task_data.append(class_data[indices])
            task_targets.extend([class_idx] * k_shot)

        # Combine data and targets for this task
        task_data = np.concatenate(task_data, axis=0)
        task_targets = np.asarray(task_targets)

        # Convert data and targets to tensors
        task_data = torch.from_numpy(task_data).float()
        task_targets = torch.from_numpy(task_targets).long()
        few_shot = copy.deepcopy(self.dataset)
        few_shot.data = task_data
        few_shot.targets = task_targets
        return few_shot



