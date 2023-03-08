import numpy as np
import torch
from torchvision import datasets, transforms

class EMNISTTasks:
    def __init__(self, root, num_tasks, num_classes, num_samples_per_class):
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class

        # Download and load EMNIST dataset
        self.dataset = datasets.EMNIST(root, split='letters', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
        self.data = self.dataset.data.numpy()
        self.targets = self.dataset.targets.numpy()

        # Split the dataset into tasks
        self.tasks = []
        num_images = self.data.shape[0]
        task_size = num_classes * num_samples_per_class

        for task_idx in range(num_tasks):
            # Select classes for this task
            class_indices = np.random.choice(np.unique(self.targets), size=num_classes, replace=False)
            task_data, task_targets = [], []

            for class_idx in class_indices:
                # Select samples for this class
                class_data = self.data[self.targets == class_idx]
                indices = np.random.choice(len(class_data), size=num_samples_per_class, replace=False)
                task_data.append(class_data[indices])
                task_targets.extend([class_idx] * num_samples_per_class)

            # Combine data and targets for this task
            task_data = np.concatenate(task_data, axis=0)
            task_targets = np.asarray(task_targets)

            # Convert data and targets to tensors
            task_data = torch.from_numpy(task_data).float()
            task_targets = torch.from_numpy(task_targets).long()

            # Add task to the list of tasks
            self.tasks.append((task_data, task_targets))

    # def get_random_task(self, n_way, k_shot):





dset = EMNISTTasks(root="./data", num_tasks=100, num_classes=5, num_samples_per_class=5)
