import os

import numpy as np
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms

from utils import list_dir, list_files


def read_image(path, size=None):
    img = Image.open(path, mode='r').convert('L')
    if size is not None:
        img = img.resize(size)
    return img


class ImageCache(object):
    def __init__(self):
        self.cache = {}

    def read_image(self, path, size=None):
        key = (path, size)
        if key not in self.cache:
            self.cache[key] = read_image(path, size)
        else:
            pass  #print 'reusing cache', key
        return self.cache[key]


class FewShot(data.Dataset):
    '''
    Dataset for K-shot N-way classification
    '''
    def __init__(self, paths, meta=None, parent=None):
        self.paths = paths
        self.meta = {} if meta is None else meta
        self.parent = parent
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]['path']
        if self.parent.cache is None:
            image = read_image(path, self.parent.size)
        else:
            image = self.parent.cache.read_image(path, self.parent.size)
        if self.parent.transform_image is not None:
            image = self.parent.transform_image(image)
        label = self.paths[idx]
        if self.parent.transform_label is not None:
            label = self.parent.transform_label(label)
        return self.transform(image), label



class AbstractMetaFIGR(object):

    def __init__(self, characters_list, cache=None, size=(28, 28),
                 transform_image=None, transform_label=None):
        self.characters_list = characters_list
        self.cache = cache
        self.size = size
        self.transform_image = transform_image
        self.transform_label = transform_label

    def __len__(self):
        return len(self.characters_list)

    def __getitem__(self, idx):
        return self.characters_list[idx]

    def get_random_task(self, N=5, K=1):
        train_task, __ = self.get_random_task_split(N, train_K=K, test_K=0)
        return train_task

    def get_random_task_split(self, N=5, train_K=1, test_K=1):
        train_samples = []
        test_samples = []
        character_indices = np.random.choice(len(self), N, replace=False)  # N 个 类 每个类 K 张
        for base_idx, idx in enumerate(character_indices):
            character, paths = self.characters_list[idx]
            for i, path in enumerate(np.random.choice(paths, train_K + test_K, replace=False)):
                new_path = {}
                new_path.update(path)
                new_path['base_idx'] = base_idx
                if i < train_K:
                    train_samples.append(new_path)
                else:
                    test_samples.append(new_path)
        train_task = FewShot(train_samples,
                            meta={'characters': character_indices, 'split': 'train'},
                            parent=self
                            )
        test_task = FewShot(test_samples,
                             meta={'characters': character_indices, 'split': 'test'},
                             parent=self
                             )
        return train_task, test_task



class MetaFIGRFolder(AbstractMetaFIGR):

    def __init__(self, root='omniglot', *args, **kwargs):
        '''
        :param root: folder containing alphabets for background and evaluation set
        '''
        self.root = root
        self.alphabets = list_dir(root)
        self._characters = {}
        for alphabet in self.alphabets:
            full_character = os.path.join(root, alphabet)
            character_idx = len(self._characters)
            self._characters[full_character] = []
            for filename in list_files(full_character, '.png'):
                self._characters[full_character].append({
                    'path': os.path.join(root, alphabet, filename),
                    'character_idx': character_idx
                })
        characters_list = np.asarray(list(self._characters.items()))
        super(MetaFIGRFolder, self).__init__(characters_list)

class MetaFIGRSplit(AbstractMetaFIGR):
    pass

def split_omniglot(meta_omniglot, validation=0.1):
    '''
    Split meta-omniglot into two meta-datasets of tasks (disjoint characters)
    '''
    n_val = int(validation * len(meta_omniglot))  #
    indices = np.arange(len(meta_omniglot))
    np.random.shuffle(indices)
    train_characters = meta_omniglot[indices[:-n_val]]
    test_characters = meta_omniglot[indices[-n_val:]]
    train = MetaFIGRSplit(train_characters, cache=meta_omniglot.cache, size=meta_omniglot.size,
                              transform_image=meta_omniglot.transform_image, transform_label=meta_omniglot.transform_label)
    test = MetaFIGRSplit(test_characters, cache=meta_omniglot.cache, size=meta_omniglot.size,
                             transform_image=meta_omniglot.transform_image, transform_label=meta_omniglot.transform_label)
    return train, test

def transform_label(paths):
    return paths['base_idx']

if __name__ == '__main__':
    # figrdataset = MetaFIGRFolder('data/FIGR-8/Data', )
    transform_image = transforms.Compose([transforms.ToTensor()])
    transform_label = transform_label
    omniglot = MetaFIGRFolder('data/FIGR-8/Data', size=(28, 28), cache=ImageCache(),
                                  transform_image=transform_image,
                                  transform_label=transform_label)

    meta_train, meta_test = split_omniglot(omniglot)

    task, _ = meta_test.get_random_task_split(5, 5)
    dataloader = torch.utils.data.DataLoader(task)
    for i in dataloader:
        torchvision.utils.save_image(i)