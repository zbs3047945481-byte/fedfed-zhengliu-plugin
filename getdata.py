from pathlib import Path
import subprocess
import urllib.request

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os
import gzip


DATASET_DEFAULTS = {
    'mnist': {'image_size': 28, 'input_channels': 1, 'num_classes': 10},
    'cifar10': {'image_size': 32, 'input_channels': 3, 'num_classes': 10},
    'cifar-10': {'image_size': 32, 'input_channels': 3, 'num_classes': 10},
    'cifar100': {'image_size': 32, 'input_channels': 3, 'num_classes': 100},
    'cifar-100': {'image_size': 32, 'input_channels': 3, 'num_classes': 100},
    'gtsrb': {'image_size': 32, 'input_channels': 3, 'num_classes': 43},
    'pathmnist': {'image_size': 32, 'input_channels': 3, 'num_classes': 9},
    'plantvillage': {'image_size': 32, 'input_channels': 3, 'num_classes': 38},
}


MEDMNIST_URLS = {
    'pathmnist': 'https://zenodo.org/records/10519652/files/pathmnist.npz?download=1',
}


def get_dataset_defaults(dataset_name):
    return dict(DATASET_DEFAULTS.get(str(dataset_name).lower(), {}))


class GetDataSet():
    def __init__(self, dataset_name, options=None):#self = 当前这个 GetDataSet 实例。
        self.dataset_name = str(dataset_name).lower()
        self.options = options or {}
        defaults = get_dataset_defaults(self.dataset_name)
        self.data_root = Path(self.options.get('data_root', './data'))
        self.image_size = int(self.options.get('image_size') or defaults.get('image_size', 32))
        self.input_channels = int(self.options.get('input_channels') or defaults.get('input_channels', 3))
        self.num_classes = int(self.options.get('num_classes') or defaults.get('num_classes', 10))
        self.dataset_split_seed = int(self.options.get('dataset_split_seed', self.options.get('seed', 3001)))
        self.dataset_train_fraction = float(self.options.get('dataset_train_fraction', 0.8))
        self.dataset_cache = bool(self.options.get('dataset_cache', True))

        self.train_data = None
        self.train_label = None
        self.train_data_size = None

        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        if self.dataset_name.startswith('mnist'):
            self.mnistDataDistribution()
        elif self.dataset_name in {'cifar10', 'cifar-10'}:
            self.cifar10DataDistribution()
        elif self.dataset_name in {'cifar100', 'cifar-100'}:
            self.cifar100DataDistribution()
        elif self.dataset_name == 'gtsrb':
            self.gtsrbDataDistribution()
        elif self.dataset_name == 'pathmnist':
            self.pathmnistDataDistribution()
        elif self.dataset_name == 'plantvillage':
            self.plantVillageDataDistribution()
        else:
            raise ValueError('Unsupported dataset: {}'.format(dataset_name))

    def _cache_path(self, name):
        safe_fraction = str(self.dataset_train_fraction).replace('.', 'p')
        cache_name = '{}_s{}_c{}_seed{}_frac{}.npz'.format(
            name,
            self.image_size,
            self.input_channels,
            self.dataset_split_seed,
            safe_fraction,
        )
        return self.data_root / 'processed' / cache_name

    def _load_cache(self, path):
        if not self.dataset_cache or not path.exists():
            return False
        print('Loading cached dataset', path)
        loaded = np.load(path)
        self.train_data = loaded['train_data'].astype(np.float32)
        self.train_label = loaded['train_label'].astype(np.int64)
        self.test_data = loaded['test_data'].astype(np.float32)
        self.test_label = loaded['test_label'].astype(np.int64)
        self.train_data_size = self.train_data.shape[0]
        self.test_data_size = self.test_data.shape[0]
        print(self.train_data.shape)
        return True

    def _save_cache(self, path):
        if not self.dataset_cache:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            train_data=self.train_data,
            train_label=self.train_label,
            test_data=self.test_data,
            test_label=self.test_label,
        )

    def _resize_nchw(self, images):
        if images.shape[-2:] == (self.image_size, self.image_size):
            return images.astype(np.float32)
        resized = []
        for start in range(0, images.shape[0], 4096):
            batch = torch.from_numpy(images[start:start + 4096].astype(np.float32))
            batch = F.interpolate(
                batch,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
            resized.append(batch.numpy())
        return np.concatenate(resized, axis=0).astype(np.float32)

    def _to_nchw_float(self, images):
        if images.ndim == 3:
            images = images[..., None]
        if images.shape[-1] == 1 and self.input_channels == 3:
            images = np.repeat(images, 3, axis=-1)
        elif images.shape[-1] == 3 and self.input_channels == 1:
            images = images.mean(axis=-1, keepdims=True)
        images = images.astype(np.float32) / 255.0
        images = np.transpose(images, (0, 3, 1, 2))
        return self._resize_nchw(images)

    def _image_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def _torchvision_to_numpy(self, dataset):
        images = []
        labels = []
        for image, label in dataset:
            if self.input_channels == 1 and image.size(0) == 3:
                image = image.mean(dim=0, keepdim=True)
            elif self.input_channels == 3 and image.size(0) == 1:
                image = image.expand(3, -1, -1)
            images.append(image.numpy())
            labels.append(int(label))
        return np.stack(images).astype(np.float32), np.asarray(labels, dtype=np.int64)

    def _finalize_arrays(self, train_data, train_label, test_data, test_label):
        self.train_data = train_data.astype(np.float32)
        self.train_label = np.asarray(train_label, dtype=np.int64).reshape(-1)
        self.test_data = test_data.astype(np.float32)
        self.test_label = np.asarray(test_label, dtype=np.int64).reshape(-1)
        self.train_data_size = self.train_data.shape[0]
        self.test_data_size = self.test_data.shape[0]
        print(self.train_data.shape)

    def mnistDataDistribution(self, ):

        #说明：这个仓库并没有用 torchvision.datasets.MNIST，而是直接读原始 gz 文件
        #所以你必须确保 ./data/MNIST/raw 下已经有这四个文件（通常是手动下载或用脚本下载）
        data_dir = r'./data/MNIST/raw'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        #把目录和文件名用当前操作系统的路径分隔符连起来，得到完整路径。
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = self.extract_images(train_images_path)
        #用当前对象自己的方法，根据“训练集图像文件路径”把 .gz 里的图像读出来，得到一张大数组，并赋给 train_images。
        # print(train_images.shape) # 图片的形状 (60000, 28, 28, 1) 60000张 28 * 28 * 1  灰色一个通道
        # print('-' * 22 + "\n")
        train_labels = self.extract_labels(train_labels_path)
        # print("-" * 5 + "train_labels" + "-" * 5)
        # print(train_labels.shape)  # label shape (60000, 10)
        # print('-' * 22 + "\n")
        test_images = self.extract_images(test_images_path)
        test_labels = self.extract_labels(test_labels_path)


        # assert train_images.shape[0] == train_labels.shape[0]
        # assert test_images.shape[0] == test_labels.shape[0]
        #
        #
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]
        #
        # assert train_images.shape[3] == 1
        # assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2])
#把训练图像从 (60000, 28, 28, 1) 变成 (60000, 1, 28, 28)，即从 NHWC 改成 NCHW，以符合 PyTorch 的约定。
        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        #把像素从 [0,255] → [0,1]
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        #因为 one-hot 里只有一个位置是 1。
        self.train_data = train_images
        self.train_label = np.argmax(train_labels == 1, axis = 1)
        self.test_data = test_images
        self.test_label = np.argmax(test_labels == 1, axis = 1)
        if self.image_size != 28 or self.input_channels != 1:
            self.train_data = self._resize_nchw(self.train_data)
            self.test_data = self._resize_nchw(self.test_data)
            if self.input_channels == 3:
                self.train_data = np.repeat(self.train_data, 3, axis=1)
                self.test_data = np.repeat(self.test_data, 3, axis=1)
        print(self.train_data.shape)

    def cifar10DataDistribution(self):
        data_dir = str(self.data_root)
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)

        train_images = train_dataset.data.astype(np.float32) / 255.0
        test_images = test_dataset.data.astype(np.float32) / 255.0

        train_data = np.transpose(train_images, (0, 3, 1, 2))
        test_data = np.transpose(test_images, (0, 3, 1, 2))
        if self.input_channels == 1:
            train_data = train_data.mean(axis=1, keepdims=True)
            test_data = test_data.mean(axis=1, keepdims=True)
        train_data = self._resize_nchw(train_data)
        test_data = self._resize_nchw(test_data)
        self._finalize_arrays(train_data, train_dataset.targets, test_data, test_dataset.targets)

    def cifar100DataDistribution(self):
        data_dir = str(self.data_root)
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True)
        train_data = np.transpose(train_dataset.data.astype(np.float32) / 255.0, (0, 3, 1, 2))
        test_data = np.transpose(test_dataset.data.astype(np.float32) / 255.0, (0, 3, 1, 2))
        if self.input_channels == 1:
            train_data = train_data.mean(axis=1, keepdims=True)
            test_data = test_data.mean(axis=1, keepdims=True)
        self._finalize_arrays(
            self._resize_nchw(train_data),
            train_dataset.targets,
            self._resize_nchw(test_data),
            test_dataset.targets,
        )

    def gtsrbDataDistribution(self):
        cache_path = self._cache_path('gtsrb')
        if self._load_cache(cache_path):
            return
        transform = self._image_transform()
        train_dataset = datasets.GTSRB(root=str(self.data_root), split='train', download=True, transform=transform)
        test_dataset = datasets.GTSRB(root=str(self.data_root), split='test', download=True, transform=transform)
        train_data, train_label = self._torchvision_to_numpy(train_dataset)
        test_data, test_label = self._torchvision_to_numpy(test_dataset)
        self._finalize_arrays(train_data, train_label, test_data, test_label)
        self._save_cache(cache_path)

    def pathmnistDataDistribution(self):
        cache_path = self._cache_path('pathmnist')
        if self._load_cache(cache_path):
            return
        npz_path = self.data_root / 'pathmnist.npz'
        if not npz_path.exists():
            self.data_root.mkdir(parents=True, exist_ok=True)
            print('Downloading PathMNIST to', npz_path)
            urllib.request.urlretrieve(MEDMNIST_URLS['pathmnist'], npz_path)
        loaded = np.load(npz_path)
        train_data = self._to_nchw_float(loaded['train_images'])
        test_data = self._to_nchw_float(loaded['test_images'])
        train_label = loaded['train_labels'].reshape(-1)
        test_label = loaded['test_labels'].reshape(-1)
        self._finalize_arrays(train_data, train_label, test_data, test_label)
        self._save_cache(cache_path)

    def _plantvillage_root(self):
        candidates = [
            self.data_root / 'PlantVillage-Dataset' / 'raw' / 'color',
            self.data_root / 'plantvillage' / 'color',
            self.data_root / 'PlantVillage' / 'raw' / 'color',
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        clone_dir = self.data_root / 'PlantVillage-Dataset'
        if not clone_dir.exists():
            self.data_root.mkdir(parents=True, exist_ok=True)
            print('Cloning PlantVillage dataset to', clone_dir)
            subprocess.run(
                ['git', 'clone', '--depth', '1', 'https://github.com/spMohanty/PlantVillage-Dataset.git', str(clone_dir)],
                check=True,
            )
        root = clone_dir / 'raw' / 'color'
        if not root.exists():
            raise FileNotFoundError('PlantVillage color image folder not found: {}'.format(root))
        return root

    def _stratified_split_indices(self, targets):
        targets = np.asarray(targets, dtype=np.int64)
        rng = np.random.default_rng(self.dataset_split_seed)
        train_indices = []
        test_indices = []
        for class_id in np.unique(targets):
            class_indices = np.where(targets == class_id)[0]
            rng.shuffle(class_indices)
            if len(class_indices) <= 1:
                train_indices.extend(class_indices.tolist())
                continue
            train_count = int(round(len(class_indices) * self.dataset_train_fraction))
            train_count = min(max(train_count, 1), len(class_indices) - 1)
            train_indices.extend(class_indices[:train_count].tolist())
            test_indices.extend(class_indices[train_count:].tolist())
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
        return train_indices, test_indices

    def plantVillageDataDistribution(self):
        cache_path = self._cache_path('plantvillage')
        if self._load_cache(cache_path):
            return
        root = self._plantvillage_root()
        full_dataset = datasets.ImageFolder(str(root), transform=self._image_transform())
        self.class_to_idx = dict(full_dataset.class_to_idx)
        train_indices, test_indices = self._stratified_split_indices(full_dataset.targets)
        train_data, train_label = self._torchvision_to_numpy(Subset(full_dataset, train_indices))
        test_data, test_label = self._torchvision_to_numpy(Subset(full_dataset, test_indices))
        self._finalize_arrays(train_data, train_label, test_data, test_label)
        self._save_cache(cache_path)


    #读 MNIST 官方二进制格式
    def extract_images(self, filename):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')

        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#extract_labels 做的事情：按 MNIST 标签格式解析 .gz 文件 → 得到一维标签数组 → 再转成 one-hot 并返回。
    def extract_labels(self, filename):
        """Extract the labels into a 1D uint8 numpy array [index]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return self.dense_to_one_hot(labels)

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
