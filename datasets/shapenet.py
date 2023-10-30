import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import os
import numpy as np
from utils.tools import coordinates_normalize
import random


class ShapeNetDataset(Dataset):
    """Shapenet 数据集"""

    def __init__(self, root="data/shapnetcorev0", mode="train", points_num=2500, normal_channel=True):
        super(ShapeNetDataset, self).__init__()
        self.points_num = points_num
        self.normal_channel = normal_channel
        self.metadata, self.seg_classes = self._load_and_handle_meta(root, mode)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        seg_type, coordinates_path, labels_path = self.metadata[index]
        coordinates = np.loadtxt(coordinates_path).astype(np.float32)
        if self.normal_channel:
            coordinates = coordinates[:, :3]
        else:
            coordinates = coordinates[: :6]
        labels = np.loadtxt(labels_path).astype(np.int32)

        assert len(coordinates) == len(labels), "The lengths of coordinates and labels do not match!"

        # normalize
        coordinates[:, :3] = coordinates_normalize(coordinates[:, :3])
        # to seg labels
        labels = self.seg_classes[seg_type][labels - 1]
        # shuffle coordinates and resample
        choice = np.random.choice(len(labels), self.points_num, replace=True)
        coordinates = coordinates[choice, :]
        labels = labels[choice]

        return seg_type, coordinates, labels


    def _load_and_handle_meta(self, root, mode):
        id2category = [line.strip().split("\t") for line in
                       open(os.path.join(root, "synsetoffset2category.txt"), "r", encoding="utf-8").readlines()]
        id2category = {item[1]: item[0] for item in id2category}

        # [(seg_type, coordinates_path, labels_path)..]
        # i.e [('Knife', 'data/shapnetcorev0/03624134/points/3d2cb9d291ec39dc58a42593b26221da.pts', 'data/shapnetcorev0/03624134/points_label/3d2cb9d291ec39dc58a42593b26221da.seg')..]
        metadata = []

        if mode == "train":
            with open(os.path.join(root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
                data_paths = [path.split("/")[1:] for path in json.load(f)]
        elif mode == "trainval":
            with open(os.path.join(root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
                data_paths = [path.split("/")[1:] for path in json.load(f)]
            with open(os.path.join(root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
                data_paths = data_paths.extend([path.split("/")[1:] for path in json.load(f)])
        elif mode == "test":
            with open(os.path.join(root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
                data_paths = [path.split("/")[1:] for path in json.load(f)]
        else:
            raise ValueError("'mode' must take a value from the options ['train', 'trainval', 'test']")

        for path in data_paths:
            seg_type = id2category[path[0]]
            coordinates_path = os.path.join(root, path[0], "points", path[1]) + ".pts"
            labels_path = os.path.join(root, path[0], "points_label", path[1]) + ".seg"
            metadata.append((seg_type, coordinates_path, labels_path))

        # Mapping from category ('Chair') to a list of int [10, 11, 12, 13] as segmentation labels
        seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        seg_classes = {k: np.array(v).astype(np.int32) for k, v in seg_classes.items()}

        return metadata, seg_classes

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for data.
        :param batch:

            Args:
                batch: batch of data

        :return:  seg_types, inputs, targets
        """
        batch_size = len(batch)
        points_num = batch[0][1].shape[0]
        features_num = batch[0][1].shape[1]

        seg_types = []
        inputs = torch.zeros(batch_size, points_num, features_num).long()
        targets = torch.zeros(batch_size, points_num).long()

        for idx in range(len(batch)):
            seg_type, coordinates, labels = batch[idx]
            seg_types.append(seg_type)
            inputs[idx, :, :] = torch.from_numpy(coordinates)
            targets[idx, :] = torch.from_numpy(labels)

        inputs = inputs.permute(0, 2, 1).contiguous()

        return seg_types, inputs, targets


if __name__ == '__main__':
    dataset = ShapeNetDataset()
    sample = dataset[0]
    print(sample)

