import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, bag_length=100, seed=7, train=True, included_digits=[0, 7], target_digit=0):
        self.bag_length = bag_length
        self.seed = seed
        self.train = train
        self.included_digits = included_digits  
        self.target_digit = target_digit

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        self.bags_list, self.labels_list = self._form_bags()

    def _form_bags(self):
        if self.train:
            loader = data_utils.DataLoader(
                datasets.MNIST('../datasets', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomRotation(degrees=90),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=self.num_in_train, shuffle=True
            )
        else:
            loader = data_utils.DataLoader(
                datasets.MNIST('../datasets', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomRotation(degrees=90),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=self.num_in_test, shuffle=True
            )

        for batch_data in loader:
            numbers = batch_data[0]
            labels = batch_data[1]


            mask = torch.zeros_like(labels, dtype=torch.bool)
            for digit in self.included_digits:
                mask |= (labels == digit)

            numbers = numbers[mask]
            labels = labels[mask]

            bags_list = []
            labels_list = []

            for i in range(0, len(labels), self.bag_length):
                end_idx = min(i + self.bag_length, len(labels))
                numbers_in_bag = numbers[i:end_idx]
                labels_in_bag = labels[i:end_idx]

                if len(labels_in_bag) < self.bag_length:
                    continue

                bags_list.append(numbers_in_bag)
                labels_list.append(labels_in_bag)

            return bags_list, labels_list

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        bag = self.bags_list[index]
        label = (self.labels_list[index] == self.target_digit).sum().item() / self.bag_length
        return bag, label