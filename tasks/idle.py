import torch
import numpy as np
from tasks.base import Task
from torch.utils.data import Dataset


class IdleTaskDataset(Dataset):
    def __init__(self, N, in_features, num_classes, time_window):
        self.N = N
        self.in_features = in_features
        self.num_classes = num_classes
        self.time_window = time_window
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.rng.normal(0, 1, (self.time_window, self.in_features))).float()
        y = self.rng.integers(0, self.num_classes, (1,))[0]
        return x, y


class IdleTask(Task):
    def __init__(self, N, in_features, num_classes, time_window):
        self.N = N
        self.in_features = in_features
        self.num_classes = num_classes
        self.time_window = time_window
        self.train_dataset = IdleTaskDataset(N, in_features, num_classes, time_window)
    
    def get_time_window(self):
        return self.time_window
    
    def has_label_each_step(self):
        return False
    
    def preprocess_data(self, input, label):
        return input.permute(1, 0, 2), label
