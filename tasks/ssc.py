import os
import torch
from typing import Callable
from typing import Any, Dict, Optional
import numpy as np
from tasks.base import Task
from torch.utils.data import Dataset
import json
from .shd_ssc.datasets import BinnedSpikingHeidelbergDigits as BinnedSHD
from .shd_ssc.datasets import BinnedSpikingSpeechCommands as BinnedSSC

# class SHDTaskDataset(Dataset):
#     def __init__(self, root, train):
#         self.root = root
#         split = 'train' if train else 'test'
#         self.X = np.load(os.path.join(root, f'{split}/{split}X_50ms.npy'))
#         self.Y = np.load(os.path.join(root, f'{split}/{split}Y_50ms.npy'))

#     def __len__(self):
#         return self.Y.shape[0]

#     def __getitem__(self, idx):
#         x = torch.from_numpy(self.X[idx]).float()
#         y = self.Y[idx]
#         return x, y

class AlignedSSC(BinnedSSC):
    def __init__(
            self,
            root: str,
            n_bins: int,
            split: str = 'train',
            data_type: str = "event",
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable[..., Any] = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable[..., Any]] = None,
            target_transform: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__(
            root,
            n_bins,
            split,
            data_type,
            frames_number,
            split_by,
            duration,
            custom_integrate_function,
            custom_integrated_frames_dir_name,
            transform,
            target_transform,
        )
        meta_file = os.path.join(root, "meta.json")
        meta = dict()
        if os.path.exists(meta_file):
            with open(meta_file, mode="r") as f:
                try:
                    meta = json.load(f)
                except json.decoder.JSONDecodeError:
                    pass
        else:
            with open(meta_file, "w"):
                pass

        if str(duration) in meta:
            self.max_frame = meta[str(duration)]
            return

        print("resolving alignment param ...")
        max_frame = max(super(AlignedSSC, self).__getitem__(i)[0].shape[0] for i in range(self.length))
        meta[str(duration)] = max_frame
        with open(os.path.join(root, "meta.json"), mode="w") as f:
            json.dump(meta, f)
        self.max_frame = max_frame

    def __getitem__(self, i: int):
        data, tag = super().__getitem__(i)
        aligned = np.zeros((self.max_frame, data.shape[1]), dtype=data.dtype)
        aligned[:data.shape[0], :] = data
        return aligned, tag


class SSCTask(Task):
    def __init__(self, root='./data/SSC', n_bins=1, time_step=20):
        self.root = root

        self.train_dataset = AlignedSSC(root, n_bins, split='train', data_type='frame',
                                duration=time_step )
        self.test_dataset = AlignedSSC(root, n_bins, split='test', data_type='frame',
                                duration=time_step )


        self.time_window = max(self.train_dataset.max_frame, self.test_dataset.max_frame)
        # print(self.time_window)
        self.train_dataset.max_frame = self.time_window
        self.test_dataset.max_frame = self.time_window
    
    def get_time_window(self):
        return self.time_window
    
    def has_label_each_step(self):
        return False
    
    def preprocess_data(self, input, label):
        return input.float().permute(1, 0, 2), label
    
# SSCTask()
