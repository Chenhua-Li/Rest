import torch
from tasks.base import Task
from spikingjelly.datasets.n_mnist import NMNIST
from torchvision import transforms


class MyToTensor:
    def __call__(self, pic):
        return torch.from_numpy(pic)


class NMNISTTask(Task):
    def __init__(self, root='./data/NMNIST'):
        self.root = root
        # self.downsample = downsample
        # trans = transforms.Compose([MyToTensor(), transforms.Resize(downsample, transforms.InterpolationMode.NEAREST_EXACT)])
        self.train_dataset = NMNIST(root=root, train=True, data_type='frame', frames_number=20, split_by='number', transform=MyToTensor())
        self.test_dataset = NMNIST(root=root, train=False, data_type='frame', frames_number=20, split_by='number', transform=MyToTensor())
    
    def get_time_window(self):
        return 20
    
    def has_label_each_step(self):
        return False
    
    def preprocess_data(self, input, label):
        return input.view(-1, 20, 34*34*2).float().permute(1, 0, 2), label
