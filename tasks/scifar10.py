from tasks.base import Task
from torchvision.datasets import CIFAR10
from torchvision import transforms


class SCIFAR10Task(Task):
    def __init__(self, root='./data/CIFAR10'):
        self.root = root
        
        transform = transforms.Compose([
            transforms.ToTensor(),  # (C, H, W) in [0,1]
            # transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023,0.1994,0.2010])
        ])
        self.train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
        self.test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)
    
    def get_time_window(self):
        return 32
    
    def has_label_each_step(self):
        return False
    
    def preprocess_data(self, input, label):
        # (batchsize, 32, 32, 3)
        return input.permute(2, 0, 1, 3).reshape(32, -1, 96).float(), label
