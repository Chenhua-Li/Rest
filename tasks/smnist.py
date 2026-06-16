from tasks.base import Task
from torchvision.datasets import MNIST
from torchvision import transforms


class SMNISTTask(Task):
    def __init__(self, root='./data/MNIST', time_window=784):
        self.root = root
        self.time_window = time_window
        assert 784 % time_window == 0
        self.train_dataset = MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
        self.test_dataset = MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    
    def get_time_window(self):
        return self.time_window
    
    def has_label_each_step(self):
        return False
    
    def preprocess_data(self, input, label):
        return input.view(self.batch_size, -1, self.time_window).float().permute(2, 0, 1), label
