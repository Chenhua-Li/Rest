import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


class Task:
    def __init__(self):
        pass
    
    def prepare_dataloader(self, batch_size, sampler_seed=0, num_workers=0,ddp=False,world_size=1,rank='cuda:0'):
        self.batch_size = batch_size
        self.sampler_seed = sampler_seed

        if hasattr(self, 'train_dataset'):
            if ddp:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                                        self.train_dataset,
                                        num_replicas=world_size,
                                        rank=rank,
                                        shuffle=True)
                self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, drop_last=True, sampler=self.train_sampler, num_workers=num_workers)
            else:
                sampler = RandomSampler(self.train_dataset, generator=torch.Generator().manual_seed(sampler_seed))
                self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, drop_last=True, sampler=sampler, num_workers=num_workers)

        if hasattr(self, 'test_dataset'):
            if ddp:
                self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                                        self.test_dataset,
                                        num_replicas=world_size,
                                        rank=rank,
                                        shuffle=False)
                self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size,sampler=self.test_sampler, drop_last=True, num_workers=num_workers)
            else:
                self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    
    def preprocess_data(self, input, label):
        raise NotImplementedError()
    
    def get_time_window(self):
        raise NotImplementedError()
    
    def has_label_each_step(self):
        raise NotImplementedError()



