import torch
from torch import nn
from neurons import *


class ReadoutBase(nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name
        self.output = None
    
    def reset_state(self, batch_size, features):
        del self.output
        self.register_buffer('output', torch.zeros(batch_size, features), persistent=False)


class LinearReadout(ReadoutBase):
    def __init__(self, in_features, out_features, bias=True, cumsum=False):
        super().__init__(name="Linear")
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.cumsum = cumsum

    def forward(self, x, init=False):
        if isinstance(x, list):
            x = x[0]

        if self.cumsum:
            if init:
                self.reset_state(x.shape[0], self.out_features)
                self.to(x.device)
            
            self.output = self.output.detach() + self.fc(x)
            return self.output
        else:
            return self.fc(x)


class SpikeReadout(ReadoutBase):
    def __init__(self, cumsum=False):
        super().__init__(name="Spike")
        self.cumsum = cumsum
    
    def forward(self, x, init=False):
        if isinstance(x, list):
            x = x[0]

        if self.cumsum:
            if init:
                self.reset_state(*x.shape)
                self.to(x.device)
            self.output = self.output.detach() + x
            return self.output
        else:
            return x


class PotentialReadout(ReadoutBase):
    def __init__(self, cumsum=False):
        super().__init__(name="Potential")
        self.cumsum = cumsum

    def forward(self, x, init=False):
        if isinstance(x, list):
            x = x[1]
        
        if self.cumsum:
            if init:
                self.reset_state(*x.shape)
                self.to(x.device)
            self.output = self.output.detach() + x
            return self.output
        else:
            return x


class PotentialSoftmaxReadout(ReadoutBase):
    def __init__(self, cumsum=False):
        super().__init__(name="PotentialSoftmax")
        self.cumsum = cumsum

    def forward(self, x, init=False):
        if isinstance(x, list):
            x = x[1]
        
        if self.cumsum:
            if init:
                self.reset_state(*x.shape)
                self.to(x.device)
            self.output = self.output.detach() + x.softmax(dim=1)
            return self.output
        else:
            return x.softmax(dim=1)

