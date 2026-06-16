import torch
from torch import nn
from neurons.surrogate_gradient import Activate, surrogate_grad


class LIF(nn.Module):
    def __init__(self, batch_size, in_features, out_features, recurrent=False, bias=True, decay=0.5, thresh=0.3, lens=0.5, temporal_detach=True, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.decay = decay
        self.thresh = thresh
        self.lens = lens
        self.temporal_detach = temporal_detach

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.recurrent = nn.Linear(out_features, out_features, bias=False) if recurrent else None
        # self.recurrent.weight.data *= 0.1

        self.mem = None
        self.spike = None
        self.old_mem = None
        self.old_spike = None
        self.input = None

    def reset_neuron_state(self):
        del self.mem, self.spike, self.old_mem, self.old_spike, self.input
        self.register_buffer('mem', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('spike', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('old_mem', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('old_spike', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('input', torch.zeros(self.batch_size, self.in_features), persistent=False)

    def forward(self, x, init=False):
        if isinstance(x, list):
            x = x[0]
        
        if init:
            self.reset_neuron_state()
            self.to(x.device)

        if self.temporal_detach:
            self.old_mem = self.mem.detach()
            self.old_spike = self.spike.detach()
        else:
            self.old_mem = self.mem
            self.old_spike = self.spike

        self.input = x.detach()
        x = self.linear(x)
        if self.recurrent:
            x = x + self.recurrent(self.old_spike)
        
        self.mem = self.decay * (self.old_mem - self.thresh * self.old_spike) + x
        self.spike = Activate.apply(self.mem - self.thresh, self.lens)

        return [self.spike, self.mem]

    @torch.no_grad()
    def temporal_derivative(self, y='u', x='u', extend=False): # du / du
        if y != 'u' or x != 'u':
            raise NotImplementedError()
        
        surrogate = surrogate_grad(self.old_mem - self.thresh, self.lens)
        ret = self.decay * (1 - self.thresh * surrogate)
        if self.recurrent is None:
            return ret.diag_embed() if extend else ret
        else:
            return ret.diag_embed() + surrogate.unsqueeze(1) * self.recurrent.weight.detach().unsqueeze(0)
    
    
    @torch.no_grad()
    def temporal_simplified(self, y='u', x='u', extend=False):  # du / du
        if y != 'u' or x != 'u':
            raise NotImplementedError()
        surrogate = surrogate_grad(self.old_mem - self.thresh, self.lens)
        ret = self.decay * (1 - self.thresh * surrogate) + \
               surrogate * self.recurrent.weight.diag().unsqueeze(0)
        return ret.diag_embed().clamp(-1.,1.) if extend else ret

    @torch.no_grad()
    def spatial_derivative(self, y, x, extend=False):
        if y == 's' and x == 'u':
            surrogate = surrogate_grad(self.mem - self.thresh, self.lens)
            return surrogate.diag_embed() if extend else surrogate
        elif y == 'u' and x == 'x':
            return self.linear.weight.detach().unsqueeze(0) if extend else self.linear.weight.detach()
        elif y == 's' and x == 'x':
            return surrogate_grad(self.mem - self.thresh, self.lens).unsqueeze(2) * self.linear.weight.detach().unsqueeze(0)
        else:
            raise ValueError(f"Invalid variable {y} {x}")
