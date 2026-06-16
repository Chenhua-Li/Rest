import torch
from torch import nn
from neurons.surrogate_gradient import Activate, surrogate_grad


class ALIF(nn.Module):
    def __init__(self, batch_size, in_features, out_features, recurrent=False, bias=True, thresh0=1.6, beta=0.184, decay=0.95, rho=0.995, lens=0.5, temporal_detach=True, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.decay = decay
        self.thresh0 = thresh0
        self.lens = lens
        self.beta = beta
        self.rho = rho
        self.temporal_detach = temporal_detach

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.recurrent = nn.Linear(out_features, out_features, bias=False) if recurrent else None

        self.mem = None
        self.spike = None
        self.adapt = None
        self.old_mem = None
        self.old_spike = None
        self.old_adapt = None
        self.input = None
    
    def reset_neuron_state(self):
        del self.mem, self.spike, self.adapt, self.old_mem, self.old_spike, self.old_adapt, self.input
        self.register_buffer('mem', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('spike', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('adapt', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('old_mem', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('old_spike', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('old_adapt', torch.zeros(self.batch_size, self.out_features), persistent=False)
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
            self.old_adapt = self.adapt.detach()
        else:
            self.old_mem = self.mem
            self.old_spike = self.spike
            self.old_adapt = self.adapt

        self.input = x.detach()
        x = self.linear(x)
        if self.recurrent:
            x = x + self.recurrent(self.old_spike)

        thresh_old = self.thresh0 + self.beta * self.old_adapt
        self.mem = self.decay * self.old_mem - thresh_old * self.old_spike + x
        self.adapt = self.rho * self.old_adapt + self.old_spike
        thresh = self.thresh0 + self.beta * self.adapt
        self.spike = Activate.apply(self.mem - thresh, self.lens)

        return [self.spike, self.mem, self.adapt]
    
    @torch.no_grad()
    def temporal_derivative(self, y='u', x='u', extend=False):
        assert x in 'ua' and y in 'ua'
        thresh_old = self.thresh0 + self.beta * self.old_adapt
        surrogate = surrogate_grad(self.old_mem - thresh_old, self.lens)
        if y == 'u' and x == 'u':
            ret = self.decay - thresh_old * surrogate
            if self.recurrent is None:
                return ret.diag_embed() if extend else ret
            else:
                return ret.diag_embed() + surrogate.unsqueeze(1) * self.recurrent.weight.detach().unsqueeze(0)
        elif y == 'a' and x == 'a':
            ret = self.rho - self.beta * surrogate
            return ret.diag_embed() if extend else ret
        elif y == 'u' and x == 'a':
            ret = -self.beta * self.old_spike + thresh_old * self.beta * surrogate
            if self.recurrent is None:
                return ret.diag_embed() if extend else ret
            else:
                return ret.diag_embed() - self.beta * surrogate.unsqueeze(1) * self.recurrent.weight.detach().unsqueeze(0)
        elif y == 'a' and x == 'u':
            ret = surrogate
            return ret.diag_embed() if extend else ret
        else:
            raise ValueError(f"Invalid variable {y} {x}")
        
    @torch.no_grad()
    def spatial_derivative(self, y, x, extend=False):
        if y == 'u' and x == 'x':
            return self.linear.weight.detach().unsqueeze(0) if extend else self.linear.weight.detach()
        
        thresh = self.thresh0 + self.beta * self.adapt
        if y == 's' and x == 'u':
            ret = surrogate_grad(self.mem - thresh, self.lens)
            return ret.diag_embed() if extend else ret
        elif y == 's' and x == 'a':
            ret = -self.beta * surrogate_grad(self.mem - thresh, self.lens)
            return ret.diag_embed() if extend else ret
        elif y == 's' and x == 'x':
            return surrogate_grad(self.mem - thresh, self.lens).unsqueeze(2) * self.linear.weight.detach().unsqueeze(0)
        else:
            raise ValueError(f"Invalid variable {y} {x}")
