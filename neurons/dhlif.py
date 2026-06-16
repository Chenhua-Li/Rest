import torch
from torch import nn
from neurons.surrogate_gradient import Activate, surrogate_grad


class DHLIF(nn.Module):
    def __init__(self, batch_size, in_features, out_features, bias=True, branch=4, thresh=0.5, lens=0.5, temporal_detach=True, **kwargs):
        super().__init__()
        if kwargs.get('recurrent', True) is False:
            raise ValueError('DHLIF does not support non-recurrent mode')

        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.branch = branch
        self.thresh = thresh
        self.lens = lens
        self.temporal_detach = temporal_detach

        self.mem = None
        self.spike = None
        self.dinput = None
        self.old_mem = None
        self.old_spike = None
        self.old_dinput = None
        self.input = None

        # self.dense = nn.Linear(in_features + out_features, out_features * branch, bias=bias)
        self.linear = nn.Linear(in_features, out_features * branch, bias=bias)
        self.recurrent = nn.Linear(out_features, out_features * branch, bias=False)
        # self.tau_m = nn.Parameter(torch.Tensor(out_features))
        # self.tau_n = nn.Parameter(torch.Tensor(out_features, branch))
        self.beta = 0.8
        self.alpha = 0.8

        self.create_mask()
        self.register_forward_pre_hook(DHLIF.apply_forward_mask)
        self.linear.weight.register_hook(lambda grad: grad.data.mul_(self.mask[:, :self.in_features]))
        self.recurrent.weight.register_hook(lambda grad: grad.data.mul_(self.mask[:, self.in_features:]))

    
    def reset_neuron_state(self):
        del self.mem, self.spike, self.dinput, self.old_mem, self.old_spike, self.old_dinput, self.input
        self.register_buffer('mem', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('spike', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('dinput', torch.zeros(self.batch_size, self.out_features, self.branch), persistent=False)
        self.register_buffer('old_mem', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('old_spike', torch.zeros(self.batch_size, self.out_features), persistent=False)
        self.register_buffer('old_dinput', torch.zeros(self.batch_size, self.out_features, self.branch), persistent=False)
        self.register_buffer('input', torch.zeros(self.batch_size, self.in_features), persistent=False)
        
    def create_mask(self):
        input_size = self.in_features + self.out_features
        self.register_buffer('mask', torch.zeros(self.out_features * self.branch, input_size), persistent=True)
        # self.perms = torch.zeros(self.out_features, self.in_features, dtype=int)
        for i in range(self.out_features):
            perm = torch.randperm(input_size)
            # self.perms[i] = perm
            for j in range(self.branch):
                self.mask[i * self.branch + j, perm[j * input_size // self.branch : (j + 1) * input_size // self.branch]] = 1
    
    @staticmethod
    def apply_forward_mask(module, inp):
        module.linear.weight.data *= module.mask[:, :module.in_features]
        module.recurrent.weight.data *= module.mask[:, module.in_features:]
        return inp
    
    def forward(self, x, init=False):
        if isinstance(x, list):
            x = x[0]

        if init:
            self.reset_neuron_state()
            self.to(x.device)

        if self.temporal_detach:
            self.old_mem = self.mem.detach()
            self.old_spike = self.spike.detach()
            self.old_dinput = self.dinput.detach()
        else:
            self.old_mem = self.mem
            self.old_spike = self.spike
            self.old_dinput = self.dinput

        self.input = x.detach()
        # beta = torch.sigmoid(self.tau_n)
        beta = self.beta
        # k_input = torch.cat([x.float(), self.old_spike.float()], dim=1)
        # self.dinput = beta * self.old_dinput + self.dense(k_input).view(self.batch_size, self.out_features, self.branch)
        x = self.linear(x) + self.recurrent(self.old_spike)
        self.dinput = beta * self.old_dinput + x.view(self.batch_size, self.out_features, self.branch)
        l_input = self.dinput.sum(dim=2)
        # alpha = torch.sigmoid(self.tau_m)
        alpha = self.alpha
        self.mem = alpha * self.old_mem + l_input - self.thresh * self.old_spike
        self.spike = Activate.apply(self.mem - self.thresh, self.lens)

        return [self.spike, self.mem, self.dinput]

    @torch.no_grad()
    def temporal_derivative(self, y='u', x='u', extend=False): # note: no other u or i on the path of du(t) / di(t-1)
        assert extend == False
        assert x in 'ui' and y in 'ui'
        surrogate = surrogate_grad(self.old_mem - self.thresh, self.lens)
        if y == 'u' and x == 'u':
            ret = self.alpha - self.thresh * surrogate
            return ret
        elif y == 'i' and x == 'i':
            return self.beta
        elif y == 'i' and x == 'u':
            return self.recurrent.weight.detach().unsqueeze(0) * surrogate.unsqueeze(1)
        else:
            raise ValueError(f"Invalid variable {y} {x}")
    
    @torch.no_grad()
    def spatial_derivative(self, y, x, extend=False):
        if y == 's' and x == 'u':
            return surrogate_grad(self.mem - self.thresh, self.lens)
        else:
            raise ValueError(f"Invalid variable {y} {x}")
