import torch
from torch import nn
from models.snn import SNN_Model, create_bptt_model
from utils import expand_conf
from scipy.spatial.distance import cdist


class Grad(nn.Module):
    def __init__(self, layer):
        super().__init__()
        for k, v in layer.named_parameters():
            self.register_buffer(k.replace('.', '_'), torch.zeros_like(v), persistent=False)

    def zero_grad(self):
        for v in self.buffers():
            v.zero_()


class BackpropBase(nn.Module):
    def __init__(self,
                 model: SNN_Model,
                 create_bptt=False,
                 temporal_detach_bptt=False):
        super().__init__()

        self.model = model
        self.create_bptt = create_bptt
        self.temporal_detach_bptt = expand_conf(temporal_detach_bptt, model.num_layers)

        self.batch_size = model.batch_size
        self.neuron_nums = model.neuron_nums
        self.neuron_type = model.neuron_type
        self.recurrent = model.recurrent
        self.bias = model.bias
        # self.device = model.device
        self.kwargs = model.kwargs
        assert False not in model.temporal_detach
        
        self.num_layers = model.num_layers
        self.grads = nn.ModuleList()

        if self.create_bptt:
            self.bptt_model = create_bptt_model(model, self.temporal_detach_bptt)
    
    # def __getattr__(self, name):
    #     if name in self.__dict__:
    #         return self.__dict__[name]
    #     elif name in self._modules:
    #         return self._modules[name]
    #     else:
    #         return self._modules['model'].__getattr__(name)
    
    def forward(self, x, time_step, update_matrices=True, verify=False, rtol=1e-4, atol=1e-6):
        if self.create_bptt:
            # with torch.no_grad():
            x_ = torch.clone(x)
            x_ = self.bptt_model(x_, time_step)

        x = self.model(x, time_step)

        if self.training and update_matrices:
            if time_step == 0:
                self.reset_matrices()
            self.update_matrices()
        
        if verify and self.create_bptt:
            assert torch.allclose(x, x_, rtol=rtol, atol=atol)
        
        return (x, x_) if self.create_bptt else x

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none)
        for mod in self.grads:
            assert isinstance(mod, Grad), "Gradients should be stored in Grad modules."
            mod.zero_grad()

    @torch.no_grad()
    def reset_matrices(self):
        raise NotImplementedError()

    @torch.no_grad()
    def update_matrices(self):
        raise NotImplementedError()

    def calc_grad(self, loss):
        raise NotImplementedError()

    def assign_grad(self):
        raise NotImplementedError()

    def print_grad(self, which='online'): # [NOTE] check the display format
        assert which in ['online', 'bptt'], "only 'online' and 'bptt' are allowed."
        target_model = self.model if which == 'online' else self.bptt_model
        print('Layer weights:')
        for layer in target_model.layers:
            print('  ', end='')
            for name, param in layer.named_parameters():
                print(name, param.grad.norm().cpu().item(), end='  ')
            print()
        print('Classifier weights:')
        print('  ', end='')
        for name, param in target_model.classifier.named_parameters():
            print(name, param.grad.norm().cpu().item(), end='  ')
        print()
    
    def print_similarity(self, mode='cosine'):
        def calc_dist(model_a, model_b, mode):
            grad_a = torch.concat([p.grad.flatten() for p in model_a.parameters()])
            grad_b = torch.concat([p.grad.flatten() for p in model_b.parameters()])
            dist = cdist(grad_a.unsqueeze(0).cpu(), grad_b.unsqueeze(0).cpu(), mode)[0][0]
            return dist

        ret = []
        for layer_online, layer_bptt in zip(self.model.layers, self.bptt_model.layers):
            dist = calc_dist(layer_online, layer_bptt, mode)
            ret.append(dist)
        # dist = calc_dist(model.model.classifier, model.bptt_model.classifier, mode)
        # ret.append(dist)
        print(ret)
    
    def verify_grad(self, rtol=1e-3, atol=1e-6):
        if not self.create_bptt:
            raise ValueError("No BPTT model to compare gradients with.")

        for ((name, param1), (_, param2)) in zip(self.model.named_parameters(), self.bptt_model.named_parameters()):
            if name != _:
                raise ValueError(f"Parameter named '{name}' not found in BPTT model.")
            assert torch.allclose(param1.grad, param2.grad, rtol=rtol, atol=atol), f"Gradient of parameter '{name}' is not equal."

    def state_dict(self, which='online', *args, **kwargs):
        assert which in ['online', 'bptt', 'all'], "only 'online', 'bptt', and 'all' are allowed."
        if which == 'online':
            return self.model.state_dict(*args, **kwargs)
        elif which == 'bptt':
            return self.bptt_model.state_dict(*args, **kwargs)
        else:
            return super().state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        return super().load_state_dict(state_dict, strict, assign)
