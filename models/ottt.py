import torch
from torch import nn
from models.base import BackpropBase, Grad
from neurons import *

class OTTTLIF(BackpropBase):
    def __init__(self, model, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])

        self.M = [torch.zeros(self.batch_size, layer.in_features + (layer.out_features if layer.recurrent is not None else 0) + (1 if layer.linear.bias is not None else 0)) for layer in self.model.layers]

    @torch.no_grad()
    def reset_matrices(self):
        for Mi in self.M:
            Mi.zero_()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for i in range(len(self.M)):
            self.M[i] = self.M[i].to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def update_matrices(self):
        for i, layer in enumerate(self.model.layers):
            assert isinstance(layer, LIF)
            cated = [layer.input]
            if layer.recurrent is not None:
                cated.append(layer.old_spike)
            if layer.linear.bias is not None:
                cated.append(torch.ones(self.batch_size, 1, device=layer.input.device))
            cated = torch.cat(cated, dim=1)
            self.M[i] = layer.decay * self.M[i] + cated

    @torch.no_grad()
    def assign_grad(self):
        for i, layer in enumerate(self.model.layers):
            grad = torch.einsum('bi,bj->ij', layer.mem.grad, self.M[i])
            self.grads[i].linear_weight += grad[:, : layer.in_features]
            layer.linear.weight.grad = self.grads[i].linear_weight.clone()
            offset = layer.in_features
            if layer.recurrent is not None:
                self.grads[i].recurrent_weight += grad[:, offset : offset + layer.out_features]
                layer.recurrent.weight.grad = self.grads[i].recurrent_weight.clone()
                offset += layer.out_features
            if layer.linear.bias is not None:
                self.grads[i].linear_bias += grad[:, offset]
                layer.linear.bias.grad = self.grads[i].linear_bias.clone()
    
    def calc_grad(self, loss):
        for layer in self.model.layers:
            layer.mem.retain_grad()
            assert layer.mem.grad is None
        loss.backward(retain_graph=True)
        self.assign_grad()


class OTTTALIF(BackpropBase):
    def __init__(self, model, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])

        self.M = [torch.zeros(self.batch_size, layer.in_features + (layer.out_features if layer.recurrent is not None else 0) + (1 if layer.linear.bias is not None else 0)) for layer in self.model.layers]
    
    @torch.no_grad()
    def reset_matrices(self):
        for Mi in self.M:
            Mi.zero_()
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for i in range(len(self.M)):
            self.M[i] = self.M[i].to(*args, **kwargs)
        return self

    @torch.no_grad()
    def update_matrices(self):
        for i, layer in enumerate(self.model.layers):
            assert isinstance(layer, ALIF)
            cated = [layer.input]
            if layer.recurrent is not None:
                cated.append(layer.old_spike)
            if layer.linear.bias is not None:
                cated.append(torch.ones(self.batch_size, 1, device=layer.input.device))
            cated = torch.cat(cated, dim=1)
            self.M[i] = layer.decay * self.M[i] + cated
            

    @torch.no_grad()
    def assign_grad(self):
        for i, layer in enumerate(self.model.layers):
            grad = torch.einsum('bi,bj->ij', layer.mem.grad, self.M[i])
            self.grads[i].linear_weight += grad[:, : layer.in_features]
            layer.linear.weight.grad = self.grads[i].linear_weight.clone()
            offset = layer.in_features
            if layer.recurrent is not None:
                self.grads[i].recurrent_weight += grad[:, offset : offset + layer.out_features]
                layer.recurrent.weight.grad = self.grads[i].recurrent_weight.clone()
                offset += layer.out_features
            if layer.linear.bias is not None:
                self.grads[i].linear_bias += grad[:, offset]
                layer.linear.bias.grad = self.grads[i].linear_bias.clone()
    
    def calc_grad(self, loss):
        for layer in self.model.layers:
            layer.mem.retain_grad()
            assert layer.mem.grad is None
        loss.backward(retain_graph=True)
        self.assign_grad()


class OTTTDHLIF(BackpropBase):
    def __init__(self, model, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)
        
        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])

        self.M = [torch.zeros(self.batch_size, layer.in_features + (layer.out_features if layer.recurrent is not None else 0) + (1 if layer.linear.bias is not None else 0)) for layer in self.model.layers]

        self.N = [torch.zeros(self.batch_size, layer.in_features + (layer.out_features if layer.recurrent is not None else 0) + (1 if layer.linear.bias is not None else 0)) for layer in self.model.layers]
    
    @torch.no_grad()
    def reset_matrices(self):
        for Mi in self.M:
            Mi.zero_()
        for Ni in self.N:
            Ni.zero_()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for i in range(len(self.M)):
            self.M[i] = self.M[i].to(*args, **kwargs)
        for i in range(len(self.N)):
            self.N[i] = self.N[i].to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def update_matrices(self):
        for i, layer in enumerate(self.model.layers):
            assert isinstance(layer, DHLIF)
            cated = [layer.input]
            if layer.recurrent is not None:
                cated.append(layer.old_spike)
            if layer.linear.bias is not None:
                cated.append(torch.ones(self.batch_size, 1, device=layer.input.device))
            cated = torch.cat(cated, dim=1)
            self.N[i] = layer.beta * self.N[i] + cated
            self.M[i] = layer.alpha * self.M[i] + self.N[i]

    @torch.no_grad()
    def assign_grad(self):
        for i, layer in enumerate(self.model.layers):
            grad = torch.einsum('bi,bj->ij', layer.mem.grad, self.M[i]).unsqueeze(1).repeat(1, layer.branch, 1).view(-1, self.M[i].shape[1])
            
            self.grads[i].linear_weight += grad[:, : layer.in_features]
            layer.linear.weight.grad = self.grads[i].linear_weight.clone() * layer.mask[:, :layer.in_features]
            offset = layer.in_features

            self.grads[i].recurrent_weight += grad[:, offset : offset + layer.out_features]
            layer.recurrent.weight.grad = self.grads[i].recurrent_weight.clone() * layer.mask[:, layer.in_features:]
            offset += layer.out_features

            if layer.linear.bias is not None:
                self.grads[i].linear_bias += grad[:, offset]
                layer.linear.bias.grad = self.grads[i].linear_bias.clone()
    
    def calc_grad(self, loss):
        for layer in self.model.layers:
            layer.mem.retain_grad()
            assert layer.mem.grad is None
        loss.backward(retain_graph=True)
        self.assign_grad()


def get_ottt(model, create_bptt=False, temporal_detach_bptt=False):
    if not all(model.temporal_detach):
        raise ValueError("OTTT requires all temporal gradients to be detached in the feedforward pass.")
    
    class_dict = {
        'lif': OTTTLIF,
        'alif': OTTTALIF,
        'dhlif': OTTTDHLIF,
    }
    # if model.neuron_type not in class_dict:
    #     raise ValueError(f"Unsupported neuron type {model.neuron_type} for OTTT.")
    return class_dict[model.neuron_type](model, create_bptt, temporal_detach_bptt)
