import torch
import numpy as np
from torch import nn
from models.base import BackpropBase, Grad, create_bptt_model
from neurons import *


class EpropLIF(BackpropBase):
    def __init__(self, model, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        assert model.classifier.name == 'Linear', "Only Linear classifier is supported for E-prop."
        
        for k, v in model.layers.named_parameters():
            assert 'bias' not in k
            if 'weight' in k:
                nn.init.normal_(v, mean=0, std=1 / np.sqrt(v.shape[1]))
        nn.init.normal_(model.classifier.weight, mean=0, std=1 / np.sqrt(model.classifier.weight.shape[1]))

        if create_bptt:
            self.bptt_model = create_bptt_model(model, temporal_detach_bptt)

        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])
        self.M = [None] * self.num_layers
        
        for i, layer in enumerate(self.model.layers):
            if layer.recurrent is None:
                M_size = layer.in_features + (1 if layer.linear.bias is not None else 0)
            else:
                M_size = (layer.in_features + layer.out_features + (1 if layer.linear.bias is not None else 0)) * layer.out_features
            self.M[i] = torch.zeros(self.batch_size, M_size, layer.out_features)
    
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
            diff_J = layer.temporal_derivative()

            if layer.recurrent is not None:
                self.M[i] @= diff_J.transpose(1, 2)
                tmp = self.M[i][:, : layer.linear.weight.numel(), :].view(self.batch_size, layer.in_features, layer.out_features, -1).diagonal(dim1=-2, dim2=-1)
                tmp += layer.input.unsqueeze(2)
                M_offset = layer.linear.weight.numel()
                if layer.recurrent is not None:
                    tmp = self.M[i][:, M_offset : M_offset + layer.recurrent.weight.numel(), :].view(self.batch_size, *layer.recurrent.weight.shape, -1).diagonal(dim1=-2, dim2=-1)
                    tmp += layer.old_spike.unsqueeze(2)
                    M_offset += layer.recurrent.weight.numel()
                if layer.linear.bias is not None:
                    tmp = self.M[i][:, M_offset : M_offset + layer.linear.bias.numel(), :].diagonal(dim1=1, dim2=2)
                    tmp += 1
            else:
                self.M[i] *= diff_J.unsqueeze(1)
                self.M[i][:, : layer.in_features, :] += layer.input.unsqueeze(2)
                if layer.linear.bias is not None:
                    self.M[i][:, -1, :] += 1
            
    @torch.no_grad()
    def assign_grad(self):
        grad_output = self.model.output.grad
        for i, layer in enumerate(self.model.layers):
            if i == self.num_layers - 1:
                B_random = self.model.classifier.weight
            else:
                B_random = torch.randn(self.model.neuron_nums[-1], layer.out_features, device=grad_output.device) / np.sqrt(layer.out_features)
            learning_signal = torch.einsum('bi,ij,bj->bj', grad_output, B_random, layer.spatial_derivative(y='s', x='u'))
            if layer.recurrent is None:
                grad = torch.einsum('bji,bi->ji', self.M[i], learning_signal).flatten()
            else:
                grad = torch.einsum('bji,bi->j', self.M[i], learning_signal)
            
            self.grads[i].linear_weight += grad[: layer.linear.weight.numel()].view(layer.in_features, layer.out_features).T
            layer.linear.weight.grad = self.grads[i].linear_weight.clone()
            offset = layer.linear.weight.numel()
            if layer.recurrent is not None:
                self.grads[i].recurrent_weight += grad[offset : offset + layer.recurrent.weight.numel()].view_as(layer.recurrent.weight).T
                layer.recurrent.weight.grad = self.grads[i].recurrent_weight.clone()
                offset += layer.recurrent.weight.numel()
            if layer.linear.bias is not None:
                self.grads[i].linear_bias += grad[offset : offset + layer.linear.bias.numel()]
                layer.linear.bias.grad = self.grads[i].linear_bias.clone()
    
    def calc_grad(self, loss):
        self.model.output.retain_grad()
        assert self.model.output.grad is None
        loss.backward(retain_graph=True)
        self.assign_grad()


class EpropALIF(BackpropBase):
    def __init__(self, model, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        assert model.classifier.name == 'Linear', "Only Linear classifier is supported for E-prop."

        for k, v in model.layers.named_parameters():
            assert 'bias' not in k
            if 'weight' in k:
                nn.init.normal_(v, mean=0, std=1 / np.sqrt(v.shape[1]))
        nn.init.normal_(model.classifier.weight, mean=0, std=1 / np.sqrt(model.classifier.weight.shape[1]))

        if create_bptt:
            self.bptt_model = create_bptt_model(model, temporal_detach_bptt)

        self.N = [None] * self.num_layers
        self.M = [None] * self.num_layers
        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])
        
        for i, layer in enumerate(self.model.layers):
            if layer.recurrent is None:
                M_size = layer.in_features + (1 if layer.linear.bias is not None else 0)
            else:
                M_size = (layer.in_features + layer.out_features + (1 if layer.linear.bias is not None else 0)) * layer.out_features
            self.N[i] = torch.zeros(self.batch_size, M_size, layer.out_features)
            self.M[i] = torch.zeros(self.batch_size, M_size, layer.out_features)
    
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
            diff_Juu = layer.temporal_derivative(y='u', x='u')
            diff_Jua = layer.temporal_derivative(y='u', x='a')
            diff_Jau = layer.temporal_derivative(y='a', x='u')
            diff_Jaa = layer.temporal_derivative(y='a', x='a')

            if layer.recurrent is not None:
                self.N[i], self.M[i] = self.N[i] @ diff_Juu.transpose(1, 2) + self.M[i] @ diff_Jua.transpose(1, 2), \
                                             self.N[i] * diff_Jau.unsqueeze(1) + self.M[i] * diff_Jaa.unsqueeze(1)

                tmp = self.N[i][:, : layer.linear.weight.numel(), :].view(self.batch_size, layer.in_features, layer.out_features, -1).diagonal(dim1=-2, dim2=-1)
                tmp += layer.input.unsqueeze(2)
                M_offset = layer.linear.weight.numel()
                if layer.recurrent is not None:
                    tmp = self.N[i][:, M_offset : M_offset + layer.recurrent.weight.numel(), :].view(self.batch_size, *layer.recurrent.weight.shape, -1).diagonal(dim1=-2, dim2=-1)
                    tmp += layer.old_spike.unsqueeze(2)
                    M_offset += layer.recurrent.weight.numel()
                if layer.linear.bias is not None:
                    tmp = self.N[i][:, M_offset : M_offset + layer.linear.bias.numel(), :].diagonal(dim1=1, dim2=2)
                    tmp += 1
            else:
                self.N[i], self.M[i] = self.N[i] * diff_Juu.unsqueeze(1) + self.M[i] * diff_Jua.unsqueeze(1), \
                                             self.N[i] * diff_Jau.unsqueeze(1) + self.M[i] * diff_Jaa.unsqueeze(1)

                self.N[i][:, : layer.in_features, :] += layer.input.unsqueeze(2)
                if layer.linear.bias is not None:
                    self.N[i][:, -1, :] += 1
            

    @torch.no_grad()
    def assign_grad(self):
        grad_output = self.model.output.grad
        for i, layer in enumerate(self.model.layers):
            if i == self.num_layers - 1:
                B_random = self.model.classifier.weight
            else:
                B_random = torch.randn(self.model.neuron_nums[-1], layer.out_features, device=grad_output.device) / np.sqrt(layer.out_features)
            learning_signal = torch.einsum('bi,ij->bj', grad_output, B_random)

            if layer.recurrent is None:
                grad = torch.einsum('bji,bi->ji', self.N[i], learning_signal * layer.spatial_derivative(y='s', x='u')).flatten() + torch.einsum('bji,bi->ji', self.M[i], learning_signal * layer.spatial_derivative(y='s', x='a')).flatten()
            else:
                grad = torch.einsum('bji,bi->j', self.N[i], learning_signal * layer.spatial_derivative(y='s', x='u')) + torch.einsum('bji,bi->j', self.M[i], learning_signal * layer.spatial_derivative(y='s', x='a'))
            
            self.grads[i].linear_weight += grad[: layer.linear.weight.numel()].view(layer.in_features, layer.out_features).T
            layer.linear.weight.grad = self.grads[i].linear_weight.clone()
            offset = layer.linear.weight.numel()
            if layer.recurrent is not None:
                self.grads[i].recurrent_weight += grad[offset : offset + layer.recurrent.weight.numel()].view_as(layer.recurrent.weight).T
                layer.recurrent.weight.grad = self.grads[i].recurrent_weight.clone()
                offset += layer.recurrent.weight.numel()
            if layer.linear.bias is not None:
                self.grads[i].linear_bias += grad[offset : offset + layer.linear.bias.numel()]
                layer.linear.bias.grad = self.grads[i].linear_bias.clone()
    
    def calc_grad(self, loss):
        self.model.output.retain_grad()
        assert self.model.output.grad is None
        loss.backward(retain_graph=True)
        self.assign_grad()


class EpropDHLIF(BackpropBase):
    def __init__(self, model, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        assert model.classifier.name == 'Linear', "Only Linear classifier is supported for E-prop."

        for k, v in model.layers.named_parameters():
            assert 'bias' not in k
            if 'weight' in k:
                nn.init.normal_(v, mean=0, std=1 / np.sqrt(v.shape[1]))
        nn.init.normal_(model.classifier.weight, mean=0, std=1 / np.sqrt(model.classifier.weight.shape[1]))

        if create_bptt:
            self.bptt_model = create_bptt_model(model, temporal_detach_bptt)

        self.N = [None] * self.num_layers
        self.M = [None] * self.num_layers
        
        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])
        
        for i, layer in enumerate(self.model.layers):
            assert isinstance(layer, DHLIF)
            M_size = layer.linear.weight.numel() + layer.recurrent.weight.numel()
            if layer.linear.bias is not None:
                M_size += layer.linear.bias.numel()
            self.N[i] = torch.zeros(self.batch_size, M_size, layer.out_features)
            self.M[i] = torch.zeros(self.batch_size, M_size, layer.out_features * layer.branch)
    
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
            diff_Juu = layer.temporal_derivative(y='u', x='u') # (batch_size, out_features)
            diff_Jii = layer.temporal_derivative(y='i', x='i') # scalar
            diff_Jiu = layer.temporal_derivative(y='i', x='u') # (batch_size, out_features * branch, out_features)

            self.M[i] = diff_Jii * self.M[i] + self.N[i] @ diff_Jiu.transpose(1, 2)


            tmp = self.M[i][:, : layer.linear.weight.numel(), :].view(self.batch_size, layer.in_features, layer.out_features * layer.branch, -1).diagonal(dim1=-2, dim2=-1)
            tmp += layer.input.unsqueeze(2)
            M_offset = layer.linear.weight.numel()

            tmp = self.M[i][:, M_offset : M_offset + layer.recurrent.weight.numel(), :].view(self.batch_size, layer.out_features, layer.out_features * layer.branch, -1).diagonal(dim1=-2, dim2=-1)
            tmp += layer.old_spike.unsqueeze(2)
            M_offset += layer.recurrent.weight.numel()

            if layer.linear.bias is not None:
                tmp = self.M[i][:, M_offset : M_offset + layer.linear.bias.numel(), :].diagonal(dim1=1, dim2=2)
                tmp += 1

            self.N[i] = self.N[i] * diff_Juu.unsqueeze(1) + self.M[i].view(self.batch_size, -1, layer.out_features, layer.branch).sum(dim=3)

    @torch.no_grad()
    def assign_grad(self):
        grad_output = self.model.output.grad
        for i, layer in enumerate(self.model.layers):
            if i == self.num_layers - 1:
                B_random = self.model.classifier.weight
            else:
                B_random = torch.randn(self.model.neuron_nums[-1], layer.out_features, device=grad_output.device) / np.sqrt(layer.out_features)
            
            learning_signal = torch.einsum('bi,ij,bj->bj', grad_output, B_random, layer.spatial_derivative(y='s', x='u'))
            grad = torch.einsum('bji,bi->j', self.N[i], learning_signal)
            
            self.grads[i].linear_weight += grad[: layer.linear.weight.numel()].view(layer.in_features, layer.out_features * layer.branch).T
            layer.linear.weight.grad = self.grads[i].linear_weight.clone() * layer.mask[:, :layer.in_features]
            offset = layer.linear.weight.numel()

            self.grads[i].recurrent_weight += grad[offset : offset + layer.recurrent.weight.numel()].view(layer.out_features, layer.out_features * layer.branch).T
            layer.recurrent.weight.grad = self.grads[i].recurrent_weight.clone() * layer.mask[:, layer.in_features:]
            offset += layer.recurrent.weight.numel()

            if layer.linear.bias is not None:
                self.grads[i].linear_bias += grad[offset : offset + layer.linear.bias.numel()]
                layer.linear.bias.grad = self.grads[i].linear_bias.clone()
    
    def calc_grad(self, loss):
        self.model.output.retain_grad()
        assert self.model.output.grad is None
        loss.backward(retain_graph=True)
        self.assign_grad()


def get_eprop(model, create_bptt=False, temporal_detach_bptt=False):
    if not all(model.temporal_detach):
        raise ValueError("E-prop requires all temporal gradients to be detached in the feedforward pass.")
    
    class_dict = {
        'lif': EpropLIF,
        'alif': EpropALIF,
        'dhlif': EpropDHLIF,
    }
    return class_dict[model.neuron_type](model, create_bptt, temporal_detach_bptt)
