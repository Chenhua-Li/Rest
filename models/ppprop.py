import torch
from torch import nn
from models.base import BackpropBase, Grad
from neurons import *

class PppropLIF(BackpropBase):
    def __init__(self, model, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])

        self.Ex = [torch.zeros(self.batch_size, layer.in_features + (layer.out_features if layer.recurrent is not None else 0) + (1 if layer.linear.bias is not None else 0)) for layer in self.model.layers]
        self.Ef = [torch.zeros(self.batch_size, layer.out_features) for layer in self.model.layers]

        self.alpha = 0.9

    @torch.no_grad()
    def reset_matrices(self):
        for Exi in self.Ex:
            Exi.zero_()
        for Efi in self.Ef:
            Efi.zero_()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for i in range(len(self.Ex)):
            self.Ex[i] = self.Ex[i].to(*args, **kwargs)
        for i in range(len(self.Ef)):
            self.Ef[i] = self.Ef[i].to(*args, **kwargs)
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
            self.Ex[i] = self.alpha * self.Ex[i] + cated

            Dt = layer.decay * (1 - layer.thresh * surrogate_grad(layer.old_mem - layer.thresh, layer.lens))
            self.Ef[i] = self.alpha * Dt * self.Ef[i] + (1 - self.alpha)
    
    
    def calc_grad(self, loss):
        for layer in self.model.layers:
            layer.mem.retain_grad()
            assert layer.mem.grad is None
        loss.backward(retain_graph=True)
        self.assign_grad()

    def assign_grad(self):
        for i, layer in enumerate(self.model.layers):
            grad = torch.einsum('bi,bj->ij', layer.mem.grad * self.Ef[i], self.Ex[i])
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


class PppropALIF(BackpropBase):
    def __init__(self, model, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])

        self.Ex = [torch.zeros(self.batch_size, layer.in_features + (layer.out_features if layer.recurrent is not None else 0) + (1 if layer.linear.bias is not None else 0)) for layer in self.model.layers]
        self.Ef = [torch.zeros(self.batch_size, layer.out_features, 2) for layer in self.model.layers]

        self.alpha = 0.9

    @torch.no_grad()
    def reset_matrices(self):
        for Exi in self.Ex:
            Exi.zero_()
        for Efi in self.Ef:
            Efi.zero_()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for i in range(len(self.Ex)):
            self.Ex[i] = self.Ex[i].to(*args, **kwargs)
        for i in range(len(self.Ef)):
            self.Ef[i] = self.Ef[i].to(*args, **kwargs)
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
            self.Ex[i] = self.alpha * self.Ex[i] + cated

            thresh_old = layer.thresh0 + layer.beta * layer.old_adapt
            surrogate = surrogate_grad(layer.old_mem - thresh_old, layer.lens)
            Duut = layer.decay - thresh_old * surrogate
            Daat = layer.rho - layer.beta * surrogate
            Duat = -layer.beta * layer.old_spike + thresh_old * layer.beta * surrogate
            Daut = surrogate

            Dt = torch.stack([torch.stack([Duut, Duat], dim=-1), torch.stack([Daut, Daat], dim=-1)], dim=-2)

            self.Ef[i] = self.alpha * torch.einsum('bijk,bik->bij', Dt, self.Ef[i]) + (1 - self.alpha) * torch.stack([
                torch.ones_like(layer.mem, device=layer.mem.device),
                # Duut,
                Daut], dim=-1)
    
    
    def calc_grad(self, loss):
        for layer in self.model.layers:
            layer.spike.retain_grad()
            layer.mem.retain_grad()
            assert layer.mem.grad is None
            assert layer.spike.grad is None
        loss.backward(retain_graph=True)
        self.assign_grad()

    def assign_grad(self):
        for i, layer in enumerate(self.model.layers):
            dot = (torch.stack([layer.mem.grad, layer.spike.grad * layer.spatial_derivative(y='s', x='a')], dim=-1) * self.Ef[i]).sum(dim=-1)
            grad = torch.einsum('bi,bj->ij', dot, self.Ex[i])
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


class PppropDHLIF(BackpropBase):
    def __init__(self, model, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])

        self.Ex = [torch.zeros(self.batch_size, layer.in_features + (layer.out_features if layer.recurrent is not None else 0) + (1 if layer.linear.bias is not None else 0)) for layer in self.model.layers]
        # # self.Ef = [torch.zeros(self.batch_size, layer.out_features, layer.branch) for layer in self.model.layers]
        # self.Ef = [torch.zeros(1) for layer in self.model.layers]
        # # self.Efs = [torch.zeros(self.batch_size, layer.out_features, layer.branch) for layer in self.model.layers]
        # self.Efs = [torch.zeros(self.batch_size, layer.out_features) for layer in self.model.layers]
        self.Ef = [torch.zeros(self.batch_size, layer.out_features, layer.branch, 2) for layer in self.model.layers]

        self.alpha = 0.9

    @torch.no_grad()
    def reset_matrices(self):
        for Exi in self.Ex:
            Exi.zero_()
        for Efi in self.Ef:
            Efi.zero_()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for i in range(len(self.Ex)):
            self.Ex[i] = self.Ex[i].to(*args, **kwargs)
        for i in range(len(self.Ef)):
            self.Ef[i] = self.Ef[i].to(*args, **kwargs)
        # for i in range(len(self.Efs)):
        #     self.Efs[i] = self.Efs[i].to(*args, **kwargs)
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
            self.Ex[i] = self.alpha * self.Ex[i] + cated

            # surrogate = surrogate_grad(layer.old_mem - layer.thresh, layer.lens)
            # Duut = layer.alpha - layer.thresh * surrogate

            # self.Ef[i] = self.alpha * layer.beta * self.Ef[i] + (1 - self.alpha)
            # self.Efs[i] = self.Ef[i] + self.alpha * Duut * self.Efs[i]

            surrogate = surrogate_grad(layer.old_mem - layer.thresh, layer.lens)
            Duut = (layer.alpha - layer.thresh * surrogate).unsqueeze(-1).repeat(1, 1, layer.branch)
            Diit = torch.ones(self.batch_size, layer.out_features, layer.branch, device=surrogate.device) * layer.beta
            Duit = Duut + torch.ones(self.batch_size, layer.out_features, layer.branch, device=surrogate.device) * layer.beta
            Diut = torch.zeros(self.batch_size, layer.out_features, layer.branch, device=surrogate.device)

            Dt = torch.stack([torch.stack([Duut, Duit], dim=-1), torch.stack([Diut, Diit], dim=-1)], dim=-2)

            self.Ef[i] = self.alpha * torch.einsum('bijmn,bijn->bijm', Dt, self.Ef[i]) + (1 - self.alpha)
    
    
    def calc_grad(self, loss):
        for layer in self.model.layers:
            layer.mem.retain_grad()
            layer.dinput.retain_grad()
            assert layer.mem.grad is None
            assert layer.dinput.grad is None
        loss.backward(retain_graph=True)
        self.assign_grad()

    def assign_grad(self):
        for i, layer in enumerate(self.model.layers):
            
            dot = (torch.stack([
                layer.mem.grad.unsqueeze(-1).repeat(1, 1, layer.branch),
                layer.dinput.grad,
                # torch.zeros(self.batch_size, layer.out_features, layer.branch, device=layer.mem.device)
                ], dim=-1) * self.Ef[i]).sum(dim=-1)
            grad = torch.einsum('bik,bj->ikj', dot, self.Ex[i]).view(-1, self.Ex[i].shape[1])

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


def get_ppprop(model, create_bptt=False, temporal_detach_bptt=False):
    if not all(model.temporal_detach):
        raise ValueError("pp-prop requires all temporal gradients to be detached in the feedforward pass.")

    class_dict = {
        'lif': PppropLIF,
        'alif': PppropALIF,
        'dhlif': PppropDHLIF,
    }
    if model.neuron_type not in class_dict:
        raise ValueError(f"Unsupported neuron type {model.neuron_type} for pp-prop.")
    return class_dict[model.neuron_type](model, create_bptt, temporal_detach_bptt)
