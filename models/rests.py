import torch
from torch import nn
from models.base import BackpropBase, Grad
from neurons import *
from utils import expand_conf


class RestLIF(BackpropBase):
    def __init__(self, model, back, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        self.N = [None] * self.num_layers
        self.back = expand_conf(back, self.num_layers)
        
        self.last_r = self.back.rfind('r')

        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])
        
        for i, layer_i in enumerate(self.model.layers):
            self.N[i] = [None] * self.num_layers
            if self.back[i] == 'r':
                M_size = 0
                extend = layer_i.recurrent is not None
                M_size += layer_i.linear.weight.numel() if extend else layer_i.in_features
                if layer_i.recurrent is not None:
                    M_size += layer_i.recurrent.weight.numel()
                if layer_i.linear.bias is not None:
                    M_size += layer_i.linear.bias.numel() if extend else 1
                self.N[i][i] = torch.zeros(self.batch_size, M_size, layer_i.out_features)

                for j, layer_j in enumerate(self.model.layers[:i]):
                    N_size = 0
                    N_size += layer_j.linear.weight.numel()
                    if layer_j.recurrent is not None:
                        N_size += layer_j.recurrent.weight.numel()
                    if layer_j.linear.bias is not None:
                        N_size += layer_j.linear.bias.numel()
                    self.N[i][j] = torch.zeros(self.batch_size, N_size, layer_i.out_features)
    
    @torch.no_grad()
    def reset_matrices(self):
        for Ni in self.N:
            for N in Ni:
                if N is not None: N.zero_()
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for i in range(len(self.N)):
            if self.N[i] is None:
                continue
            for j in range(len(self.N[i])):
                if self.N[i][j] is None:
                    continue
                self.N[i][j] = self.N[i][j].to(*args, **kwargs)
        return self

    @torch.no_grad()
    def update_matrices(self):
        for i, layer_i in enumerate(self.model.layers):
            if self.back[i] != 'r':
                continue
            diff_J = layer_i.temporal_derivative()
            extend = layer_i.recurrent is not None

            if extend:
                self.N[i][i] @= diff_J.transpose(1, 2)
                tmp = self.N[i][i][:, : layer_i.linear.weight.numel(), :].view(self.batch_size, layer_i.in_features, layer_i.out_features, -1).diagonal(dim1=-2, dim2=-1)
                tmp += layer_i.input.unsqueeze(2)
                M_offset = layer_i.linear.weight.numel()
                if layer_i.recurrent is not None:
                    tmp = self.N[i][i][:, M_offset : M_offset + layer_i.recurrent.weight.numel(), :].view(self.batch_size, *layer_i.recurrent.weight.shape, -1).diagonal(dim1=-2, dim2=-1)
                    tmp += layer_i.old_spike.unsqueeze(2)
                    M_offset += layer_i.recurrent.weight.numel()
                if layer_i.linear.bias is not None:
                    tmp = self.N[i][i][:, M_offset : M_offset + layer_i.linear.bias.numel(), :].diagonal(dim1=1, dim2=2)
                    tmp += 1
            else:
                self.N[i][i] *= diff_J.unsqueeze(1)
                self.N[i][i][:, : layer_i.in_features, :] += layer_i.input.unsqueeze(2)
                if layer_i.linear.bias is not None:
                    self.N[i][i][:, -1, :] += 1
            
            if i == 0:
                continue

            p = -1
            diff_uu = layer_i.linear.weight.T.unsqueeze(0) * self.model.layers[i - 1].spatial_derivative(y='s', x='u').unsqueeze(2)
            for j in range(i - 1, -1, -1):
                layer_j = self.model.layers[j]
                
                if p == -1 and self.back[j] == 'r':
                    p = j
                
                if extend:
                    self.N[i][j] @= diff_J.transpose(1, 2)
                else:
                    self.N[i][j] *= diff_J.unsqueeze(1)
                
                if p != -1:
                    embed = p == j and layer_j.recurrent is None
                    if embed:
                        s_term = (self.N[p][j].unsqueeze(-1) * diff_uu.unsqueeze(1)).reshape(self.batch_size, -1, layer_i.out_features)
                    else:
                        s_term = self.N[p][j] @ diff_uu
                    self.N[i][j] += s_term
                else:
                    cated = [layer_j.input]
                    if layer_j.recurrent is not None:
                        cated.append(layer_j.old_spike)
                    if layer_j.linear.bias is not None:
                        cated.append(torch.ones(self.batch_size, 1, device=layer_j.input.device))
                    cated = torch.cat(cated, dim=1)
                    tmp = self.N[i][j].view(self.batch_size, -1, layer_j.out_features, layer_i.out_features)
                    tmp += diff_uu.unsqueeze(1) * cated.unsqueeze(-1).unsqueeze(-1)

                    if j > 0:
                        diff_uu = (layer_j.linear.weight.T.unsqueeze(0) * self.model.layers[j - 1].spatial_derivative(y='s', x='u').unsqueeze(2)) @ diff_uu

    @torch.no_grad()
    def assign_grad(self):
        grad_last = self.model.layers[self.last_r].mem.grad

        for i, layer in enumerate(self.model.layers[:self.last_r + 1]):
            if i == self.last_r and layer.recurrent is None:
                grads = torch.einsum('bji,bi->ji', self.N[self.last_r][i], grad_last).flatten()
            else:
                grads = torch.addbmm(torch.zeros(1, 1, device=grad_last.device), self.N[self.last_r][i], grad_last.unsqueeze(-1), beta=0).squeeze()
            
            self.grads[i].linear_weight += grads[: layer.linear.weight.numel()].view(layer.in_features, layer.out_features).T
            layer.linear.weight.grad = self.grads[i].linear_weight.clone()
            offset = layer.linear.weight.numel()
            if layer.recurrent is not None:
                self.grads[i].recurrent_weight += grads[offset : offset + layer.recurrent.weight.numel()].view_as(layer.recurrent.weight).T
                layer.recurrent.weight.grad = self.grads[i].recurrent_weight.clone()
                offset += layer.recurrent.weight.numel()
            if layer.linear.bias is not None:
                self.grads[i].linear_bias += grads[offset : offset + layer.linear.bias.numel()]
                layer.linear.bias.grad = self.grads[i].linear_bias.clone()
    
    def calc_grad(self, loss):
        if self.last_r == -1:
            loss.backward(retain_graph=True)
        else:
            self.model.layers[self.last_r].mem.retain_grad()
            assert self.model.layers[self.last_r].mem.grad is None or torch.allclose(self.model.layers[self.last_r].mem.grad, 0.)
            loss.backward(retain_graph=True)
            self.assign_grad()


class RestALIF(BackpropBase):
    def __init__(self, model, back, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        self.N = [None] * self.num_layers
        self.M = [None] * self.num_layers
        self.back = expand_conf(back, self.num_layers)
        
        self.last_r = self.back.rfind('r')

        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])
        
        for i, layer_i in enumerate(self.model.layers):
            self.N[i] = [None] * self.num_layers
            self.M[i] = [None] * self.num_layers
            if self.back[i] == 'r':
                M_size = 0
                extend = layer_i.recurrent is not None
                M_size += layer_i.linear.weight.numel() if extend else layer_i.in_features
                if layer_i.recurrent is not None:
                    M_size += layer_i.recurrent.weight.numel()
                if layer_i.linear.bias is not None:
                    M_size += layer_i.linear.bias.numel() if extend else 1
                self.N[i][i] = torch.zeros(self.batch_size, M_size, layer_i.out_features)
                self.M[i][i] = torch.zeros(self.batch_size, M_size, layer_i.out_features)

                for j, layer_j in enumerate(self.model.layers[:i]):
                    N_size = 0
                    N_size += layer_j.linear.weight.numel()
                    if layer_j.recurrent is not None:
                        N_size += layer_j.recurrent.weight.numel()
                    if layer_j.linear.bias is not None:
                        N_size += layer_j.linear.bias.numel()
                    self.N[i][j] = torch.zeros(self.batch_size, N_size, layer_i.out_features)
                    self.M[i][j] = torch.zeros(self.batch_size, N_size, layer_i.out_features)
    
    @torch.no_grad()
    def reset_matrices(self):
        for Ni in self.N:
            for N in Ni:
                if N is not None: N.zero_()
        for Mi in self.M:
            for M in Mi:
                if M is not None: M.zero_()
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for i in range(len(self.N)):
            if self.N[i] is None:
                continue
            for j in range(len(self.N[i])):
                if self.N[i][j] is None:
                    continue
                self.N[i][j] = self.N[i][j].to(*args, **kwargs)
        for i in range(len(self.M)):
            if self.M[i] is None:
                continue
            for j in range(len(self.M[i])):
                if self.M[i][j] is None:
                    continue
                self.M[i][j] = self.M[i][j].to(*args, **kwargs)
        return self

    @torch.no_grad()
    def update_matrices(self):
        for i, layer_i in enumerate(self.model.layers):
            if self.back[i] != 'r':
                continue
            diff_Juu = layer_i.temporal_derivative(y='u', x='u')
            diff_Jua = layer_i.temporal_derivative(y='u', x='a')
            diff_Jau = layer_i.temporal_derivative(y='a', x='u')
            diff_Jaa = layer_i.temporal_derivative(y='a', x='a')
            extend = layer_i.recurrent is not None

            if extend:
                self.N[i][i], self.M[i][i] = self.N[i][i] @ diff_Juu.transpose(1, 2) + self.M[i][i] @ diff_Jua.transpose(1, 2), \
                                             self.N[i][i] * diff_Jau.unsqueeze(1) + self.M[i][i] * diff_Jaa.unsqueeze(1)

                tmp = self.N[i][i][:, : layer_i.linear.weight.numel(), :].view(self.batch_size, layer_i.in_features, layer_i.out_features, -1).diagonal(dim1=-2, dim2=-1)
                tmp += layer_i.input.unsqueeze(2)
                M_offset = layer_i.linear.weight.numel()
                if layer_i.recurrent is not None:
                    tmp = self.N[i][i][:, M_offset : M_offset + layer_i.recurrent.weight.numel(), :].view(self.batch_size, *layer_i.recurrent.weight.shape, -1).diagonal(dim1=-2, dim2=-1)
                    tmp += layer_i.old_spike.unsqueeze(2)
                    M_offset += layer_i.recurrent.weight.numel()
                if layer_i.linear.bias is not None:
                    tmp = self.N[i][i][:, M_offset : M_offset + layer_i.linear.bias.numel(), :].diagonal(dim1=1, dim2=2)
                    tmp += 1
            else:
                self.N[i][i], self.M[i][i] = self.N[i][i] * diff_Juu.unsqueeze(1) + self.M[i][i] * diff_Jua.unsqueeze(1), \
                                             self.N[i][i] * diff_Jau.unsqueeze(1) + self.M[i][i] * diff_Jaa.unsqueeze(1)

                self.N[i][i][:, : layer_i.in_features, :] += layer_i.input.unsqueeze(2)
                if layer_i.linear.bias is not None:
                    self.N[i][i][:, -1, :] += 1
            
            if i == 0:
                continue

            p = -1
            diff_uu = layer_i.linear.weight.T.unsqueeze(0) * self.model.layers[i - 1].spatial_derivative(y='s', x='u').unsqueeze(2)
            diff_ua = layer_i.linear.weight.T.unsqueeze(0) * self.model.layers[i - 1].spatial_derivative(y='s', x='a').unsqueeze(2)
            for j in range(i - 1, -1, -1):
                layer_j = self.model.layers[j]
                if p == -1 and self.back[j] == 'r':
                    p = j
                
                if extend:
                    self.N[i][j], self.M[i][j] = self.N[i][j] @ diff_Juu.transpose(1, 2) + self.M[i][j] @ diff_Jua.transpose(1, 2), \
                                                 self.N[i][j] * diff_Jau.unsqueeze(1) + self.M[i][j] * diff_Jaa.unsqueeze(1)
                else:
                    self.N[i][j], self.M[i][j] = self.N[i][j] * diff_Juu.unsqueeze(1) + self.M[i][j] * diff_Jua.unsqueeze(1), \
                                                 self.N[i][j] * diff_Jau.unsqueeze(1) + self.M[i][j] * diff_Jaa.unsqueeze(1)
                
                if p != -1:
                    embed = p == j and layer_j.recurrent is None
                    if embed:
                        s_term = (self.N[p][j].unsqueeze(-1) * diff_uu.unsqueeze(1)).reshape(self.batch_size, -1, layer_i.out_features) + \
                                 (self.M[p][j].unsqueeze(-1) * diff_ua.unsqueeze(1)).reshape(self.batch_size, -1, layer_i.out_features)
                    else:
                        s_term = self.N[p][j] @ diff_uu + self.M[p][j] @ diff_ua
                    self.N[i][j] += s_term
                else:
                    cated = [layer_j.input]
                    if layer_j.recurrent is not None:
                        cated.append(layer_j.old_spike)
                    if layer_j.linear.bias is not None:
                        cated.append(torch.ones(self.batch_size, 1, device=layer_j.input.device))
                    cated = torch.cat(cated, dim=1)
                    tmp = self.N[i][j].view(self.batch_size, -1, layer_j.out_features, layer_i.out_features)
                    tmp += diff_uu.unsqueeze(1) * cated.unsqueeze(-1).unsqueeze(-1)

                    if j > 0:
                        diff_uu, diff_ua = (layer_j.linear.weight.T.unsqueeze(0) * self.model.layers[j - 1].spatial_derivative(y='s', x='u').unsqueeze(2)) @ diff_uu, \
                                           (layer_j.linear.weight.T.unsqueeze(0) * self.model.layers[j - 1].spatial_derivative(y='s', x='a').unsqueeze(2)) @ diff_uu

    @torch.no_grad()
    def assign_grad(self):
        grad_mem_last = self.model.layers[self.last_r].spike.grad * self.model.layers[self.last_r].spatial_derivative(y='s', x='u')
        grad_adapt_last = self.model.layers[self.last_r].spike.grad * self.model.layers[self.last_r].spatial_derivative(y='s', x='a')

        for i, layer in enumerate(self.model.layers[:self.last_r + 1]):
            if i == self.last_r and layer.recurrent is None:
                grads = torch.einsum('bji,bi->ji', self.N[self.last_r][i], grad_mem_last).flatten() + \
                        torch.einsum('bji,bi->ji', self.M[self.last_r][i], grad_adapt_last).flatten()
            else:
                grads = torch.addbmm(torch.zeros(1, 1, device=grad_mem_last.device), self.N[self.last_r][i], grad_mem_last.unsqueeze(-1), beta=0).squeeze() + \
                        torch.addbmm(torch.zeros(1, 1, device=grad_adapt_last.device), self.M[self.last_r][i], grad_adapt_last.unsqueeze(-1), beta=0).squeeze()
            
            self.grads[i].linear_weight += grads[: layer.linear.weight.numel()].view(layer.in_features, layer.out_features).T
            layer.linear.weight.grad = self.grads[i].linear_weight.clone()
            offset = layer.linear.weight.numel()
            if layer.recurrent is not None:
                self.grads[i].recurrent_weight += grads[offset : offset + layer.recurrent.weight.numel()].view_as(layer.recurrent.weight).T
                layer.recurrent.weight.grad = self.grads[i].recurrent_weight.clone()
                offset += layer.recurrent.weight.numel()
            if layer.linear.bias is not None:
                self.grads[i].linear_bias += grads[offset : offset + layer.linear.bias.numel()]
                layer.linear.bias.grad = self.grads[i].linear_bias.clone()
    
    def calc_grad(self, loss):
        if self.last_r == -1:
            loss.backward(retain_graph=True)
        else:
            # self.model.layers[self.last_r].mem.retain_grad()
            self.model.layers[self.last_r].spike.retain_grad()

            # assert self.model.layers[self.last_r].mem.grad is None or torch.allclose(self.model.layers[self.last_r].mem.grad, 0.)
            # assert self.model.layers[self.last_r].adapt.grad is None or torch.allclose(self.model.layers[self.last_r].adapt.grad, 0.)
            assert self.model.layers[self.last_r].spike.grad is None or torch.allclose(self.model.layers[self.last_r].spike.grad, 0.)

            loss.backward(retain_graph=True)
            self.assign_grad()


class RestDHLIF(BackpropBase):
    def __init__(self, model, back, create_bptt=False, temporal_detach_bptt=False):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        self.N = [None] * self.num_layers
        self.M = [None] * self.num_layers
        self.back = expand_conf(back, self.num_layers)
        
        self.last_r = self.back.rfind('r')

        if self.last_r != -1:
            self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])
        
        for i, layer_i in enumerate(self.model.layers):
            assert isinstance(layer_i, DHLIF)
            self.N[i] = [None] * self.num_layers
            self.M[i] = [None] * self.num_layers
            if self.back[i] == 'r':
                M_size = 0
                M_size += layer_i.linear.weight.numel()
                M_size += layer_i.recurrent.weight.numel()
                if layer_i.linear.bias is not None:
                    M_size += layer_i.linear.bias.numel()
                self.N[i][i] = torch.zeros(self.batch_size, M_size, layer_i.out_features)
                self.M[i][i] = torch.zeros(self.batch_size, M_size, layer_i.out_features * layer_i.branch)

                for j, layer_j in enumerate(self.model.layers[:i]):
                    N_size = 0
                    N_size += layer_j.linear.weight.numel()
                    N_size += layer_j.recurrent.weight.numel()
                    if layer_j.linear.bias is not None:
                        N_size += layer_j.linear.bias.numel()
                    self.N[i][j] = torch.zeros(self.batch_size, N_size, layer_i.out_features)
                    self.M[i][j] = torch.zeros(self.batch_size, N_size, layer_i.out_features * layer_i.branch)
    
    @torch.no_grad()
    def reset_matrices(self):
        for Ni in self.N:
            for N in Ni:
                if N is not None: N.zero_()
        for Mi in self.M:
            for M in Mi:
                if M is not None: M.zero_()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for i in range(len(self.N)):
            if self.N[i] is None:
                continue
            for j in range(len(self.N[i])):
                if self.N[i][j] is None:
                    continue
                self.N[i][j] = self.N[i][j].to(*args, **kwargs)
        for i in range(len(self.M)):
            if self.M[i] is None:
                continue
            for j in range(len(self.M[i])):
                if self.M[i][j] is None:
                    continue
                self.M[i][j] = self.M[i][j].to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def update_matrices(self):
        for i, layer_i in enumerate(self.model.layers):
            assert isinstance(layer_i, DHLIF)
            if self.back[i] != 'r':
                continue
            diff_Juu = layer_i.temporal_derivative(y='u', x='u') # (batch_size, out_features)
            diff_Jii = layer_i.temporal_derivative(y='i', x='i') # scalar
            diff_Jiu = layer_i.temporal_derivative(y='i', x='u') # (batch_size, out_features * branch, out_features)

            self.M[i][i] = diff_Jii * self.M[i][i] + self.N[i][i] @ diff_Jiu.transpose(1, 2)


            tmp = self.M[i][i][:, : layer_i.linear.weight.numel(), :].view(self.batch_size, layer_i.in_features, layer_i.out_features * layer_i.branch, -1).diagonal(dim1=-2, dim2=-1)
            tmp += layer_i.input.unsqueeze(2)
            M_offset = layer_i.linear.weight.numel()

            tmp = self.M[i][i][:, M_offset : M_offset + layer_i.recurrent.weight.numel(), :].view(self.batch_size, layer_i.out_features, layer_i.out_features * layer_i.branch, -1).diagonal(dim1=-2, dim2=-1)
            tmp += layer_i.old_spike.unsqueeze(2)
            M_offset += layer_i.recurrent.weight.numel()

            if layer_i.linear.bias is not None:
                tmp = self.M[i][i][:, M_offset : M_offset + layer_i.linear.bias.numel(), :].diagonal(dim1=1, dim2=2)
                tmp += 1

            self.N[i][i] = self.N[i][i] * diff_Juu.unsqueeze(1) + self.M[i][i].view(self.batch_size, -1, layer_i.out_features, layer_i.branch).sum(dim=3)
            
            if i == 0:
                continue

            p = -1
            diff_iu = layer_i.linear.weight.T.unsqueeze(0) * self.model.layers[i - 1].spatial_derivative(y='s', x='u').unsqueeze(2)

            for j in range(i - 1, -1, -1):
                layer_j = self.model.layers[j]
                if p == -1 and self.back[j] == 'r':
                    p = j
                
                
                self.M[i][j] = diff_Jii * self.M[i][j] + self.N[i][j] @ diff_Jiu.transpose(1, 2)

                if p != -1:
                    self.M[i][j] += self.N[p][j] @ diff_iu
                else:
                    cated = [layer_j.input, layer_j.old_spike]
                    if layer_j.linear.bias is not None:
                        cated.append(torch.ones(self.batch_size, 1, device=layer_j.input.device))
                    cated = torch.cat(cated, dim=1)
                    # tmp = self.M[i][j].view(self.batch_size, -1, layer_j.out_features * layer_j.branch, layer_i.out_features * layer_i.branch)
                    # tmp += (diff_iu.unsqueeze(1) * cated.unsqueeze(-1).unsqueeze(-1)).unsqueeze()
                    # tmp(b, i, j1*jb, k1*kb) += diff_iu(b, j1, k1*kb) * cated(b, i)
                    tmp = self.M[i][j].view(self.batch_size, -1, layer_j.branch, layer_i.out_features * layer_i.branch)
                    tmp += (diff_iu.unsqueeze(1) * cated.unsqueeze(-1).unsqueeze(-1)).contiguous().view(self.batch_size, -1, layer_i.out_features * layer_i.branch).unsqueeze(2)

                    assert isinstance(layer_j, DHLIF)

                    if j > 0:
                        diff_iu = (layer_j.linear.weight.T.view(layer_j.in_features, layer_j.out_features, layer_j.branch).sum(dim=2).unsqueeze(0) * self.model.layers[j - 1].spatial_derivative(y='s', x='u').unsqueeze(2)) @ diff_iu
                
                self.N[i][j] = self.N[i][j] * diff_Juu.unsqueeze(1) + self.M[i][j].view(self.batch_size, -1, layer_i.out_features, layer_i.branch).sum(dim=3)

    @torch.no_grad()
    def assign_grad(self):
        grad_last = self.model.layers[self.last_r].mem.grad

        for i, layer in enumerate(self.model.layers[:self.last_r + 1]):
            grads = torch.addbmm(torch.zeros(1, 1, device=grad_last.device), self.N[self.last_r][i], grad_last.unsqueeze(-1), beta=0).squeeze()
            
            self.grads[i].linear_weight += grads[: layer.linear.weight.numel()].view(layer.in_features, layer.out_features * layer.branch).T
            layer.linear.weight.grad = self.grads[i].linear_weight.clone() * layer.mask[:, :layer.in_features]
            offset = layer.linear.weight.numel()

            self.grads[i].recurrent_weight += grads[offset : offset + layer.recurrent.weight.numel()].view(layer.out_features, layer.out_features * layer.branch).T
            layer.recurrent.weight.grad = self.grads[i].recurrent_weight.clone() * layer.mask[:, layer.in_features:]
            offset += layer.recurrent.weight.numel()

            if layer.linear.bias is not None:
                self.grads[i].linear_bias += grads[offset : offset + layer.linear.bias.numel()]
                layer.linear.bias.grad = self.grads[i].linear_bias.clone()
    
    def calc_grad(self, loss):
        if self.last_r == -1:
            loss.backward(retain_graph=True)
        else:
            self.model.layers[self.last_r].mem.retain_grad()
            assert self.model.layers[self.last_r].mem.grad is None or torch.allclose(self.model.layers[self.last_r].mem.grad, 0.)
            loss.backward(retain_graph=True)
            self.assign_grad()


def get_rests(model, back='r', create_bptt=False, temporal_detach_bptt=False, **kwargs) -> BackpropBase:
    if not all(model.temporal_detach):
        raise ValueError("REST-S requires all temporal gradients to be detached in the feedforward pass.")
    
    class_dict = {
        'lif': RestLIF,
        'alif': RestALIF,
        'dhlif': RestDHLIF,
    }
    return class_dict[model.neuron_type](model, back, create_bptt, temporal_detach_bptt, **kwargs)


def get_bp(model, create_bptt=False, temporal_detach_bptt=False) -> BackpropBase:
    if not all(model.temporal_detach):
        raise ValueError("BP requires all temporal gradients to be detached in the feedforward pass.")
    
    class_dict = {
        'lif': RestLIF,
        'alif': RestALIF,
        'dhlif': RestDHLIF,
    }
    return class_dict[model.neuron_type](model, 'b', create_bptt, temporal_detach_bptt)
