import torch
from torch import nn
from models.base import BackpropBase, Grad
from neurons import *


class UoroModule(nn.Module):
    def __init__(self, layer: LIF, num=1,init="binary"):
        super().__init__()
        assert isinstance(layer, LIF)
        # self.device = device
        self.num = num
        self.batch_size = layer.batch_size
        self.n_neurons = layer.out_features
        self.init=init
        if layer.recurrent is None:
            self.input_size = layer.in_features
        else:
            self.input_size = layer.in_features + layer.out_features
        
        if layer.linear.bias is not None:
            self.input_size += 1
        
        self.init_scale = 1e-1
        self.register_buffer('A', torch.zeros(self.batch_size, self.n_neurons, self.num), persistent=False)
        self.register_buffer('B', torch.zeros(self.batch_size, self.input_size, self.num), persistent=False)
        self.register_buffer('nu', torch.empty(self.batch_size, self.n_neurons, self.num), persistent=False)

    @torch.no_grad()
    def reset_uoro(self):
        self.A.detach_().zero_()
        self.B.detach_().zero_()
        self.nu.detach_().zero_()
        if self.init == "binary":
            self.nu=(torch.randint(0,2,self.nu.shape, dtype=torch.float32)*2-1).to(self.A.device)
        elif self.init == "randn":
            self.nu=torch.randn(self.nu.shape, dtype=torch.float32).to(self.A.device)
        elif self.init == "ort":
            torch.nn.init.orthogonal_(self.nu)
            self.nu = self.nu - self.nu.mean(dim=1, keepdim=True)        # self.nu = self.nu - self.nu.mean(dim=1, keepdim=True)
        # self.nu=(torch.randint(0,2,self.nu.shape, dtype=torch.float32)*2-1).to(self.A.device)

    @torch.no_grad()
    def update_uoro(self, m_t, df_mem):
        # Compute Jacobian J(t)
        bs = self.batch_size

        if df_mem.dim() == 3:
            A_forward = df_mem @ self.A
        else:
            A_forward = df_mem.unsqueeze(2) * self.A
        # A_forward = df_mem.unsqueeze(-1)

        B_norm = torch.norm(self.B, dim=1)
        A_norm = torch.norm(A_forward, dim=1)
        M_projection = torch.einsum('bik,bj->bjk', self.nu, m_t)
        M_norm = torch.norm(M_projection, dim=1)
        nu_norm = torch.norm(self.nu, dim=1)

        epsilon = 1e-7
        p0 = torch.sqrt((B_norm + epsilon) / (A_norm + epsilon)).reshape(-1, self.num) + epsilon
        p1 = torch.sqrt((M_norm + epsilon) / (nu_norm + epsilon)).reshape(-1, self.num) + epsilon

        # Update A and B
        self.A = p0[:, None, :] * A_forward + p1[:, None, :] * self.nu
        self.B = (1. / p0[:, None, :]) * self.B + (1. / p1[:, None, :]) * M_projection

        if self.init == "binary":
            self.nu=(torch.randint(0,2,self.nu.shape, dtype=torch.float32)*2-1).to(self.A.device)
        elif self.init == "randn":
            self.nu=torch.randn(self.nu.shape, dtype=torch.float32).to(self.A.device)
        elif self.init == "ort":
            torch.nn.init.orthogonal_(self.nu)
            self.nu = self.nu - self.nu.mean(dim=1, keepdim=True)

        return torch.einsum('bik,bjk->bij', self.A,self.B)

class Uoro(BackpropBase):
    def __init__(self, model, create_bptt=False, temporal_detach_bptt=False,num=1,init="binary"):
        super().__init__(model, create_bptt, temporal_detach_bptt)

        # self.grads = nn.ModuleList([class UoroModule(nn.Module):(layer) for layer in self.model.layers])

        self.U = nn.ModuleList([UoroModule(layer, num=num,init=init) for layer in self.model.layers[:]])
        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers])

        self.M = [torch.zeros(self.batch_size, layer.in_features + (layer.out_features if layer.recurrent is not None else 0) + (1 if layer.linear.bias is not None else 0)) for layer in self.model.layers]

    @torch.no_grad()
    def reset_matrices(self):
        for Mi in self.M:
            Mi.zero_()
        for U in self.U:
            U.reset_uoro()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for i in range(len(self.M)):
            self.M[i] = self.M[i].to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def update_matrices(self):
        for i, layer in enumerate(self.model.layers):
            assert isinstance(layer, LIF)
            diff_J = layer.temporal_derivative()
            
            vars = [layer.input]
            if layer.recurrent is not None:
                vars.append(layer.old_spike)
            if layer.linear.bias is not None:
                vars.append(torch.ones(self.batch_size, 1, device=layer.input.device))
            m_t = torch.concat(vars, dim=1)

            self.M[i]=self.U[i].update_uoro(m_t, diff_J)

            # if layer.recurrent is None:
            #     self.M[i] = diff_J.unsqueeze(2) * self.M[i]
            # else:
            #     self.M[i] = diff_J @ self.M[i]

         
            # mem_per=layer.old_mem+eps*self.U[i].A.squeeze()

            # mem_perdict =layer.update(layer.input,mem_per)
            
            # self.M[i]=self.U[i].update_uoro(m_t, (mem_perdict-layer.mem)/eps)
            
            # self.M[i] += m_t.unsqueeze(1)


    @torch.no_grad()
    def assign_grad(self):
        for i, layer in enumerate(self.model.layers):
            grad = torch.einsum('bi,bij->ij', layer.mem.grad, self.M[i])
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




def get_uoro(model, create_bptt=False, temporal_detach_bptt=False,num=1,init="binary"):
    assert model.neuron_type == 'lif'
    return Uoro(model, create_bptt, temporal_detach_bptt,num,init)

