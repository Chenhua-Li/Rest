import torch
from torch import nn
from models.base import BackpropBase, Grad
from neurons import *
from utils import expand_conf


class UoroModule(nn.Module):
    def __init__(self, layer: LIF, num=1, init="binary"):
        super().__init__()
        assert isinstance(layer, LIF)
        # self.device = device
        self.num = num
        self.batch_size = layer.batch_size
        self.n_neurons = layer.out_features
        self.input_size = layer.in_features
        self.init = init
        if layer.recurrent is not None:
            self.input_size += layer.out_features

        if layer.linear.bias is not None:
            self.input_size += 1

        self.register_buffer('A', torch.zeros(self.batch_size, self.n_neurons, self.num), persistent=False)
        self.register_buffer('B', torch.zeros(self.batch_size, self.input_size, self.num), persistent=False)
        self.register_buffer('nu', torch.empty(self.batch_size, self.n_neurons, self.num), persistent=False)

    @torch.no_grad()
    def reset_uoro(self):
        self.A.detach_().zero_()
        self.B.detach_().zero_()
        # self.nu.detach_().zero_()

        if self.init == "binary":
            self.nu = (torch.randint(0, 2, self.nu.shape, dtype=torch.float32) * 2 - 1).to(self.A.device)
        elif self.init == "randn":
            self.nu = torch.randn(self.nu.shape, dtype=torch.float32).to(self.A.device)
        elif self.init == "ort":
            inits = torch.randn(self.n_neurons,int(self.batch_size*self.num),  dtype=torch.float32).to(self.A.device)
            torch.nn.init.orthogonal_(inits)
            self.nu = inits.reshape(self.n_neurons,self.batch_size, self.num).permute(1,0,2)
            # self.nu = self.nu - self.nu.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def update_uoro(self, m_t, df_mem):
        # Compute Jacobian J(t)
        bs = self.batch_size

        if df_mem.dim() == 3:
            A_forward = df_mem @ self.A
        else:
            A_forward = df_mem.unsqueeze(2) * self.A

        epsilon = 1e-6
        # todo: differences
        nume = (self.nu * A_forward).sum(dim=1)
        denom = (A_forward.pow(2)).sum(dim=1)+epsilon
        alpha = (nume/denom).clamp(-1e5,1e5)
        M_projection = m_t.unsqueeze(2) * alpha.unsqueeze(1)

        # M_projection = torch.einsum('bik,bj->bjk', self.nu, m_t)
        M_norm = torch.norm(M_projection, dim=1)
        nu_norm = torch.norm(self.nu, dim=1)
        #

        B_norm = torch.norm(self.B, dim=1)
        A_norm = torch.norm(A_forward, dim=1)


        p0 = torch.sqrt((B_norm + epsilon) / (A_norm + epsilon)).reshape(-1, self.num) + epsilon
        p1 = torch.sqrt((M_norm + epsilon) / (nu_norm + epsilon)).reshape(-1, self.num) + epsilon
        p0 = p0.clamp(1e-3,1e3)
        p1 = p1.clamp(1e-3, 1e3)
        # Update A and B
        self.A = p0[:, None, :] * A_forward + p1[:, None, :] * self.nu
        self.B = (1. / p0[:, None, :]) * self.B + (1. / p1[:, None, :]) * M_projection

        if self.init == "binary":
            self.nu = (torch.randint(0, 2, self.nu.shape, dtype=torch.float32) * 2 - 1).to(self.A.device)
        elif self.init == "randn":
            self.nu = torch.randn(self.nu.shape, dtype=torch.float32).to(self.A.device)
        elif self.init == "ort":
            inits = torch.randn(self.n_neurons,int(self.batch_size*self.num),  dtype=torch.float32).to(self.A.device)
            torch.nn.init.orthogonal_(inits)
            self.nu = inits.reshape(self.n_neurons,self.batch_size, self.num).permute(1,0,2)


class RestUS(BackpropBase):
    def __init__(self, model, back, create_bptt=False, temporal_detach_bptt=False, num=1, init="binary"):
        super().__init__(model, create_bptt, temporal_detach_bptt)
        assert model.neuron_type == 'lif'
        self.back = expand_conf(back, self.num_layers)
        self.last_r = self.back.rfind('r')
        # print('last_r', self.last_r)
        # print('back', self.back)

        # print('uoro sample number',num)
        # print('init', init)
        self.U = nn.ModuleList([UoroModule(layer, num=num, init=init) for layer in self.model.layers[:self.last_r + 1]])
        self.M = [None] * self.num_layers
        self.N = [None] * self.num_layers
        self.dim = num

        self.grads = nn.ModuleList([Grad(layer) for layer in self.model.layers[:self.last_r + 1]])

        for i, layer_i in enumerate(self.model.layers):
            assert isinstance(layer_i, LIF)
            # use accurate intra-layer grads if no recurrent connections
            # todo
            self.N[i] = [None] * self.num_layers

            if self.back[i] == 'b':
                continue

            if layer_i.recurrent is None:
                self.M[i] = torch.zeros(self.batch_size,
                                        layer_i.out_features,
                                        layer_i.in_features + (0 if layer_i.linear.bias is None else 1),
                                        )
            else:
                self.M[i] = torch.zeros(self.batch_size,
                                        layer_i.out_features,
                                        layer_i.in_features  + layer_i.out_features+ (0 if layer_i.linear.bias is None else 1),
                                        )
            # otherwise, use approximate ones (UORO)

            for j, layer_j in enumerate(self.model.layers[:i]):
                assert isinstance(layer_j, LIF)
                self.N[i][j] = torch.zeros(self.batch_size, layer_i.out_features, layer_j.out_features, num)

    @torch.no_grad()
    def reset_matrices(self):
        for U in self.U:
            U.reset_uoro()
        for M in self.M:
            if M is not None: M.zero_()
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
        for i in range(len(self.M)):
            if self.M[i] is None:
                continue
            self.M[i] = self.M[i].to(*args, **kwargs)
        return self

    @torch.no_grad()
    def update_matrices(self):
        if self.last_r == -1:
            return
        
        diff_Js = [None] * self.num_layers

        for i, layer in enumerate(self.model.layers[:self.last_r + 1]):
            assert isinstance(layer, LIF)

            diff_J = layer.temporal_derivative()
            vars = [layer.input]
            if layer.recurrent is not None:
                vars.append(layer.old_spike)
            if layer.linear.bias is not None:
                vars.append(torch.ones(self.batch_size, 1, device=layer.input.device))
            m_t = torch.concat(vars, dim=1)

            if self.back[i] == 'r':
                if layer.recurrent is None:
                    self.M[i] = diff_J.unsqueeze(2) * self.M[i] + m_t.unsqueeze(1)
                else:
                    diff_J_temp = layer.temporal_simplified()
                    self.M[i] = diff_J_temp.unsqueeze(2) * self.M[i] + m_t.unsqueeze(1)


            self.U[i].update_uoro(m_t, diff_J)

            diff_Js[i] = diff_J

        self.part1s = [None] * self.last_r
        for i, layer_i in enumerate(self.model.layers[1:], 1):
            assert isinstance(layer_i, LIF)

            if self.back[i] == 'b':
                continue

            diff_J = diff_Js[i]
            p = -1
            diff_uu = layer_i.linear.weight.unsqueeze(0) * self.model.layers[i - 1].spatial_derivative(y='s', x='u').unsqueeze(1)

            for j in range(i - 1, -1, -1):
                layer_j = self.model.layers[j]

                if p == -1 and self.back[j] == 'r':
                    p = j
                
                if layer_i.recurrent is None:
                    part1 = torch.einsum('bi,bick->bick', diff_J, self.N[i][j])
                else:
                    part1 = torch.einsum('bij,bjck->bick', diff_J, self.N[i][j])
                
                if p == -1 or p == j:
                # if i == j + 1:
                    part2 = torch.einsum('bij,bjk->bijk', diff_uu, self.U[j].A)
                else:
                    # part2 = torch.einsum('bij,bjck->bick', diff_uu, self.N[i - 1][j])
                    part2 = torch.einsum('bij,bjck->bick', diff_uu, self.N[p][j])

                if p == -1 and j > 0:
                    diff_uu @= layer_j.linear.weight.unsqueeze(0) * self.model.layers[j - 1].spatial_derivative(y='s', x='u').unsqueeze(1)

                self.N[i][j] = part1 + part2

                if i == self.last_r:
                    self.part1s[j] = part1.clone()

        del diff_Js

    @torch.no_grad()
    def assign_grad(self):
        grad_last = self.model.layers[self.last_r].mem.grad

        for i, layer in enumerate(self.model.layers[:self.last_r + 1]):
            assert isinstance(layer, LIF)
            if i == self.last_r: # back[i] == r always holds
                grads = torch.einsum('bi,bij->ij', grad_last, self.M[i])
            else:
                if self.back[i] == 'r':
                    grads = torch.einsum('bi,bij->ij', layer.mem.grad, self.M[i]) + \
                            torch.einsum('bc,bcik,bjk->ij', grad_last, self.part1s[i], self.U[i].B) / self.dim
                else:
                    grads = torch.einsum('bc,bcik,bjk->ij', grad_last, self.N[self.last_r][i], self.U[i].B) / self.dim

            self.grads[i].linear_weight += grads[:, :layer.in_features]
            layer.linear.weight.grad = self.grads[i].linear_weight.clone()
            if layer.recurrent is not None:
                self.grads[i].recurrent_weight += grads[:, layer.in_features:(layer.in_features + layer.out_features)]
                layer.recurrent.weight.grad = self.grads[i].recurrent_weight.clone()
            if layer.linear.bias is not None:
                self.grads[i].linear_bias += grads[:, -1]
                layer.linear.bias.grad = self.grads[i].linear_bias.clone()

    def calc_grad(self, loss):
        if self.last_r == -1:
            loss.backward(retain_graph=True)
        else:
            for layer in self.model.layers[:self.last_r + 1]:
                layer.mem.retain_grad()
                assert layer.mem.grad is None or torch.allclose(layer.mem.grad, 0.)
            loss.backward(retain_graph=True)
            self.assign_grad()



def get_restus(model, back, create_bptt=False, temporal_detach_bptt=False, num=1, init="randn") -> RestUS:
    assert model.neuron_type == 'lif'
    return RestUS(model, back, create_bptt, temporal_detach_bptt, num, init)

