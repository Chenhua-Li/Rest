import torch
from torch import nn
from torch.autograd import Function


class Activate(Function):
    @staticmethod
    def forward(ctx, input, lens):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return input.gt(0.).float()
    
    @staticmethod
    def backward(ctx, grad):
        input, = ctx.saved_tensors
        lens = ctx.lens
        grad_input = grad.clone()
        return grad_input * surrogate_grad(input, lens), None


def surrogate_grad(x, lens):
    return (torch.abs(x) < lens).float() / (2 * lens)
    # return (abs(x) < lens).float()

