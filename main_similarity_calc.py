import os
import sys
import torch
import numpy as np
from scipy.spatial.distance import cdist
from models import *
from tasks import *
from utils import setup_seed
from argparse import ArgumentParser


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def calc_dist(model_a, model_b, mode):
    grad_a = torch.concat([p.grad.flatten() for p in model_a.parameters()])
    grad_b = torch.concat([p.grad.flatten() for p in model_b.parameters()])
    dist = cdist(grad_a.unsqueeze(0).cpu(), grad_b.unsqueeze(0).cpu(), mode)[0][0]
    # scale = 10000
    # dist_scale = cdist(grad_a.unsqueeze(0).cpu() * scale, grad_b.unsqueeze(0).cpu() * scale, mode)[0][0]
    # dist_max = cdist(grad_a.unsqueeze(0).cpu() / grad_a.abs().max().cpu().item(), grad_b.unsqueeze(0).cpu() / grad_b.abs().max().cpu().item(), mode)[0][0]
    # print(grad_a.abs().max().cpu().item(), grad_b.abs().max().cpu().item(), dist, dist_scale, dist_max, flush=True)
    return dist

def calc_similarity(model, mode='cosine', layers_separated=False):
    if layers_separated:
        ret = []
        for layer_online, layer_bptt in zip(model.model.layers, model.bptt_model.layers):
            dist = calc_dist(layer_online, layer_bptt, mode)
            ret.append(dist)
        dist = calc_dist(model.model.classifier, model.bptt_model.classifier, mode)
        ret.append(dist)
        return ret
    else:
        dist = calc_dist(model.model, model.bptt_model, mode)
        return ret

def run(dataset,
        neuron_type,
        algo,
        device,
        branch,
        # filename,
        seed,
        num_hidden_layers,
        back,
        args,
        ):

    task_dict = {
        'smnist': SMNISTTask,
        'nmnist': NMNISTTask,
        'shd': SHDTask,
        'scifar10': SCIFAR10Task,
        'adding': AddingTask,
    }
    neuron_nums_dict = {
        'smnist': [1] + [128] * num_hidden_layers + [10],        # [TODO] check again!
        'nmnist': [34*34*2] + [128] * num_hidden_layers + [10],
        'shd': [700] + [128] * num_hidden_layers + [20],
        'scifar10': [96] + [128] * num_hidden_layers + [10],
        'adding': [2] + [64] * num_hidden_layers + [10]
    }
    kwargs = {
        'lif': {
            'decay': 0.5,
            'thresh': 0.3,
        },
        'alif': { #decay=0.9, thresh0=0.8, rho=0.99, beta=0.15
            'decay': 0.9,
            'thresh0': 0.8,
            'rho': 0.99,
            'beta': 0.15,
        },
        'dhlif': {# DHLIF: thresh=1.6, (alpha=beta=0.8)
            'thresh': 1.6,
            'branch': branch,
        },
    }
    task_kwargs = {} if dataset != 'adding' else {'seq_len': 100}
    task = task_dict[dataset](**task_kwargs)

    if seed is not None:
        setup_seed(seed)

    snn = SNN_Model(batch_size=1,
                    neuron_nums=neuron_nums_dict[dataset],
                    recurrent=bool(args.recurrent), # optional
                    bias=(algo != 'eprop'), # [NOTE] only in eprop
                    neuron_type=neuron_type,
                    **kwargs[neuron_type]
                    )
    
    algo_kwargs = {} if algo not in ['rests', 'restus'] else {'back': back}
    if algo in ['restu', 'restus']:
        algo_kwargs['num'] = 15
        algo_kwargs['init'] = 'randn'
    snn = globals()[f'get_{algo}'](snn, create_bptt=True, **algo_kwargs).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()
    # if os.path.exists(filename):
    #     data = dict(np.load(filename))
    # else:
    #     data = dict()
    
    # if data.get(f'seed_{seed}', None) is not None:
    #     similarity = data[f'seed_{seed}'].tolist()
    # else:
    #     similarity = []
    similarity = []


    task.prepare_dataloader(batch_size=1)

    for batch_idx, (inputs, target) in enumerate(task.test_loader):
        if batch_idx >= 100:
            break

        if len(similarity) > batch_idx:
            continue

        inputs, target = task.preprocess_data(inputs, target)
        inputs = inputs.to(device)
        target = target.to(device)
        # target = torch.nn.functional.one_hot(target, num_classes=neuron_nums_dict[dataset][-1]).float()

        snn.zero_grad()

        for i in range(task.get_time_window()):
            out_online, out_bptt = snn(inputs[i], time_step=i)
            target_i = target[i] if dataset == 'adding' else target
            loss_online, loss_bptt = criterion(out_online, target_i), criterion(out_bptt, target_i)
            snn.calc_grad(loss_online)
            loss_bptt.backward(retain_graph=True)
        
        tmp = calc_similarity(snn, layers_separated=True)
        similarity.append(tmp)

        # data[f'seed_{seed}'] = np.array(similarity)

        # np.savez(filename, **data)

        print([1 - x for x in tmp])



arg = ArgumentParser()
arg.add_argument('--dataset', type=str, choices=['smnist', 'nmnist', 'shd', 'scifar10', 'adding'])
arg.add_argument('--neuron', type=str, choices=['lif', 'alif', 'dhlif'])
arg.add_argument('--recurrent', type=int, default=1)
arg.add_argument('--algo', type=str, choices=['rests', 'bp', 'restu', 'restus', 'ppprop', 'ostl', 'ottt', 'eprop'])
arg.add_argument('--back', type=str, default='r')
arg.add_argument('--node', type=int, default=0)
arg.add_argument('--branch', type=int, default=2)
arg.add_argument('--N', type=int, default=3)
args = arg.parse_args()


# filename = f'./similarity_results/{args.N}/{args.dataset}/{args.neuron}{str(args.branch) if args.neuron == "dhlif" else ""}_{args.algo}.npz'

# os.makedirs(os.path.dirname(filename), exist_ok=True)
# print(filename, flush=True)

for s in range(10):
    run(
        dataset=args.dataset,
        neuron_type=args.neuron,
        algo=args.algo,
        device=torch.device(f'cuda:{args.node}'),
        branch=args.branch,
        # filename=filename,
        seed=s,
        num_hidden_layers=args.N,
        back=args.back,
        args=args,
    )
