import os
import torch
from argparse import ArgumentParser
import time
from models import *
from utils import *
from tasks import *
from interfaces import *
from distutils.util import strtobool # type: ignore

import torch.distributed as dist            
from torch.nn.parallel import DistributedDataParallel




neuron_hyper_list = {
    'lif': ['decay', 'thresh'],
    'alif': ['decay', 'thresh0', 'rho', 'beta'],
    'dhlif': ['thresh', 'branch'],
}

def back_check(x: str):
    x = x.lower()
    assert x.count('r') + x.count('b') == len(x)
    return x


def parse_config():
    argparser = ArgumentParser()

    # !important
    argparser.add_argument('--dataset', type=str, choices=['nmnist', 'smnist', 'scifar10', 'shd','ssc'])
    argparser.add_argument('--algo', type=str, choices=['rests', 'restu', 'restus', 'bptt', 'bp', 'uoro', 'ppprop', 'ostl', 'ottt', 'eprop'])
    argparser.add_argument('--neuron', default='lif', type=str, choices=['lif', 'alif', 'dhlif'])
    argparser.add_argument('--back', type=back_check)
    # argparser.add_argument('--node', type=int, default=0)

    # set by program, do not pass in command line
    argparser.add_argument('--name', type=str)
    argparser.add_argument('--batch-size', type=int)
    argparser.add_argument('--cfg', type=int, nargs='+')
    argparser.add_argument('--recurrent', type=lambda x: bool(strtobool(str(x))))
    argparser.add_argument('--readout', type=str, choices=['linear', 'spike', 'potential', 'potential_softmax'])
    argparser.add_argument('--cumsum', type=lambda x: bool(strtobool(str(x))))
    argparser.add_argument('--online-update', type=lambda x: bool(strtobool(str(x))))
    argparser.add_argument('--num', type=int, default=1)
    argparser.add_argument('--init', type=str,default='binary', choices=['binary', 'randn', 'ort'])
    argparser.add_argument('--epochs', type=int)
    argparser.add_argument('--lr', type=float)
    argparser.add_argument('--optim', type=str, choices=['adam', 'sgd', 'adamw'])
    argparser.add_argument('--scheduler', type=str, choices=['reduce-lr-on-plateau', 'cosine-annealing', 'step-lr', 'none'])

    argparser.add_argument('--decay', type=float)
    argparser.add_argument('--thresh', type=float)
    argparser.add_argument('--thresh0', type=float)
    argparser.add_argument('--beta', type=float)
    argparser.add_argument('--rho', type=float)
    argparser.add_argument('--branch', type=int)
    # argparser.add_argument('--lens', type=float, default=0.5)

    
    # argparser.add_argument('--step-size', type=int, default=50)
    # argparser.add_argument('--gamma', type=float, default=0.8)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--loss', type=str, default='ce', choices=['mse', 'ce'])
    argparser.add_argument('--allow-tf32', type=lambda x: bool(strtobool(str(x))), default=True)
    # argparser.add_argument('--grad-clip', type=float, default=0.0)
    # argparser.add_argument('--report-iter', type=int, default=10)
    # argparser.add_argument('--save-iter', action='store_true', default=False)

    # argparser.add_argument('--resume', default='')

    args = argparser.parse_args()

    default_values = {
        'nmnist': {
            'cfg': [34*34*2, 800, 10], # N/A
            # 'cfg': [34*34*2, 128, 128, 10], # bs=16, DHLIF-1 20547 MiB (for LIF, half)
            'recurrent': False,
            'readout': 'linear',
            'cumsum': False,
            'online_update': True,
            'batch_size': 256,
            'epochs': 100,
            'lr': 1e-4,
            'optim': 'adam',
            'scheduler': 'none',
            'neuron_hyper': {
                'lif': {
                    'decay': 0.9,
                    'thresh': 0.8,
                },
                'alif': {
                    'decay': 0.9,
                    'thresh0': 0.8,
                    'rho': 0.99,
                    'beta': 0.15,
                },
                'dhlif': {
                    'thresh': 1.6,
                    'branch': 1,
                },
            }
        },
        'smnist': {
            'cfg': [1, 64, 256, 256, 10], # bs=16, DHLIF-1 22153 MiB (for LIF, half)
            'recurrent': True,
            'readout': 'linear',
            'cumsum': False,
            'online_update': True,
            'batch_size': 32,
            'epochs': 100,
            'lr': 1e-4,
            'optim': 'adam',
            'scheduler': 'none',
            'neuron_hyper': {
                'lif': {
                    'decay': 0.9,
                    'thresh': 0.8,
                },
                'alif': {
                    'decay': 0.9,
                    'thresh0': 0.8,
                    'rho': 0.99,
                    'beta': 0.15,
                },
                'dhlif': {
                    'thresh': 1.6,
                    'branch': 1,
                },
            }
        },
        'scifar10': {
            'cfg': [96, 128, 256, 256, 10],
            'recurrent': True,
            'readout': 'linear',
            'cumsum': True,
            'online_update': False,
            'batch_size': 32,
            'epochs': 100,
            'lr': 1e-4,
            'optim': 'adam',
            'scheduler': 'cosine-annealing',
            'neuron_hyper': {
                'lif': {
                    'decay': 0.9,
                    'thresh': 0.8,
                },
                'alif': {
                    'decay': 0.9,
                    'thresh0': 0.8,
                    'rho': 0.99,
                    'beta': 0.15,
                },
                'dhlif': {
                    'thresh': 1.6,
                    'branch': 1,
                },
            }
        },
        'shd': {
            'cfg': [700, 128, 128, 128, 20], # bs=16, DHLIF-1 10351 MiB (for LIF, half)
            'recurrent': True,
            'readout': 'linear',
            'cumsum': True,
            'online_update': False,
            'batch_size': 64,
            'epochs': 200,
            'lr': 3e-4,
            'optim': 'adam',
            'scheduler': 'none',
            'neuron_hyper': {
                'lif': {
                    # 'decay': 0.9,
                    # 'thresh': 1.0,
                    'decay': 0.6,
                    'thresh': 0.5,
                },
                'alif': {
                    'decay': 0.9,
                    'thresh0': 0.8,
                    'rho': 0.99,
                    'beta': 0.15,
                },
                'dhlif': {
                    'thresh': 1.6,
                    'branch': 1,
                },
            }
        },
        'ssc': {
            'cfg': [700, 128, 128, 128, 35], 
            'recurrent': True,
            'readout': 'linear',
            'cumsum': True,
            'online_update': False,
            'num': 1,
            'init': 'binary',
            'batch_size': 128,
            'epochs': 300,
            'lr': 1e-4,
            'optim': 'adam',
            'scheduler': 'none',
            'neuron_hyper': {
                'lif': {
                    # 'decay': 0.9,
                    # 'thresh': 1.0,
                    'decay': 0.6,
                    'thresh': 0.5,
                },
                'alif': {
                    'decay': 0.9,
                    'thresh0': 0.8,
                    'rho': 0.99,
                    'beta': 0.15,
                },
                'dhlif': {
                    'thresh': 1.6,
                    'branch': 1,
                },
            }
        }
    }

    if args.algo == 'bptt':
        if args.online_update == True:
            raise ValueError("bptt cannot use online update")
        args.online_update = False
    
    if args.algo in ['rests', 'restus']:
        if args.back is None:
            args.back = 'r'
    else:
        if args.back is not None:
            raise ValueError("back cannot be set when algo is not REST-S or REST-US")
        delattr(args, 'back')
    
    for attr in ['cfg', 'recurrent', 'batch_size', 'epochs', 'lr', 'optim', 'scheduler', 'readout', 'cumsum', 'online_update','num','init']:
        if attr in default_values[args.dataset]:
            setattr(args, attr, default_values[args.dataset][attr])    

    if args.name is None:
        args.name = f'{args.dataset}/{args.neuron}/{args.algo}'
    
    neuron_hyper_list_all = ['decay', 'thresh', 'thresh0', 'beta', 'rho', 'branch']
    for attr in neuron_hyper_list_all:
        if attr not in neuron_hyper_list[args.neuron]:
            delattr(args, attr)
        elif getattr(args, attr) is None:
            setattr(args, attr, default_values[args.dataset]['neuron_hyper'][args.neuron][attr])

    return args

def main():

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # 初始化进程组
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank} is running on GPU {local_rank}")

    args = parse_config()
    if local_rank == 0:
        print(args)
        print(rank)

    save_dir = os.path.join('./exp', args.name, f'exp_{time.strftime("%Y_%m_%d_%H_%M_%S")}')
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=False)
        save_config(args, save_dir)
    # if rank == 0:
    #  print("Save to", save_dir)

    # process configuration and make directory
    # if args.resume == '':
    #     save_dir = os.path.join('./exp', args.name, f'exp_{time.strftime("%Y_%m_%d_%H_%M_%S")}')
    #     os.makedirs(save_dir, exist_ok=False)
    #     save_config(args, save_dir)
    #     print("Save to", save_dir)
    # else:
    #     save_dir = args.resume
    #     args = reload_config(args, save_dir)
    #     print(f"Resume from {save_dir}")

    # device = torch.device('cuda')
    device = local_rank
    # print(device)

    setup_seed(args.seed)
    model = SNN_Model(
        batch_size=args.batch_size,
        neuron_nums=args.cfg,
        neuron_type=args.neuron,
        recurrent=args.recurrent,
        temporal_detach=(False if args.algo == 'bptt' else True),
        readout=args.readout,
        readout_cumsum=args.cumsum,
        **{k: getattr(args, k) for k in neuron_hyper_list[args.neuron]},
    ).to(device)
    # print("okkkk")
    snn = DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
        find_unused_parameters= True,
        static_graph=False,  # default is False, should be true when use activation checkpointing in E2E
    )

    
    criterion = loss_dict[args.loss]()
    optimizer = optimizer_dict[args.optim](snn.parameters(), lr=args.lr)

    if args.scheduler == 'reduce-lr-on-plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',
            factor=0.7,
            patience=10,
            min_lr=1e-6,
        )
    elif args.scheduler == 'cosine-annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )
    elif args.scheduler == 'step-lr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=10,
            gamma=0.8,
        )
    else:
        scheduler = None

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) if args.step_lr else None

    # if args.resume != '':
    #     load_latest_checkpoint(save_dir, snn, optimizer, None, device)

    if args.algo == 'rests':
        snn = get_rests(snn.module, args.back).to(device)
    elif args.algo == 'restus':
        snn = get_restus(snn.module, args.back, num=args.num, init=args.init).to(device)
    elif args.algo in ['restu', 'uoro']:
        snn = globals()[f'get_{args.algo}'](snn.module,num=args.num,init=args.init).to(device)
    elif args.algo not in ['bptt', 'bp']:
        snn = globals()[f'get_{args.algo}'](snn.module).to(device)

    task = globals()[args.dataset.upper() + 'Task']()

    trainer = Trainer(
        model=snn,
        task=task,
        batch_size=args.batch_size,
        epochs=args.epochs,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        loss_config=LossConfig.EACH_STEP,
        accuracy_config=(AccuracyConfig.LAST_STEP_FINAL_OUTPUT
                        if args.cumsum else AccuracyConfig.LAST_STEP_AVERAGE_OUTPUT),
        ddp=True,
        rank=local_rank,
        world_size=world_size,
    )

    trainer.run(
        save_dir=save_dir,
        eval=True,
        device=device,
        allow_tf32=args.allow_tf32,
        online_update=args.online_update,
        packed_backprop=(args.algo == 'bptt'),
    )

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
