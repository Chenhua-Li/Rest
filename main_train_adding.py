import os
import torch
from argparse import ArgumentParser
import time
from models import *
from utils import *
from tasks import *
from interfaces import *


def parse_config():
    argparser = ArgumentParser()

    argparser.add_argument('--name', type=str, default='adding')

    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--n-train', type=int, default=10000)
    argparser.add_argument('--n-test', type=int, default=2000)
    argparser.add_argument('--seq-len', type=int, default=500)
    argparser.add_argument('--num-classes', type=int, default=10)
    # argparser.add_argument('--dilation', type=int, default=1)

    argparser.add_argument('--cfg', type=int, nargs='+', default=[2, 64, 64, 64, 10])
    argparser.add_argument('--recurrent', action='store_true', default=False)
    argparser.add_argument('--decay', type=float, default=0.5)
    argparser.add_argument('--thresh', type=float, default=0.3)
    # argparser.add_argument('--lens', type=float, default=0.5)
    argparser.add_argument('--back', type=str, default='rrr')
    argparser.add_argument('--algo', type=str, default='rests', choices=['rests', 'restu', 'restus', 'bptt', 'bp', 'uoro', 'ppprop', 'ostl', 'ottt', 'eprop'])


    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--node', type=int, default=0)
    argparser.add_argument('--epochs', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=4e-5)
    argparser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'])
    argparser.add_argument('--step-lr', action='store_true', default=True)
    argparser.add_argument('--step-size', type=int, default=100)
    argparser.add_argument('--gamma', type=float, default=0.8)
    argparser.add_argument('--loss', type=str, default='ce', choices=['mse', 'ce'])
    argparser.add_argument('--allow-tf32', action='store_true', default=True)
    # argparser.add_argument('--grad-clip', type=float, default=0.0)
    # argparser.add_argument('--report-iter', type=int, default=10)
    # argparser.add_argument('--save-iter', action='store_true', default=False)

    argparser.add_argument('--resume', default='')

    argparser.add_argument('--num', type=int, default=15)
    argparser.add_argument('--init', type=str, default="randn")


    args = argparser.parse_args()
    return args

args = parse_config()
print(args)

# process configuration and make directory
if args.resume == '':
    save_dir = os.path.join('./exp', args.name, args.algo, f'exp_{time.strftime("%Y_%m_%d_%H_%M_%S")}')
    os.makedirs(save_dir, exist_ok=False)
    save_config(args, save_dir)
    print("Save to", save_dir)
else:
    save_dir = args.resume
    args = reload_config(args, save_dir)
    print(f"Resume from {save_dir}")

device = torch.device(f'cuda:{args.node}')

setup_seed(args.seed)
snn = SNN_Model(
    batch_size=args.batch_size,
    neuron_nums=args.cfg,
    neuron_type='lif',
    recurrent=args.recurrent,
    temporal_detach=(False if args.algo == 'bptt' else True),
    decay=args.decay,
    thresh=args.thresh,
)

criterion = loss_dict[args.loss]()
optimizer = optimizer_dict[args.optim](snn.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) if args.step_lr else None

if args.resume != '':
    load_latest_checkpoint(save_dir, snn, optimizer, scheduler, device)

if args.algo == 'rests':
    snn = get_rests(snn, args.back).to(device)
elif args.algo == 'restus':
    snn = get_restus(snn, args.back, num=args.num, init=args.init).to(device)
elif args.algo in ['restu', 'uoro']:
    snn = globals()[f'get_{args.algo}'](snn, num=args.num, init=args.init).to(device)
elif args.algo not in ['bptt', 'bp']:
    snn = globals()[f'get_{args.algo}'](snn).to(device)

task = AddingTask(seq_len=args.seq_len, num_classes=args.num_classes, N_train=args.n_train, N_test=args.n_test)
trainer = Trainer(model=snn,
                  task=task,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  optimizer=optimizer,
                  criterion=criterion,
                  scheduler=scheduler,
                  loss_config=LossConfig.EACH_STEP,
                  accuracy_config=AccuracyConfig.EACH_STEP)

trainer.run(save_dir=save_dir, eval=True, device=device, allow_tf32=args.allow_tf32, packed_backprop=(args.algo == 'bptt'))
