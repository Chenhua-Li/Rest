import argparse
import torch
from models import *
from time import time
from utils import setup_seed
from tasks import AddingTask


arg = argparse.ArgumentParser()
arg.add_argument('--algo', type=str, choices=['rests', 'bptt'])
arg.add_argument('--back', type=str, choices=['rrr', 'rrb', 'rbr', 'brr', 'rbb', 'brb', 'bbr', 'bbb'])
arg.add_argument('--metric', type=str, choices=['memory', 'speed'])
arg.add_argument('--len', type=int, default=500)
arg.add_argument('--node', type=int, default=0)
args = arg.parse_args()

setup_seed(0)

device = torch.device(f'cuda:{args.node}')

batch_size = 256
cfg = [2, 64, 64, 64, 10]
model = SNN_Model(
    batch_size=batch_size,
    neuron_nums=cfg,
    neuron_type='lif',
    recurrent=True,
    bias=True,
    temporal_detach=True if args.algo == 'rests' else [x == 'b' for x in args.back],
).to(device)

if args.algo == 'rests':
    model = get_rests(model, back=args.back).to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-4)
criteria = torch.nn.CrossEntropyLoss()

# with torch.no_grad():
#     model.layers_bptt[0].linear.weight += 1

if args.metric == 'memory':
    warmup = 0
    testing = 1
elif args.metric == 'speed':
    warmup = 1
    testing = 1

def train(warmup, testing):
    task = AddingTask(seq_len=args.len)

    task.prepare_dataloader(batch_size)

    for idx, (data, target) in enumerate(task.train_loader):
        if idx == warmup:
            if args.metric == 'memory':
                torch.cuda.reset_peak_memory_stats(device)
            elif args.metric == 'speed':
                torch.cuda.synchronize(device)
                st = time()

        data, target = task.preprocess_data(data, target)
        data = data.to(device)
        target = target.to(device)

        model.zero_grad()
        optim.zero_grad()

        # if args.algo == 'bptt':
        #     loss = 0.

        for t in range(args.len):
            output = model(data[t], time_step=t)
            loss_t = criteria(output, target[t])
            if args.algo == 'rests':
                model.calc_grad(loss_t)
            else:
                loss_t.backward(retain_graph=True)
                # loss += loss_t

        # if args.algo == 'bptt':
        #     loss.backward()
        
        optim.step()

        # if mem is not None:
        #     mem.append(torch.cuda.memory_allocated(device) / 1024 / 1024)

        if idx == warmup + testing - 1:
            break
    
    if args.metric == 'memory':
        return torch.cuda.memory_stats(device)['allocated_bytes.all.peak'] / 1024 / 1024
    else:
        torch.cuda.synchronize(device)
        ed = time()
        return (ed - st) / testing


ret = train(warmup, testing)
print(args.algo, args.back, args.len, ret)
