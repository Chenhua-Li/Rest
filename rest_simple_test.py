import os
import sys
import torch
import time
from torch.utils.data import DataLoader
from distutils.util import strtobool # type: ignore

# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

from models import *
from utils import *
from tasks import *
from interfaces import *

algo = 'rests'
task_name = 'smnist'
device = torch.device('cuda')

task = globals()[task_name.upper() + 'Task']()
task.prepare_dataloader(batch_size=32, sampler_seed=0, ddp=False, world_size=1, rank=0)

# load the first batch
input, label = next(iter(task.train_loader))
input, label = task.preprocess_data(input, label)
input = input.to(device)
label = label.to(device)
# shape (timewindow, batch_size, input_size)
print(input.size())

snn = SNN_Model(
    batch_size=32,
    neuron_nums=[1, 64, 256, 256, 10],
    neuron_type='lif',
    recurrent=True,
    temporal_detach=(False if algo == 'bptt' else True),
    readout='linear',
    readout_cumsum=False,
    decay=0.9,
    thresh=0.8
).to(device)

# infer
for t in range(task.get_time_window()):
    output_t = snn(input[t], time_step=t)
    if not t:
        print(output_t)
        print(output_t.size())