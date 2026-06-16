import os
import torch
import random
import numpy as np
from torch import nn
from configparser import ConfigParser


def expand_conf(x, num):
    if isinstance(x, bool):
        return [x] * num
    
    if isinstance(x, str):
        if len(x) == 1:
            return x * num
        assert len(x) == num
        return x
    
    if isinstance(x, list):
        if len(x) == 1:
            return x * num
        assert len(x) == num
        return x
    
    raise ValueError(f'Invalid type: {type(x)}')


def setup_seed(seed, deter=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = deter


def reload_config(args, dir):
    config = ConfigParser()
    config.read(os.path.join(dir, 'config.ini'))

    kw = args._get_kwargs()

    for key, _ in kw:
        if key not in ['resume', 'node'] and config['CUSTOM'].get(key) is not None:
            if isinstance(_, list):
                args.__setattr__(key, eval(config['CUSTOM'].get(key)))
            else:
                args.__setattr__(key, type(_)(config['CUSTOM'].get(key)))
    
    return args


def save_config(args, save_dir):
    if hasattr(args, 'resume') and args.resume != '':
        raise ValueError('Can not save config when resuming!')
    
    config = ConfigParser()
    config['CUSTOM'] = vars(args)

    if config.has_option('CUSTOM', 'resume'):
        config.remove_option('CUSTOM', 'resume')
    
    with open(os.path.join(save_dir, 'config.ini'), 'w') as f:
        config.write(f)


loss_dict = {
    'mse': nn.MSELoss,
    'ce': nn.CrossEntropyLoss,
}
optimizer_dict = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}

def load_latest_checkpoint(save_dir, model, optimizer, scheduler, device):
    weight_dict = torch.load(os.path.join(save_dir, 'checkpoint_last.pth'), map_location=device, weights_only=True)
    weight_dict_renamed = {}

    for key in weight_dict.keys():
        t = key.split('.')
        if t[0] in ['layers', 'classifier']:
            weight_dict_renamed[key] = weight_dict[key]
        elif t[1] in ['layers', 'classifier']:
            weight_dict_renamed['.'.join(t[1:])] = weight_dict[key]
        else:
            raise Warning(f'Parameter with key "{key}" not loaded.')

    model.load_state_dict(weight_dict_renamed)
    optimizer.load_state_dict(torch.load(os.path.join(save_dir, 'optimizer_last.pth'), map_location=device, weights_only=True))
    
    if scheduler is not None:
        scheduler.load_state_dict(torch.load(os.path.join(save_dir, 'scheduler_last.pth'), map_location=device, weights_only=True))
