import torch
import torch.nn as nn
from tasks import *
from interfaces.utils import *
from models.base import BackpropBase


class Verifier:
    def __init__(self, model: BackpropBase, batch_size: int, criterion: nn.Module, task: Task = None, loss_config: LossConfig=LossConfig.EACH_STEP):
        self.model = model
        self.task = task if task is not None else IdleTask(N=1000, in_features=model.neuron_nums[0], num_classes=model.neuron_nums[-1], time_window=100)
        self.batch_size = batch_size
        self.criterion = criterion
        self.loss_config = loss_config
        self.time_window = self.task.get_time_window()
        self.optimizer_online = torch.optim.Adam(model.model.parameters(), lr=1e-4)
        self.optimizer_bptt = torch.optim.Adam(model.bptt_model.parameters(), lr=1e-4)

    def run(self, device=torch.device('cuda')):
        self.model.to(device)
        self.model.train()
        
        flag_label_t = self.loss_config == LossConfig.EACH_STEP and self.task.has_label_each_step()

        self.task.prepare_dataloader(self.batch_size)

        for batch_idx, (input, label) in enumerate(self.task.train_loader):
            input, label = self.task.preprocess_data(input, label)
            input = input.to(device)
            label = label.to(device)

            self.optimizer_online.zero_grad()
            self.optimizer_bptt.zero_grad()
            self.model.zero_grad()

            for t in range(self.time_window):
                output_online, output_bptt = self.model(input[t], time_step=t, verify=True)
                # print(output_online, output_bptt)
                # print(self.model.model.layers[0].mem, self.model.bptt_model.layers[0].mem)
                # assert torch.allclose(self.model.model.layers[0].mem, self.model.bptt_model.layers[0].mem)
                # assert torch.allclose(self.model.model.layers[0].dinput, self.model.bptt_model.layers[0].dinput)

                label_t = label[t] if flag_label_t else label
                
                if self.loss_config == LossConfig.EACH_STEP:
                    loss_online_t = self.criterion(output_online, label_t)
                    loss_bptt_t = self.criterion(output_bptt, label_t)

                    self.model.calc_grad(loss_online_t)
                    loss_bptt_t.backward(retain_graph=True)

                    self.model.verify_grad()

                    print(f'timestep {t} pass')

            if self.loss_config == LossConfig.LAST_STEP:
                loss_online = self.criterion(output_online, label)
                loss_bptt = self.criterion(output_bptt, label)

                self.model.calc_grad(loss_online)
                loss_bptt.backward()

                self.model.verify_grad()

            print('batch pass')

            self.optimizer_online.step()
            self.optimizer_bptt.step()


# def get_verifier(model: BackpropBase, batch_size):
#     return Verifier(model, batch_size, nn.CrossEntropyLoss(), None, LossConfig.EACH_STEP)
