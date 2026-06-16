import os
import sys
import torch
import torch.nn as nn
from time import time
from tasks import *
from interfaces.utils import *
from models.base import BackpropBase
import torch.distributed as dist  
from collections import defaultdict

False_ = object()

def save_as_npz(model, save_path):

    np_dict = {
        **{f'grad_{name}': param.grad.cpu().numpy() if param.grad is not None else np.array([])
           for name, param in model.named_parameters()}
    }
    

    np.savez_compressed(save_path, **np_dict)

class GradientHistory:
    def __init__(self, model):
        self.model = model
        self.history = defaultdict(list)
        
    def add_epoch(self, epoch):
        for name, param in self.model.named_parameters():
            grad = param.grad.cpu().numpy() if param.grad is not None else np.array([])
            self.history[f'epoch{epoch}_{name}'].append(grad)
    
    def save_to_npz(self, save_path):
        # 将列表转换为numpy数组
        np_dict = {k: np.stack(v) if v[0].size > 0 else np.array([]) 
                  for k, v in self.history.items()}
        np.savez_compressed(save_path, **np_dict)
    def load_from_npz(self, load_path):
        loaded = np.load(load_path)
        self.history = defaultdict(list)
        for k in loaded.files:
            self.history[k].append(loaded[k])


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 task: Task,
                 batch_size: int,
                 epochs: int,
                 optimizer: nn.Module,
                 criterion: nn.Module,
                 scheduler: nn.Module = None,
                #  iters: int = None,
                 grad_clip: float = None,
                 loss_config = LossConfig.EACH_STEP,
                 accuracy_config = AccuracyConfig.EACH_STEP,
                 ddp=False,
                 rank=0,
                 world_size=1,
                 update_step=1,
                 ):
        
        self.ddp=ddp
        self.rank=rank
        self.world_size=world_size
        self.model = model
        self.task = task
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler if scheduler is not None else torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1.0)

        self.grad_clip = grad_clip if grad_clip is not None else 0.
        self.loss_config = loss_config
        self.accuracy_config = accuracy_config
        
        self.time_window = task.get_time_window()
        self.is_online = isinstance(self.model, BackpropBase)
        self.update_step = update_step
        self.grad = GradientHistory(self.model)

    def _train(self,
               epoch,
               save_dir,
               save_checkpoint_epoch=False,
               report_every_n_iters=10,
               device=torch.device('cuda'),
               online_update=False,
               packed_backprop=False,
            ):
        # self.task.train_sampler.set_epoch(epoch)
        self.model.train()
        summary = Metrics()

        start_time = time()

        flag_label_t = (self.loss_config == LossConfig.EACH_STEP or self.accuracy_config == AccuracyConfig.EACH_STEP) and self.task.has_label_each_step()

        # [BEGIN] dataset iteration
        if self.ddp:
            self.task.train_sampler.set_epoch(epoch)
        
        for batch_idx, (input, label) in enumerate(self.task.train_loader):
            input, label = self.task.preprocess_data(input, label)
            input = input.to(device)
            label = label.to(device)

            self.model.zero_grad()
            self.optimizer.zero_grad()

            outputs = 0.
            loss = 0.
            accuracy = 0.
            
            if packed_backprop:
                loss_packed = 0.

            # [BEGIN] time window iteration
            for t in range(self.time_window):
                output_t = self.model(input[t], time_step=t)
            
                label_t = label[t] if flag_label_t else label

                if self.loss_config == LossConfig.EACH_STEP:
                    loss_t = self.criterion(output_t, label_t)
                    loss += loss_t.cpu().item()
                    if self.is_online:
                        self.model.calc_grad(loss_t)
                    elif packed_backprop:
                        loss_packed += loss_t
                    else:
                        loss_t.backward(retain_graph=True)

                    if online_update:
                        if self.ddp:
                            dist.barrier()
                            for param in self.model.parameters():
                                if param.grad is not None:
                                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.model.zero_grad()
                    
                if self.accuracy_config == AccuracyConfig.EACH_STEP:
                    accuracy_t = (torch.argmax(output_t, dim=1) == label_t).float().mean().cpu().item()
                    accuracy += accuracy_t
                elif self.accuracy_config == AccuracyConfig.LAST_STEP_AVERAGE_OUTPUT:
                    outputs += output_t.detach()
            # [END] time window iteration
            outputs /= self.time_window
            loss /= self.time_window
            accuracy /= self.time_window

            
            match self.loss_config:
                case LossConfig.EACH_STEP:
                    if packed_backprop:
                        loss_packed.backward()
                case LossConfig.LAST_STEP:
                    loss = self.criterion(output_t, label)
                    if self.is_online:
                        self.model.calc_grad(loss)
                    else:
                        loss.backward()

            match self.accuracy_config:
                case AccuracyConfig.LAST_STEP_AVERAGE_OUTPUT:
                    accuracy = (torch.argmax(outputs, dim=1) == label).float().mean().cpu().item()
                case AccuracyConfig.LAST_STEP_FINAL_OUTPUT:
                    accuracy = (torch.argmax(output_t, dim=1) == label).float().mean().cpu().item()
            if self.ddp:
                dist.barrier()
                loss = torch.tensor(loss, device=device)
                accuracy = torch.tensor(accuracy, device=device)
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                loss=loss.cpu().item()/self.world_size
                accuracy=accuracy.cpu().item()/self.world_size
            summary.update(loss, accuracy)

            if self.grad_clip > 0.:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            if not online_update:
                if self.ddp:
                    dist.barrier()
                    for param in self.model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                self.optimizer.step()
            if self.rank == 0:
                if (batch_idx + 1) % report_every_n_iters == 0:
                    print(f'    Epoch [{epoch + 1} / {self.epochs}], Iter [{batch_idx + 1} / {len(self.task.train_loader)}], Loss: {loss:.6f}, Acc: {accuracy:.4f}', flush=True)
            
                
        # [END] dataset iteration
        # dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        summary.finalize()

        if self.rank == 0:
            print(f'Train summary: Epoch [{epoch + 1} / {self.epochs}], Loss: {summary.loss:.6f}, Acc: {summary.accuracy:.4f}, Time elapsed: {time() - start_time:.2f}s', flush=True)
            
            if save_checkpoint_epoch:
                torch.save(self.model.state_dict(), os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            
            torch.save(self.model.state_dict(), os.path.join(save_dir, 'checkpoint_last.pth'))
            torch.save(self.optimizer.state_dict(), os.path.join(save_dir, 'optimizer_last.pth'))
            torch.save(self.scheduler.state_dict(), os.path.join(save_dir, 'scheduler_last.pth'))

        return summary
    
    @torch.no_grad()
    def test(self, epoch, device=torch.device('cuda')):
        self.model.eval()
        summary = Metrics()

        flag_label_t = (self.loss_config == LossConfig.EACH_STEP or self.accuracy_config == AccuracyConfig.EACH_STEP) and self.task.has_label_each_step()

        # [BEGIN] dataset iteration
        for batch_idx, (input, label) in enumerate(self.task.test_loader):
            input, label = self.task.preprocess_data(input, label)
            input = input.to(device)
            label = label.to(device)

            outputs = 0.
            loss = 0.
            accuracy = 0.

            # [BEGIN] time window iteration
            for t in range(self.time_window):
                output_t = self.model(input[t], time_step=t)
                
                label_t = label[t] if flag_label_t else label

                if self.loss_config == LossConfig.EACH_STEP:
                    loss_t = self.criterion(output_t, label_t)
                    loss += loss_t.cpu().item()
                
                if self.accuracy_config == AccuracyConfig.EACH_STEP:
                    accuracy_t = (torch.argmax(output_t, dim=1) == label_t).float().mean().cpu().item()
                    accuracy += accuracy_t
                elif self.accuracy_config == AccuracyConfig.LAST_STEP_AVERAGE_OUTPUT:
                    outputs += output_t.detach()
            # [END] time window iteration
            
            outputs /= self.time_window
            loss /= self.time_window
            accuracy /= self.time_window
            
            match self.loss_config:
                case LossConfig.EACH_STEP:
                    pass
                case LossConfig.LAST_STEP:
                    loss = self.criterion(output_t, label)

            match self.accuracy_config:
                case AccuracyConfig.LAST_STEP_AVERAGE_OUTPUT:
                    accuracy = (torch.argmax(outputs, dim=1) == label).float().mean().cpu().item()
                case AccuracyConfig.LAST_STEP_FINAL_OUTPUT:
                    accuracy = (torch.argmax(output_t, dim=1) == label).float().mean().cpu().item()


            if self.ddp:
                dist.barrier()
                loss = torch.tensor(loss, device=device)
                accuracy = torch.tensor(accuracy, device=device)

                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                
                # if self.rank == 0:
                loss=loss.cpu().item()/self.world_size
                accuracy=accuracy.cpu().item()/self.world_size
            summary.update(loss, accuracy)

        # [END] dataset iteration

        summary.finalize()
        if self.rank == 0:
            print(f'Test summary: Epoch [{epoch + 1} / {self.epochs}], Loss: {summary.loss:.6f}, Acc: {summary.accuracy:.4f}', flush=True)

        return summary
        
    
    def run(self,
            save_dir: str,
            eval=True,
            save_checkpoint_epoch=False,
            report_every_n_iters=10,
            device=torch.device('cuda'),
            allow_tf32=False,
            sampler_seed=0,
            online_update=False,
            packed_backprop=False,
            # world_size=1,
            # rank=0,
            ):
        # self.rank=rank
        start_epoch = self.scheduler.last_epoch if self.scheduler is not None else 0

        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            sys.stdout = open(os.path.join(save_dir, 'log.out'), 'w' if start_epoch == 0 else 'a')
            sys.stderr = open(os.path.join(save_dir, 'log.err'), 'w' if start_epoch == 0 else 'a')
        
        self.model.to(device)
        
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        if self.loss_config != LossConfig.EACH_STEP and packed_backprop is not False_:
            raise Warning('`packed_backprop` is ignored when loss_config is not `EACH_STEP`')

        if self.loss_config == LossConfig.LAST_STEP:
            if online_update:
                raise ValueError('`online_update` is not supported when loss_config is `LAST_STEP`')
            if packed_backprop:
                raise ValueError('`packed_backprop` is not supported when loss_config is `LAST_STEP`')
        
        if self.loss_config == LossConfig.EACH_STEP and online_update and packed_backprop:
            raise ValueError('`online_update` and `packed_backprop` are conflicted when loss_config is `EACH_STEP`')
        
        # if packed_backprop is False_:
        #     packed_backprop = False

        loss_saves = {'train_iter': [], 'train': []}
        accuracy_saves = {'train_iter': [], 'train': []}

        if eval:
            loss_saves['test'] = []
            accuracy_saves['test'] = []

        if self.rank == 0:
            if os.path.exists(os.path.join(save_dir, 'loss_saves.npz')):
                data = np.load(os.path.join(save_dir, 'loss_saves.npz'))
                for key in data.keys():
                    loss_saves[key] = data[key].tolist()

            if os.path.exists(os.path.join(save_dir, 'accuracy_saves.npz')):
                data = np.load(os.path.join(save_dir, 'accuracy_saves.npz'))
                for key in data.keys():
                    accuracy_saves[key] = data[key].tolist()

             # compatibility with previous version
            if os.path.exists(os.path.join(save_dir, 'acc_saves.npz')):
                data = np.load(os.path.join(save_dir, 'acc_saves.npz'))
                for key in data.keys():
                    accuracy_saves[key] = data[key].tolist()
        
        # if self.ddp:
        self.task.prepare_dataloader(self.batch_size, sampler_seed,ddp=self.ddp,world_size=self.world_size,rank=self.rank)
        if self.rank == 0:
            print('Start training from epoch', start_epoch + 1, flush=True)
        for epoch in range(start_epoch, self.epochs):
            train_summary = self._train(epoch=epoch,
                save_dir=save_dir,
                save_checkpoint_epoch=save_checkpoint_epoch,
                # save_checkpoint_iter=save_checkpoint_iter,
                report_every_n_iters=report_every_n_iters,
                # save_every_n_iters=save_every_n_iters,
                device=device,
                online_update=online_update,
                packed_backprop=packed_backprop,
                )
            
            loss_saves['train_iter'].append(train_summary.loss_iter)
            loss_saves['train'].append(train_summary.loss)
            accuracy_saves['train_iter'].append(train_summary.accuracy_iter)
            accuracy_saves['train'].append(train_summary.accuracy)
            
            if eval:
                eval_summary = self.test(device=device, epoch=epoch)
                loss_saves['test'].append(eval_summary.loss)
                accuracy_saves['test'].append(eval_summary.accuracy)
            
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(eval_summary.accuracy)
            else:
                self.scheduler.step()

                if self.rank == 0:
                    np.savez(os.path.join(save_dir, 'loss_saves.npz'), **loss_saves)
                    np.savez(os.path.join(save_dir, 'accuracy_saves.npz'), **accuracy_saves)
            
            if self.ddp:
                dist.barrier()

        # if self.rank == 0:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        torch.backends.cuda.matmul.allow_tf32 = False

        return True

    def rec_grad(self,
               epoch,
               save_dir,
               device=torch.device('cuda'),
               online_update=False,
               packed_backprop=False,
            ):
        report_every_n_iters=1

        self.model.train()
        summary = Metrics()

        start_time = time()

        flag_label_t = (self.loss_config == LossConfig.EACH_STEP or self.accuracy_config == AccuracyConfig.EACH_STEP) and self.task.has_label_each_step()

        if self.ddp:
            self.task.train_sampler.set_epoch(epoch)
        for batch_idx, (input, label) in enumerate(self.task.train_loader):
            
            input, label = self.task.preprocess_data(input, label)
            input = input.to(device)
            label = label.to(device)


            self.optimizer.zero_grad()
            self.model.zero_grad()

            outputs = 0.
            loss = 0.
            accuracy = 0.

            # if self.is_online is False :
            loss_packed = 0.

            for t in range(self.time_window):

                output_t = self.model(input[t], time_step=t)
                label_t = label[t] if flag_label_t else label
                if self.loss_config == LossConfig.EACH_STEP :
                    loss_t = self.criterion(output_t.squeeze(1), label_t)
                    loss += loss_t.cpu().item()
                    if self.ddp:
                        if self.is_online:
                
                            self.model.calc_grad(loss_t)
                        else:
                            # loss_packed += loss_t
                            loss_t.backward(retain_graph=True)
                        if online_update and (t+1)%self.update_step==0:
                            if self.is_online is not True :
                                loss_packed=0

                            dist.barrier()

                            for param in self.model.parameters():
                                if param.grad is not None:
                                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

                        self.grad.add_epoch(epoch)
                        self.optimizer.zero_grad()
                        self.model.zero_grad()
                    else:
                        if self.is_online:
                            self.model.calc_grad(loss_t)
                        elif packed_backprop:
                            loss_packed += loss_t
                        else:
                            loss_t.backward()

                        if online_update:
                            self.grad.add_epoch(epoch)
                            self.optimizer.zero_grad()
                            self.model.zero_grad()
                    
                if self.accuracy_config == AccuracyConfig.EACH_STEP:
                    accuracy_t = (torch.argmax(output_t, dim=1) == label_t).float().mean().cpu().item()
                    accuracy += accuracy_t
                elif self.accuracy_config == AccuracyConfig.LAST_STEP_AVERAGE_OUTPUT:
                    outputs += output_t.detach()
            # [END] time window iteration
            
            outputs /= self.time_window
            loss /= self.time_window
            accuracy /= self.time_window

            
            match self.loss_config:
                case LossConfig.EACH_STEP:
                    # if packed_backprop:
                    if self.is_online is False and online_update is False:
                        pass
                        # loss_packed.backward()
                case LossConfig.LAST_STEP:
                    loss = self.criterion(output_t, label)
                    if self.ddp:
                        dist.barrier()
                        with self.model.no_sync():
                            if self.is_online:
                                self.model.module.calc_grad(loss_t)
                            else:
                                loss_t.backward(retain_graph=True)
                    else:
                        if self.is_online:
                            self.model.calc_grad(loss_t)
                        else:
                            loss_t.backward(retain_graph=True)

            self.grad.save_to_npz(save_dir)


# def get_trainer(
#         task: str,
#         model: nn.Module,
#         batch_size,
#         epochs,
#         optimizer,
#         lr,
#         # iters=None,
#         loss='ce',
#         scheduler=None,
#         grad_clip=None):
#     """
#     The `get_trainer` function returns a `Trainer` object for a given task. The function is only for convenience and is NOT recommended for the main training code.

#     Parameters
#     ----------
#     task : str
#         The task to train the model on, one of 'smnist', 'nmnist', 'adding', 'shd'
#     model : `SNN_Model` or `BackpropBase`
#         The model to train
#     optimizer : str
#         The optimizer to use, one of 'adamw', 'adam', 'sgd'
#     loss : str, one of 'mse', 'ce'
#         The loss function to use
#     """
#     loss_dict = {
#         'mse': nn.MSELoss,
#         'ce': nn.CrossEntropyLoss,
#     }
#     optimizer_dict = {
#         'adamw': torch.optim.AdamW,
#         'adam': torch.optim.Adam,
#         'sgd': torch.optim.SGD,
#     }
#     criterion = loss_dict[loss]()
#     optimizer = optimizer_dict[optimizer](model.parameters(), lr=lr)

#     task_dict = {
#         'smnist': {
#             'class': SMNISTTask,
#             'args': {
#                 'root': './data/MNIST',
#                 'time_window': 112,
#             },
#             'loss_config': LossConfig.EACH_STEP,
#             'accuracy_config': AccuracyConfig.LAST_STEP_AVERAGE_OUTPUT,
#         },
#         'nmnist': {
#             'class': NMNISTTask,
#             'args': {
#                 'root': './data/NMNIST',
#             },
#             'loss_config': LossConfig.EACH_STEP,
#             'accuracy_config': AccuracyConfig.LAST_STEP_AVERAGE_OUTPUT,
#         },
#         'adding': {
#             'class': AddingTask,
#             'args': {
#                 'root': './data/adding_task',
#                 'seq_len': 500,
#                 'num_classes': 10,
#                 'N_train': 10000,
#                 'N_test': 2000,
#             },
#             'loss_config': LossConfig.EACH_STEP,
#             'accuracy_config': AccuracyConfig.EACH_STEP,
#         },
#         'shd': {
#             'class': SHDTask,
#             'args': {
#                 'root': './data/SHD',
#             },
#             'loss_config': LossConfig.EACH_STEP,
#             'accuracy_config': AccuracyConfig.EACH_STEP,
#         }
#     }
#     loss_config = task_dict[task]['loss_config']
#     accuracy_config = task_dict[task]['accuracy_config']
#     task = task_dict[task]['class'](**task_dict[task]['args'])

#     return Trainer(
#         model=model,
#         task=task,
#         batch_size=batch_size,
#         epochs=epochs,
#         optimizer=optimizer,
#         criterion=criterion,
#         scheduler=scheduler,
#         # iters,
#         grad_clip=grad_clip,
#         loss_config=loss_config,
#         accuracy_config=accuracy_config,
#         )
