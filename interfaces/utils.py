import torch
import numpy as np


class LossConfig:
    """
    A class to define the loss configuration

    Attributes
    ----------
    EACH_STEP
        Calculate loss at each time step
    LAST_STEP
        Calculate loss at the last time step
    """
    EACH_STEP = 0
    LAST_STEP = 1


class AccuracyConfig:
    """
    A class to define the accuracy configuration

    Attributes
    ----------
    EACH_STEP
        Calculate accuracy at each time step
    LAST_STEP_AVERAGE_OUTPUT
        Calculate accuracy at the last time step by averaging the outputs
    LAST_STEP_FINAL_OUTPUT
        Calculate accuracy at the last time step using the final output
    """
    EACH_STEP = 0
    LAST_STEP_AVERAGE_OUTPUT = 1
    LAST_STEP_FINAL_OUTPUT = 2


class Metrics:
    def __init__(self):
        self.loss = None
        self.accuracy = None
        self.loss_iter = []
        self.accuracy_iter = []
    
    def update(self, loss, accuracy):
        if isinstance(loss, torch.Tensor):
            loss = loss.cpu().item()
        self.loss_iter.append(loss)
        self.accuracy_iter.append(accuracy)

    def finalize(self):
        self.loss = np.mean(self.loss_iter)
        self.accuracy = np.mean(self.accuracy_iter)
