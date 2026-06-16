import torch
from torch import nn
from neurons import *
from utils import expand_conf
from models.readout import *


neurons_dict = {
    'lif': LIF,
    'alif': ALIF,
    'dhlif': DHLIF,
}

classifiers_dict = {
    'linear': LinearReadout,
    'spike': SpikeReadout,
    'potential': PotentialReadout,
    'potential_softmax': PotentialSoftmaxReadout,
    # 'SummedSpike': SummedSpikeClassifier,
    # 'SummedPotential': SummedPotentialClassifier,
}


class SNN_Model(nn.Module):
    def __init__(
            self,
            batch_size,
            neuron_nums,
            neuron_type,
            recurrent=False,
            bias=True,
            temporal_detach=True,
            readout='linear',
            readout_cumsum=False,
            **kwargs,
            ):
        
        super().__init__()

        self.batch_size = batch_size
        self.neuron_nums = neuron_nums
        self.neuron_type = neuron_type.lower()
        self.readout = readout
        self.readout_cumsum = readout_cumsum

        self.kwargs = kwargs

        if readout == 'linear':
            self.num_layers = len(neuron_nums) - 2
        else:
            self.num_layers = len(neuron_nums) - 1

        self.recurrent = expand_conf(recurrent, self.num_layers)
        self.bias = expand_conf(bias, self.num_layers)
        self.temporal_detach = expand_conf(temporal_detach, self.num_layers)
        
        self.layers = nn.ModuleList([
            neurons_dict[self.neuron_type](
                batch_size=self.batch_size,
                in_features=self.neuron_nums[i],
                out_features=self.neuron_nums[i + 1],
                recurrent=self.recurrent[i],
                bias=self.bias[i],
                temporal_detach=self.temporal_detach[i],
                # device=self.device,
                **self.kwargs
            ) for i in range(self.num_layers)
        ])
        if readout == 'linear':
            self.classifier = LinearReadout(
                in_features=self.neuron_nums[-2],
                out_features=self.neuron_nums[-1],
                bias=True,
                cumsum=readout_cumsum,
            )
        elif readout in ['spike', 'potential', 'potential_softmax']:
            self.classifier = classifiers_dict[readout](cumsum=readout_cumsum)
        # elif classifier in ['SummedSpike', 'SummedPotential']:
        #     self.classifier = classifiers_dict[classifier](detach_history=True)
        else:
            raise ValueError(f"Classifier {readout} not supported. Supported classifiers are: {list(classifiers_dict.keys())}")
        # print(f'classifier: {self.classifier.name}')
        # self.classifier = nn.Linear(self.neuron_nums[-2], self.neuron_nums[-1], bias=True)

    def forward(self, x, time_step):
        for i, layer in enumerate(self.layers):
            x = layer(x, init=(time_step == 0))
        self.output = self.classifier(x, init=(time_step == 0))
        return self.output


def create_bptt_model(model: SNN_Model, temporal_detach_bptt=False, copy_weights=True):
    bptt_model = SNN_Model(
        batch_size=model.batch_size,
        neuron_nums=model.neuron_nums,
        neuron_type=model.neuron_type,
        recurrent=model.recurrent,
        bias=model.bias,
        temporal_detach=temporal_detach_bptt,
        readout=model.readout,
        readout_cumsum=model.readout_cumsum,
        **model.kwargs,
    )
    if copy_weights:
        bptt_model.layers.load_state_dict(model.layers.state_dict())
        bptt_model.classifier.load_state_dict(model.classifier.state_dict())
    
    return bptt_model
