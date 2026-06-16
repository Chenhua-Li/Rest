# Environment Setup

## Python version
Python 3.11 or higher.

## Install general dependencies

```bash
pip install -r requirements.txt
```

## Install the latest SpikingJelly from source

```bash
git clone https://github.com/fangwei123456/spikingjelly.git

cd spikingjelly

pip install .
```

# Code Instructions

To run the complete training by different algorithms, please use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m main_train --dataset DATASET --algo ALGO --back BACK
```

## Parameter Descriptions

- **dataset**: The name of the dataset. Available options include:
  - `scifar10` (rSeq CIFAR-10)
  - `nmnist` (N-MNIST)
  - `shd` (SHD)
  - `ssc` (SSC)
- **algo**: The name of the algorithm. Available options include:
  - `rests` (REST-S)
  - `restu` (REST-U)
  - `ostl` (OSTL)
  - `ottt` (OTTT)
  - `ppprop` (pp-prop)
  - `eprop` (e-prop)
  - `uoro` (UORO)
  - `bptt` (BPTT)
  - `bp` (BP)
- **back**: Used only when the algorithm is REST-S, to specify the strategy (e.g., `rrb`).

## Example

Run the REST-S algorithm with a specified strategy on rSeq CIFAR-10 dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python -m main_train --dataset scifar10 --algo rests --back rrb
```
