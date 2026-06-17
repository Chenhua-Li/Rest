"""Minimal BPTT vs REST-S comparison on one synthetic temporal task.

Run from the repository root:

    python test/bptt_vs_rests_demo.py

The task is intentionally small and self-contained. Each sample is a short
sequence with two input channels. The label is 1 if channel 0 has the larger
sum over time, otherwise 0.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import dataclass

import torch
from torch import nn


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import SNN_Model, get_rests  # noqa: E402


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor


def make_batch(time_window: int, batch_size: int, device: torch.device) -> Batch:
    """Create a tiny temporal classification batch.

    x shape: [time, batch, input_features]
    y shape: [batch]
    """

    x = torch.rand(time_window, batch_size, 2, device=device)
    y = (x[:, :, 0].sum(dim=0) > x[:, :, 1].sum(dim=0)).long()
    return Batch(x=x, y=y)


def build_snn(
    batch_size: int,
    temporal_detach: bool,
    device: torch.device,
) -> SNN_Model:
    return SNN_Model(
        batch_size=batch_size,
        neuron_nums=[2, 16, 2],
        neuron_type="lif",
        recurrent=True,
        temporal_detach=temporal_detach,
        readout="linear",
        readout_cumsum=False,
        decay=0.8,
        thresh=0.5,
    ).to(device)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_bptt(
    model: SNN_Model,
    steps: int,
    time_window: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> list[tuple[float, float]]:
    """Classic BPTT: keep the temporal graph, then update once per sequence."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for _ in range(steps):
        batch = make_batch(time_window, batch_size, device)
        optimizer.zero_grad()
        model.zero_grad()

        loss = 0.0
        last_output = None
        for t in range(time_window):
            last_output = model(batch.x[t], time_step=t)
            loss = loss + criterion(last_output, batch.y)

        loss = loss / time_window
        loss.backward()
        optimizer.step()

        assert last_output is not None
        history.append((loss.item(), accuracy_from_logits(last_output.detach(), batch.y)))

    return history


def train_rests_online(
    model: nn.Module,
    steps: int,
    time_window: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> list[tuple[float, float]]:
    """REST-S online update: detach temporal graph and update per time step."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for _ in range(steps):
        batch = make_batch(time_window, batch_size, device)
        model.zero_grad()
        optimizer.zero_grad()

        total_loss = 0.0
        last_output = None
        for t in range(time_window):
            last_output = model(batch.x[t], time_step=t)
            loss_t = criterion(last_output, batch.y)
            total_loss += loss_t.item()

            model.calc_grad(loss_t)
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        assert last_output is not None
        history.append(
            (
                total_loss / time_window,
                accuracy_from_logits(last_output.detach(), batch.y),
            )
        )

    return history


def print_summary(name: str, history: list[tuple[float, float]]) -> None:
    first_loss, first_acc = history[0]
    last_loss, last_acc = history[-1]
    print(
        f"{name:12s} | "
        f"loss {first_loss:.4f} -> {last_loss:.4f} | "
        f"acc {first_acc:.3f} -> {last_acc:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--time-window", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bptt_model = build_snn(
        batch_size=args.batch_size,
        temporal_detach=False,
        device=device,
    )

    rests_base = build_snn(
        batch_size=args.batch_size,
        temporal_detach=True,
        device=device,
    )
    rests_base.load_state_dict(copy.deepcopy(bptt_model.state_dict()))
    rests_model = get_rests(rests_base, back="r").to(device)

    bptt_history = train_bptt(
        bptt_model,
        steps=args.steps,
        time_window=args.time_window,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    rests_history = train_rests_online(
        rests_model,
        steps=args.steps,
        time_window=args.time_window,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    print(f"device: {device}")
    print("same task: label = argmax(sum over time of two input channels)")
    print_summary("BPTT", bptt_history)
    print_summary("REST-S", rests_history)


if __name__ == "__main__":
    main()
