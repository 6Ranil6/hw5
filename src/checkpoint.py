"""Atomic local recovery; trusted checkpoints are never published as adapters."""
import os
import random
from pathlib import Path
import numpy as np
import torch


def save_checkpoint(path, model, optimizer, scheduler, state, device):
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if device.type == "mps":
        rng["mps"] = torch.mps.get_rng_state()
    if device.type == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state_all()
    payload = dict(state=state, rng=rng,
                   parameters={n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
                   optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_checkpoint(path, model, optimizer, scheduler, expected, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)  # only our own local checkpoint
    state = payload["state"]
    for key, value in expected.items():
        if state[key] != value:
            raise ValueError(f"Checkpoint mismatch: {key}; remove {path} to start a new run")
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != payload["parameters"].keys():
        raise ValueError("Checkpoint parameter names differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(payload["parameters"][name].to(parameter.device))
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if device.type == "mps":
        torch.mps.set_rng_state(rng["mps"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state
