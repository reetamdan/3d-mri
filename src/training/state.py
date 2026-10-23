"""Training state shared by VQ and diffusion, including exact resume essentials."""
from copy import deepcopy
import random
import numpy as np
import torch


def rng_state():
    numpy = np.random.get_state()
    return dict(torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                numpy_rng=dict(generator=numpy[0], keys=numpy[1].tolist(), position=numpy[2],
                               has_gaussian=numpy[3], cached_gaussian=numpy[4]),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(checkpoint, device):
    torch.set_rng_state(checkpoint['torch_rng'])
    if checkpoint.get('python_rng'):
        random.setstate(checkpoint['python_rng'])
    if checkpoint.get('numpy_rng'):
        n = checkpoint['numpy_rng']
        np.random.set_state((n['generator'], np.array(n['keys'], dtype=np.uint32), n['position'], n['has_gaussian'], n['cached_gaussian']))
    if device.type == 'cuda' and checkpoint.get('cuda_rng'):
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])


class ModelEMA:
    def __init__(self, model, decay=.995):
        if not 0 < decay < 1:
            raise ValueError('EMA decay must be in (0,1)')
        self.decay, self.updates = decay, 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1+self.updates)/(10+self.updates))
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value.detach(), 1-decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates, shadow=self.shadow)

    def load_state_dict(self, state, device):
        self.decay, self.updates = state['decay'], state['updates']
        self.shadow = {k: v.to(device) for k, v in state['shadow'].items()}

    def evaluation_model(self, model):
        result = deepcopy(model)
        result.load_state_dict(self.shadow)
        return result.eval()


def precision(config, device):
    enabled = bool(config.get('mixed_precision', False) and device.type == 'cuda')
    dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=enabled and dtype == torch.float16)
    return enabled, dtype, scaler


def scheduler_for(optimizer, config):
    kind = config.get('lr_schedule', 'none')
    if kind == 'none':
        return None
    if kind == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.get('lr_patience', 10), factor=.5)
    raise ValueError('lr_schedule must be none or plateau')
