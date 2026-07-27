import copy

import torch


class ModelEMA:
    """Exponential moving average of a model's weights, kept as a separate eval-only copy."""

    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.module.to(device)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.module.state_dict()
        for k, v in model.state_dict().items():
            ema_v = ema_state[k]
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(v.detach().to(ema_v.dtype), alpha=1.0 - self.decay)
            else:
                ema_v.copy_(v)

    def state_dict(self):
        return self.module.state_dict()
