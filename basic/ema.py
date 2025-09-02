import torch


def init_ema(model_ema):
    for p in model_ema.parameters():
        p.requires_grad_(False)


def update_ema(model, model_ema, num_updates=-1, decay=0.9999):
    _cdecay = min(decay, (1 + num_updates) / (10 + num_updates))

    with torch.no_grad():
        msd = model.state_dict()
        for k, ema_v in model_ema.state_dict().items():
            model_v = msd[k].detach()
            # if self.device:
            #     model_v = model_v.to(device=self.device)
            ema_v.copy_(ema_v * _cdecay + (1. - _cdecay) * model_v)
    return