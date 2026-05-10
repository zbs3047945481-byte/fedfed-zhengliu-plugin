import torch


def build_main_optimizer(params, options):
    optimizer_name = str(options.get('optimizer_name', 'sgd')).lower()
    lr = float(options.get('lr', 0.01))
    weight_decay = float(options.get('weight_decay', 1e-4))
    if optimizer_name == 'sgd':
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(options.get('momentum', 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(options.get('nesterov', False)),
        )
    if optimizer_name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError('Unsupported optimizer_name: {}'.format(optimizer_name))


def adjust_main_learning_rate(optimizer, round_i, options):
    schedule = str(options.get('lr_schedule', 'none')).lower()
    base_lr = float(options.get('lr', 0.01))
    if schedule == 'inverse_round':
        lr = base_lr / (round_i + 1)
    elif schedule == 'none':
        lr = base_lr
    else:
        raise ValueError('Unsupported lr_schedule: {}'.format(schedule))
    for group in optimizer.param_groups:
        group['lr'] = lr
