import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from getdata import GetDataSet
from src.fed_server.fedavg import FedAvgTrainer
from src.models.models import choose_model
from src.options import input_options
from src.plugins.fedfed_modules import build_fedfed_generator
from src.utils.tools import (
    configure_runtime,
    get_each_client_data_index,
    get_runtime_device,
    resolve_heterogeneity_options,
    set_random_seed,
)


@torch.no_grad()
def transform_dataset(data, labels, generator, mode, device, batch_size, limit):
    if limit > 0:
        data = data[:limit]
        labels = labels[:limit]
    loader = DataLoader(
        TensorDataset(torch.tensor(data).float(), torch.tensor(labels).long()),
        batch_size=batch_size,
        shuffle=False,
    )
    xs_out, y_out = [], []
    generator.eval()
    for x, y in loader:
        x = x.to(device)
        if mode == 'x':
            z = x
        else:
            xr = generator(x)
            xs = x - xr
            z = xs if mode == 'xs' else xr
        xs_out.append(z.cpu())
        y_out.append(y)
    return TensorDataset(torch.cat(xs_out, dim=0), torch.cat(y_out, dim=0))


def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.numel()
            loss_sum += loss.item() * y.numel()
    return {'acc': correct / max(total, 1), 'loss': loss_sum / max(total, 1)}


def train_classifier(options, train_ds, test_ds, device, mode):
    model = choose_model(options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'])
    train_loader = DataLoader(
        train_ds,
        batch_size=options['batch_size'],
        shuffle=True,
        num_workers=max(int(options.get('dataloader_num_workers', 0)), 0),
        pin_memory=device.type == 'cuda' and bool(options.get('dataloader_pin_memory', True)),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=options['batch_size'],
        shuffle=False,
        num_workers=max(int(options.get('dataloader_num_workers', 0)), 0),
        pin_memory=device.type == 'cuda' and bool(options.get('dataloader_pin_memory', True)),
    )
    history = []
    started = time.time()
    for epoch in range(int(options.get('diagnostic_epochs', 10))):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, test_loader, device)
        history.append({'epoch': epoch + 1, **metrics})
        print('DIAG {} epoch {} acc {:.2f}% loss {:.4f}'.format(
            mode, epoch + 1, metrics['acc'] * 100, metrics['loss']
        ))
    best = max(history, key=lambda item: item['acc'])
    final = history[-1]
    return {
        'mode': mode,
        'best_acc': best['acc'],
        'best_epoch': best['epoch'],
        'final_acc': final['acc'],
        'final_loss': final['loss'],
        'duration_min': round((time.time() - started) / 60.0, 2),
        'history': history,
    }


def main():
    options = input_options()
    options = resolve_heterogeneity_options(options)
    options['plugin_name'] = 'fedfed_image'
    options['fedfed_two_stage'] = True
    configure_runtime(options)
    set_random_seed(options['seed'])
    device = get_runtime_device(options)

    dataset = GetDataSet(options['dataset_name'])
    client_indices = get_each_client_data_index(
        dataset.train_label,
        options['num_of_clients'],
        options,
    )
    trainer = FedAvgTrainer(options, dataset, client_indices)
    trainer._maybe_run_fedfed_two_stage()

    generator = build_fedfed_generator(
        options.get('fedfed_generator_type', 'paper_beta_vae'),
        int(options.get('fedfed_input_channels', 3)),
        latent_channels=int(options.get('fedfed_vae_latent_channels', 32)),
        z_dim=int(options.get('fedfed_vae_z_dim', 2048)),
    ).to(device)
    generator.load_state_dict({
        key: value.to(device)
        for key, value in trainer.server_plugin.generator_state.items()
    })

    batch_size = int(options['batch_size'])
    train_limit = int(options.get('diagnostic_train_limit', 20000))
    test_limit = int(options.get('diagnostic_test_limit', 10000))
    results = []
    for mode in ('x', 'xs', 'xr'):
        train_ds = transform_dataset(
            dataset.train_data, dataset.train_label, generator, mode, device, batch_size, train_limit
        )
        test_ds = transform_dataset(
            dataset.test_data, dataset.test_label, generator, mode, device, batch_size, test_limit
        )
        results.append(train_classifier(options, train_ds, test_ds, device, mode))

    out_dir = os.path.join('result', options['dataset_name'], 'fedfed_xs_diagnostic_' + options.get('experiment_tag', 'run'))
    os.makedirs(out_dir, exist_ok=True)
    out = {
        'options': options,
        'results': results,
    }
    with open(os.path.join(out_dir, 'diagnostic_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(out_dir, 'diagnostic_summary.csv'), 'w', encoding='utf-8') as f:
        f.write('mode,best_acc,final_acc,best_epoch,final_loss,duration_min\n')
        for row in results:
            f.write('{},{:.6f},{:.6f},{},{:.6f},{}\n'.format(
                row['mode'],
                row['best_acc'],
                row['final_acc'],
                row['best_epoch'],
                row['final_loss'],
                row['duration_min'],
            ))
    print('DIAG_OUT', out_dir)


if __name__ == '__main__':
    main()
