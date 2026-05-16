import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image

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


def parse_variants(text):
    if not text:
        return [
            ("none", "none", 0.0, 0.0, False, None),
            ("g010_015", "gaussian", 0.10, 0.15, True, None),
            ("g015_020", "gaussian", 0.15, 0.20, True, None),
            ("g020_025", "gaussian", 0.20, 0.25, True, None),
        ]
    variants = []
    for item in text.split(","):
        parts = item.split(":")
        if len(parts) not in {4, 5, 6}:
            raise ValueError("Variant must be name:type:std1:std2[:ce_on_noisy][:clip_norm]")
        name, noise_type, std1, std2 = parts[:4]
        ce_on_noisy = len(parts) == 5 and parts[4].lower() in {"true", "1", "yes"}
        if len(parts) == 6:
            ce_on_noisy = parts[4].lower() in {"true", "1", "yes"}
        clip_norm = None if len(parts) < 6 or parts[5].lower() in {"", "none", "off", "no"} else float(parts[5])
        variants.append((name, noise_type, float(std1), float(std2), ce_on_noisy, clip_norm))
    return variants


def clip_sensitive_feature(xs, clip_norm):
    clip_norm = float(clip_norm or 0.0)
    if clip_norm <= 0.0:
        return xs
    flat = xs.flatten(1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(clip_norm / norms, max=1.0)
    return (flat * scale).view_as(xs)


def add_noise(x, noise_type, std, mean=0.0, noise_shape="paper"):
    if noise_type in {"none", "", "off"} or std <= 0:
        return x.clone()
    size = x.shape[-3:] if str(noise_shape).lower() == "paper" and x.dim() >= 3 else x.shape
    if noise_type == "gaussian":
        return x + (torch.randn(size, device=x.device, dtype=x.dtype) * std + mean)
    if noise_type == "laplace":
        dist = torch.distributions.Laplace(
            torch.full(size, mean, device=x.device, dtype=x.dtype),
            torch.full(size, std, device=x.device, dtype=x.dtype),
        )
        return x + dist.sample()
    raise ValueError("Unsupported noise type: {}".format(noise_type))


def parse_bool(value):
    return str(value).lower() in {"1", "true", "yes", "on"}


def use_dynamic_noise_train(options):
    return parse_bool(options.get("privacy_dynamic_noise_train", False))


class DynamicProtectedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, xs, y, noise_type, std, noise_shape="paper"):
        self.xs = xs
        self.y = y
        self.noise_type = noise_type
        self.std = float(std)
        self.noise_shape = noise_shape

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xp = add_noise(self.xs[idx], self.noise_type, self.std, noise_shape=self.noise_shape)
        return xp, self.y[idx]


def protected_dataset(tensors, options, dynamic):
    if dynamic and options.get("fedfed_noise_type", "none") not in {"none", "", "off"}:
        return DynamicProtectedTensorDataset(
            tensors["xs"],
            tensors["y"],
            options.get("fedfed_noise_type", "none"),
            float(options.get("fedfed_noise_std1", 0.0)),
            options.get("fedfed_noise_shape", "paper"),
        )
    return TensorDataset(tensors["xp"], tensors["y"])


def tensor_stats(values):
    values = values.detach().float().cpu().numpy()
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


@torch.no_grad()
def build_feature_tensors(data, labels, generator, device, batch_size, limit, noise_type, std1, clip_norm=0.0, noise_shape="paper"):
    if limit > 0:
        data = data[:limit]
        labels = labels[:limit]
    loader = DataLoader(
        TensorDataset(torch.tensor(data).float(), torch.tensor(labels).long()),
        batch_size=batch_size,
        shuffle=False,
    )
    xs, xp, xr, x_all, y_all = [], [], [], [], []
    x_norms, xs_norms, xp_norms, xr_norms, ratios = [], [], [], [], []
    generator.eval()
    for x, y in loader:
        x = x.to(device)
        recon = generator(x)
        sensitive_raw = x - recon
        sensitive = clip_sensitive_feature(sensitive_raw, clip_norm)
        protected = add_noise(sensitive, noise_type, std1, noise_shape=noise_shape)
        x_norm = x.flatten(1).norm(p=2, dim=1)
        xs_norm = sensitive.flatten(1).norm(p=2, dim=1)
        xp_norm = protected.flatten(1).norm(p=2, dim=1)
        xr_norm = recon.flatten(1).norm(p=2, dim=1)
        ratio = xs_norm / x_norm.clamp_min(1e-12)
        x_all.append(x.cpu())
        xs.append(sensitive.cpu())
        xp.append(protected.cpu())
        xr.append(recon.cpu())
        y_all.append(y)
        x_norms.append(x_norm.cpu())
        xs_norms.append(xs_norm.cpu())
        xp_norms.append(xp_norm.cpu())
        xr_norms.append(xr_norm.cpu())
        ratios.append(ratio.cpu())
    tensors = {
        "x": torch.cat(x_all),
        "xs": torch.cat(xs),
        "xp": torch.cat(xp),
        "xr": torch.cat(xr),
        "y": torch.cat(y_all),
    }
    stats = {
        "x_norm": tensor_stats(torch.cat(x_norms)),
        "xs_norm": tensor_stats(torch.cat(xs_norms)),
        "xp_norm": tensor_stats(torch.cat(xp_norms)),
        "xr_norm": tensor_stats(torch.cat(xr_norms)),
        "xs_x_ratio": tensor_stats(torch.cat(ratios)),
    }
    return tensors, stats


def evaluate_classifier(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            correct += logits.argmax(dim=1).eq(y).sum().item()
            total += y.numel()
            loss_sum += loss.item() * y.numel()
    return {"acc": correct / max(total, 1), "loss": loss_sum / max(total, 1)}


def train_classifier(options, train_ds, test_ds, device, tag):
    model = choose_model(options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(options.get("lr", 0.001)))
    train_loader = DataLoader(train_ds, batch_size=options["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=options["batch_size"], shuffle=False, num_workers=0)
    history = []
    for epoch in range(int(options.get("diagnostic_epochs", 6))):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        metrics = evaluate_classifier(model, test_loader, device)
        history.append({"epoch": epoch + 1, **metrics})
        print("PROBE_CLS {} epoch {} acc {:.2f}% loss {:.4f}".format(
            tag, epoch + 1, metrics["acc"] * 100, metrics["loss"]
        ))
    best = max(history, key=lambda item: item["acc"])
    return model, {
        "tag": tag,
        "best_acc": best["acc"],
        "best_epoch": best["epoch"],
        "final_acc": history[-1]["acc"],
        "final_loss": history[-1]["loss"],
        "history": history,
    }


def attack_features(model, x, y, device, batch_size):
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        model.eval()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            prob = F.softmax(logits, dim=1)
            top_prob, _ = torch.sort(prob, dim=1, descending=True)
            top3 = top_prob[:, :3]
            conf = top_prob[:, :1]
            margin = (top_prob[:, :1] - top_prob[:, 1:2])
            entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
            true_prob = prob.gather(1, yb.view(-1, 1))
            ce = F.cross_entropy(logits, yb, reduction="none").view(-1, 1)
            feats.append(torch.cat([top3, conf, margin, entropy, true_prob, ce], dim=1).cpu())
    return torch.cat(feats)


class AttackMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def binary_auc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(np.int64)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = ranks[pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def run_mia(target_model, train_tensors, test_tensors, options, device):
    member_count = min(len(train_tensors["xp"]) // 2, int(options.get("privacy_mia_samples", 1000)))
    nonmember_count = min(len(test_tensors["xp"]), member_count)
    member_x = train_tensors["xp"][:member_count]
    member_y = train_tensors["y"][:member_count]
    nonmember_x = test_tensors["xp"][:nonmember_count]
    nonmember_y = test_tensors["y"][:nonmember_count]
    x_feat = torch.cat([
        attack_features(target_model, member_x, member_y, device, options["batch_size"]),
        attack_features(target_model, nonmember_x, nonmember_y, device, options["batch_size"]),
    ])
    y_attack = torch.cat([torch.ones(member_count), torch.zeros(nonmember_count)])
    perm = torch.randperm(y_attack.numel())
    x_feat = x_feat[perm]
    y_attack = y_attack[perm]
    split = int(0.7 * y_attack.numel())
    x_train, y_train = x_feat[:split].to(device), y_attack[:split].to(device)
    x_test, y_test = x_feat[split:].to(device), y_attack[split:].to(device)
    attack = AttackMLP(x_feat.size(1)).to(device)
    opt = torch.optim.Adam(attack.parameters(), lr=0.001)
    for _ in range(int(options.get("privacy_mia_epochs", 30))):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(attack(x_train), y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = attack(x_test)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        labels = y_test.cpu().numpy()
        scores = prob.cpu().numpy()
        pred_np = pred.cpu().numpy()
    tp = float(((pred_np == 1) & (labels == 1)).sum())
    fp = float(((pred_np == 1) & (labels == 0)).sum())
    fn = float(((pred_np == 0) & (labels == 1)).sum())
    acc = float((pred_np == labels).mean())
    return {
        "attack_auc": binary_auc(scores, labels),
        "attack_acc": acc,
        "member_recall": tp / max(tp + fn, 1.0),
        "member_precision": tp / max(tp + fp, 1.0),
        "member_count": int(member_count),
        "nonmember_count": int(nonmember_count),
    }


def train_shadow_classifier(options, train_tensors, test_tensors, device, tag):
    epochs = int(options.get("privacy_shadow_epochs", 0) or options.get("diagnostic_epochs", 6))
    model = choose_model(options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(options.get("lr", 0.001)))
    dynamic = use_dynamic_noise_train(options)
    train_ds = ConcatDataset([
        TensorDataset(train_tensors["x"], train_tensors["y"]),
        protected_dataset(train_tensors, options, dynamic),
    ])
    train_loader = DataLoader(train_ds, batch_size=options["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(TensorDataset(test_tensors["x"], test_tensors["y"]), batch_size=options["batch_size"], shuffle=False, num_workers=0)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        metrics = evaluate_classifier(model, test_loader, device)
        print("PROBE_SHADOW {} epoch {} acc {:.2f}% loss {:.4f}".format(
            tag, epoch + 1, metrics["acc"] * 100, metrics["loss"]
        ))
    return model


def slice_tensors(tensors, start, count):
    return {key: value[start:start + count] for key, value in tensors.items()}


def train_attack_from_features(x_feat, y_attack, options, device):
    perm = torch.randperm(y_attack.numel())
    x_feat = x_feat[perm]
    y_attack = y_attack[perm]
    attack = AttackMLP(x_feat.size(1)).to(device)
    opt = torch.optim.Adam(attack.parameters(), lr=0.001)
    x_train = x_feat.to(device)
    y_train = y_attack.to(device)
    for _ in range(int(options.get("privacy_mia_epochs", 30))):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(attack(x_train), y_train)
        loss.backward()
        opt.step()
    return attack


def eval_attack(attack, x_feat, y_attack, device):
    with torch.no_grad():
        logits = attack(x_feat.to(device))
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        labels = y_attack.cpu().numpy()
        scores = prob.cpu().numpy()
        pred_np = pred.cpu().numpy()
    tp = float(((pred_np == 1) & (labels == 1)).sum())
    fp = float(((pred_np == 1) & (labels == 0)).sum())
    fn = float(((pred_np == 0) & (labels == 1)).sum())
    acc = float((pred_np == labels).mean())
    return {
        "attack_auc": binary_auc(scores, labels),
        "attack_acc": acc,
        "member_recall": tp / max(tp + fn, 1.0),
        "member_precision": tp / max(tp + fp, 1.0),
    }


def run_shadow_mia(target_model, train_tensors, test_tensors, options, device):
    shadow_models = max(int(options.get("privacy_shadow_models", 3)), 1)
    requested_eval = int(options.get("privacy_mia_samples", 1000))
    target_count = min(len(train_tensors["xp"]) // 2, len(test_tensors["xp"]) // 4, requested_eval)
    target_count = max(target_count, 1)
    aux_start = target_count
    aux_available = max(len(test_tensors["xp"]) - aux_start, 0)
    per_shadow_train = min(int(options.get("privacy_shadow_train_samples", 500)), aux_available // max(2 * shadow_models, 1))
    per_shadow_test = min(int(options.get("privacy_shadow_test_samples", 500)), aux_available // max(2 * shadow_models, 1))
    per_shadow = min(per_shadow_train, per_shadow_test)
    if per_shadow <= 0:
        fallback = run_mia(target_model, train_tensors, test_tensors, options, device)
        fallback["attack_mode"] = "direct_fallback_insufficient_shadow_data"
        return fallback

    attack_feature_rows = []
    attack_labels = []
    cursor = aux_start
    used_shadow_models = 0
    for shadow_idx in range(shadow_models):
        if cursor + 2 * per_shadow > len(test_tensors["xp"]):
            break
        shadow_train = slice_tensors(test_tensors, cursor, per_shadow)
        cursor += per_shadow
        shadow_test = slice_tensors(test_tensors, cursor, per_shadow)
        cursor += per_shadow
        shadow_model = train_shadow_classifier(options, shadow_train, shadow_test, device, "shadow{}".format(shadow_idx + 1))
        attack_feature_rows.append(attack_features(shadow_model, shadow_train["xp"], shadow_train["y"], device, options["batch_size"]))
        attack_labels.append(torch.ones(per_shadow))
        attack_feature_rows.append(attack_features(shadow_model, shadow_test["xp"], shadow_test["y"], device, options["batch_size"]))
        attack_labels.append(torch.zeros(per_shadow))
        used_shadow_models += 1

    if used_shadow_models == 0:
        fallback = run_mia(target_model, train_tensors, test_tensors, options, device)
        fallback["attack_mode"] = "direct_fallback_no_shadow_model"
        return fallback

    attack = train_attack_from_features(torch.cat(attack_feature_rows), torch.cat(attack_labels), options, device)
    target_member_x = train_tensors["xp"][:target_count]
    target_member_y = train_tensors["y"][:target_count]
    target_nonmember_x = test_tensors["xp"][:target_count]
    target_nonmember_y = test_tensors["y"][:target_count]
    target_feat = torch.cat([
        attack_features(target_model, target_member_x, target_member_y, device, options["batch_size"]),
        attack_features(target_model, target_nonmember_x, target_nonmember_y, device, options["batch_size"]),
    ])
    target_labels = torch.cat([torch.ones(target_count), torch.zeros(target_count)])
    metrics = eval_attack(attack, target_feat, target_labels, device)
    metrics.update({
        "attack_mode": "shadow",
        "shadow_models": int(used_shadow_models),
        "shadow_train_samples": int(per_shadow),
        "shadow_test_samples": int(per_shadow),
        "member_count": int(target_count),
        "nonmember_count": int(target_count),
    })
    return metrics


class InversionNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def psnr(mse):
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def run_inversion(train_tensors, test_tensors, options, device, out_path):
    channels = train_tensors["x"].shape[1]
    model = InversionNet(channels).to(device)
    train_count = min(len(train_tensors["xp"]), int(options.get("privacy_inversion_train_samples", 10000)))
    test_count = min(len(test_tensors["xp"]), int(options.get("privacy_inversion_test_samples", 512)))
    train_ds = TensorDataset(train_tensors["xp"][:train_count], train_tensors["x"][:train_count])
    train_loader = DataLoader(train_ds, batch_size=options["batch_size"], shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(int(options.get("privacy_inversion_epochs", 8))):
        model.train()
        loss_sum = total = 0
        for xp, x in train_loader:
            xp = xp.to(device)
            x = x.to(device)
            opt.zero_grad()
            recon = model(xp)
            loss = F.l1_loss(recon, x)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
        print("PROBE_INV epoch {} l1 {:.5f}".format(epoch + 1, loss_sum / max(total, 1)))
    with torch.no_grad():
        xp = test_tensors["xp"][:test_count].to(device)
        x = test_tensors["x"][:test_count].to(device)
        recon = model(xp).clamp(0.0, 1.0)
        mse = F.mse_loss(recon, x).item()
        l1 = F.l1_loss(recon, x).item()
        n = min(8, x.size(0))
        rows = []
        diff = (x[:n] - recon[:n]).abs()
        rows.extend([x[:n].cpu(), xp[:n].clamp(0, 1).cpu(), recon[:n].cpu(), diff.cpu()])
        grid = make_grid(torch.cat(rows, dim=0), nrow=n, padding=2)
        save_image(grid, out_path)
    return {"inversion_mse": mse, "inversion_l1": l1, "inversion_psnr": psnr(mse), "figure": out_path}


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    options = input_options()
    options = resolve_heterogeneity_options(options)
    options["plugin_name"] = "fedfed_image"
    options["fedfed_two_stage"] = True
    options.setdefault("diagnostic_epochs", 6)
    options["privacy_dynamic_noise_train"] = parse_bool(os.environ.get("FEDFED_DYNAMIC_NOISE_TRAIN", "false"))
    options["dataloader_num_workers"] = 0
    configure_runtime(options)
    base_tag = options.get("experiment_tag", "privacy_probe")
    variants = parse_variants(os.environ.get("FEDFED_PRIVACY_VARIANTS", ""))
    out_root = os.path.join("result", options["dataset_name"], "fedfed_privacy_probe_" + base_tag)
    fig_dir = os.path.join(out_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    device = get_runtime_device(options)
    dataset = GetDataSet(options["dataset_name"])
    summary_rows = []
    norm_rows = []
    started = time.time()

    for name, noise_type, std1, std2, ce_on_noisy, clip_norm in variants:
        print("PROBE_VARIANT", name, noise_type, std1, std2, ce_on_noisy)
        run_options = copy.deepcopy(options)
        run_options["experiment_tag"] = base_tag + "_" + name
        run_options["fedfed_noise_type"] = noise_type
        run_options["fedfed_noise_std1"] = std1
        run_options["fedfed_noise_std2"] = std2
        run_options["fedfed_distill_ce_on_noisy_xs"] = ce_on_noisy
        if clip_norm is not None:
            run_options["fedfed_clip_norm"] = clip_norm
        set_random_seed(int(run_options["seed"]))
        client_indices = get_each_client_data_index(dataset.train_label, run_options["num_of_clients"], run_options)
        trainer = FedAvgTrainer(run_options, dataset, client_indices)
        trainer._maybe_run_fedfed_two_stage()
        generator = build_fedfed_generator(
            run_options.get("fedfed_generator_type", "paper_beta_vae"),
            int(run_options.get("fedfed_input_channels", 3)),
            latent_channels=int(run_options.get("fedfed_vae_latent_channels", 32)),
            z_dim=int(run_options.get("fedfed_vae_z_dim", 2048)),
        ).to(device)
        generator.load_state_dict({k: v.to(device) for k, v in trainer.server_plugin.generator_state.items()})
        train_tensors, train_stats = build_feature_tensors(
            dataset.train_data,
            dataset.train_label,
            generator,
            device,
            int(run_options["batch_size"]),
            int(run_options.get("diagnostic_train_limit", 20000)),
            noise_type,
            std1,
            float(run_options.get("fedfed_clip_norm", 0.0)),
            run_options.get("fedfed_noise_shape", "paper"),
        )
        test_tensors, test_stats = build_feature_tensors(
            dataset.test_data,
            dataset.test_label,
            generator,
            device,
            int(run_options["batch_size"]),
            int(run_options.get("diagnostic_test_limit", 10000)),
            noise_type,
            std1,
            float(run_options.get("fedfed_clip_norm", 0.0)),
            run_options.get("fedfed_noise_shape", "paper"),
        )
        for source, stats in [("train", train_stats), ("test", test_stats)]:
            row = {"variant": name, "source": source, "noise_type": noise_type, "std1": std1, "std2": std2, "clip_norm": run_options.get("fedfed_clip_norm", 0.0)}
            for stat_name, values in stats.items():
                for key, value in values.items():
                    row["{}_{}".format(stat_name, key)] = value
            norm_rows.append(row)
        x_train = TensorDataset(train_tensors["x"], train_tensors["y"])
        dynamic_noise_train = use_dynamic_noise_train(run_options)
        xp_train = protected_dataset(train_tensors, run_options, dynamic_noise_train)
        x_test = TensorDataset(test_tensors["x"], test_tensors["y"])
        xp_test = TensorDataset(test_tensors["xp"], test_tensors["y"])
        _, x_metrics = train_classifier(run_options, x_train, x_test, device, name + "_x_to_x")
        _, xp_metrics = train_classifier(run_options, xp_train, xp_test, device, name + "_xp_to_xp")
        mixed_model, mixed_metrics = train_classifier(
            run_options,
            ConcatDataset([x_train, xp_train]),
            x_test,
            device,
            name + "_x_plus_xp_to_x",
        )
        mia = run_shadow_mia(mixed_model, train_tensors, test_tensors, run_options, device)
        inversion = run_inversion(
            train_tensors,
            test_tensors,
            run_options,
            device,
            os.path.join(fig_dir, "inversion_{}.png".format(name)),
        )
        row = {
            "variant": name,
            "noise_type": noise_type,
            "std1": std1,
            "std2": std2,
            "clip_norm": run_options.get("fedfed_clip_norm", 0.0),
            "noise_shape": run_options.get("fedfed_noise_shape", "paper"),
            "dynamic_noise_train": dynamic_noise_train,
            "ce_on_noisy_xs": ce_on_noisy,
            "x_to_x_final_acc": x_metrics["final_acc"],
            "xp_to_xp_final_acc": xp_metrics["final_acc"],
            "x_plus_xp_to_x_final_acc": mixed_metrics["final_acc"],
            "x_plus_xp_to_x_best_acc": mixed_metrics["best_acc"],
        }
        row.update(mia)
        row.update(inversion)
        summary_rows.append(row)
        with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)
        write_csv(os.path.join(out_root, "summary.csv"), summary_rows)
        write_csv(os.path.join(out_root, "norm_stats.csv"), norm_rows)

    with open(os.path.join(out_root, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "options": options,
            "variants": variants,
            "duration_min": round((time.time() - started) / 60.0, 2),
        }, f, indent=2)
    print("PROBE_OUT", out_root)


if __name__ == "__main__":
    main()
