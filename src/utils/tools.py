import random

import numpy as np
import torch


def get_runtime_device(options=None):
    options = options or {}
    use_accelerator = bool(options.get('gpu', False))
    if not use_accelerator:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def accelerator_available():
    if torch.cuda.is_available():
        return True
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return True
    return False


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(options=None):
    options = options or {}
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(options.get('torch_cudnn_benchmark', True))


def resolve_heterogeneity_options(options=None):
    options = dict(options or {})
    if not options.get('unify_heterogeneity_alpha', False):
        return options

    alpha = float(options.get('dirichlet_alpha', 0.3))
    alpha = max(alpha, 1e-6)

    # Keep old option compatibility. The paper-style LDA partition no longer
    # uses quantity_skew_beta inside Dirichlet splits.
    options['quantity_skew_beta'] = alpha

    feature_anchor = max(float(options.get('feature_alpha_anchor', 0.1)), 1e-6)
    feature_strength = min(1.0, feature_anchor / alpha)

    max_scale_delta = float(options.get('feature_max_scale_delta', 0.15))
    max_bias_std = float(options.get('feature_max_bias_std', 0.05))
    max_noise_std = float(options.get('feature_max_noise_std', 0.05))

    scale_delta = max_scale_delta * feature_strength
    options['feature_scale_low'] = 1.0 - scale_delta
    options['feature_scale_high'] = 1.0 + scale_delta
    options['feature_bias_std'] = max_bias_std * feature_strength
    options['feature_noise_std'] = max_noise_std * feature_strength
    options['feature_unified_strength'] = feature_strength
    return options


def _split_by_counts(indices, counts):
    assignments = []
    start = 0
    for count in counts:
        end = start + count
        assignments.append(indices[start:end].tolist())
        start = end
    return assignments


def _ensure_min_samples(assignments, min_samples):
    if min_samples <= 0:
        return assignments

    lengths = [len(items) for items in assignments]#每个客户端的数据量
    for client_id, current_len in enumerate(lengths):#遍历每个客户端，确保每个客户端的数据量至少为min_samples
        while current_len < min_samples:#如果当前客户端的数据量小于 min_samples，就继续移动数据
            donor_id = int(np.argmax(lengths))#找到数据量最大的客户端
            if donor_id == client_id or lengths[donor_id] <= min_samples:#如果 donor_id 就是当前客户端，或者 donor_id 的数据量已经小于等于 min_samples，就跳出循环
                break
            assignments[client_id].append(assignments[donor_id].pop())#把 donor_id 的数据移动到 client_id
            lengths[client_id] += 1#client_id 的数据量增加1
            lengths[donor_id] -= 1#donor_id 的数据量减少1
            current_len += 1#current_len 增加1
    return assignments


def _sample_client_capacities(client_num, total_num, beta, min_samples):
    if client_num * min_samples > total_num:
        raise ValueError('min_samples_per_client is too large for the current dataset size.')

    base = np.full(client_num, min_samples, dtype=int)
    remaining = total_num - base.sum()
    if remaining <= 0:
        return base

    weights = np.random.dirichlet(np.full(client_num, beta))
    extra = np.random.multinomial(remaining, weights)
    return base + extra


def _build_iid_partition(train_labels, client_num, min_samples, enable_quantity_skew, quantity_skew_beta):
    shuffled_indices = np.random.permutation(len(train_labels))
    if enable_quantity_skew:
        counts = _sample_client_capacities(client_num, len(train_labels), quantity_skew_beta, min_samples)
    else:
        counts = np.full(client_num, len(train_labels) // client_num, dtype=int)
        counts[:len(train_labels) % client_num] += 1
    return _split_by_counts(shuffled_indices, counts)

#模拟label skew和quantity skew两种数据异质性
def _build_dirichlet_partition_once(labels, client_num, alpha, enable_quantity_skew,
                                    dirichlet_balance=False, dirichlet_min_p=None):
    assignments = [[] for _ in range(client_num)]
    total_num = len(labels)
    for cls in np.unique(labels):
        class_indices = np.where(labels == cls)[0]
        np.random.shuffle(class_indices)
        class_weights = np.random.dirichlet(np.repeat(alpha, client_num))
        if dirichlet_balance:
            sorted_by_weight = np.argsort(class_weights, axis=0)
            if cls != 0:
                used_counts = np.array([len(items) for items in assignments])
                sorted_by_usage = np.argsort(used_counts, axis=0)
                class_weights[sorted_by_usage] = class_weights[sorted_by_weight[::-1]]
        elif enable_quantity_skew:
            class_weights = np.array([
                weight * (len(items) < total_num / client_num)
                for weight, items in zip(class_weights, assignments)
            ])
        if dirichlet_min_p is not None:
            class_weights += float(dirichlet_min_p)
        class_weights = class_weights / class_weights.sum()
        split_points = (np.cumsum(class_weights) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, split_points)
        for client_id, split in enumerate(splits):
            assignments[client_id].extend(split.tolist())
    return assignments


def _build_dirichlet_partition(train_labels, client_num, alpha, min_samples, enable_quantity_skew,
                               quantity_skew_beta, dirichlet_balance=False, dirichlet_min_p=None):
    labels = np.asarray(train_labels) #np.asarray() 将输入转换为NumPy数组
    class_num = len(np.unique(labels))
    paper_min_samples = class_num if enable_quantity_skew else min_samples
    min_required = max(int(min_samples), int(paper_min_samples))
    max_retries = 100
    for _ in range(max_retries):
        assignments = _build_dirichlet_partition_once(
            labels,
            client_num,
            alpha,
            enable_quantity_skew,
            dirichlet_balance=dirichlet_balance,
            dirichlet_min_p=dirichlet_min_p,
        )
        if min_required <= 0 or min(len(items) for items in assignments) >= min_required:
            for items in assignments:
                np.random.shuffle(items)
            return assignments

    raise ValueError(
        'Failed to build a Dirichlet partition satisfying min_samples_per_client={} '
        'after {} retries. Consider increasing dirichlet_alpha, reducing min_samples_per_client, '
        'or reducing num_of_clients.'.format(min_required, max_retries)
    )


def get_each_client_data_index(train_labels, client_num, options=None):
    options = options or {}
    strategy = options.get('partition_strategy', 'dirichlet')
    min_samples = options.get('min_samples_per_client', 0)
    enable_quantity_skew = options.get('enable_quantity_skew', False) #是否启用数量偏斜
    quantity_skew_beta = options.get('quantity_skew_beta', 1.0) #数量偏斜参数

    if strategy == 'iid':
        return _build_iid_partition(
            train_labels,
            client_num,
            min_samples,
            enable_quantity_skew,
            quantity_skew_beta,
        )

    if strategy == 'dirichlet':
        return _build_dirichlet_partition(
            train_labels,
            client_num,
            options.get('dirichlet_alpha', 0.3),
            min_samples,
            enable_quantity_skew,
            quantity_skew_beta,
            dirichlet_balance=options.get('dirichlet_balance', False),
            dirichlet_min_p=options.get('dirichlet_min_p', None),
        )

    raise ValueError('Unsupported partition strategy: {}'.format(strategy))


def build_client_feature_skews(client_num, options):
    if not options.get('enable_feature_skew', False):
        return [None] * client_num

    low = options.get('feature_scale_low', 1.0)
    high = options.get('feature_scale_high', 1.0)
    bias_std = options.get('feature_bias_std', 0.0)
    noise_std = options.get('feature_noise_std', 0.0)

    skews = []
    for _ in range(client_num):
        skews.append({
            'scale': float(np.random.uniform(low, high)),
            'bias': float(np.random.normal(0.0, bias_std)),
            'noise_std': float(max(noise_std, 0.0)),
        })
    return skews


def apply_feature_skew(data, skew):
    if skew is None:
        return data.copy()

    transformed = data.astype(np.float32).copy()
    transformed = transformed * skew['scale'] + skew['bias']
    if skew['noise_std'] > 0:
        transformed = transformed + np.random.normal(
            0.0,
            skew['noise_std'],
            size=transformed.shape,
        ).astype(np.float32)
    return np.clip(transformed, 0.0, 1.0)
