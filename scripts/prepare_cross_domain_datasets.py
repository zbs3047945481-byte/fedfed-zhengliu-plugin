#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from getdata import GetDataSet, get_dataset_defaults


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["gtsrb", "pathmnist", "plantvillage"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--dataset-train-fraction", type=float, default=0.8)
    args = parser.parse_args()

    for dataset_name in args.datasets:
        defaults = get_dataset_defaults(dataset_name)
        options = {
            "data_root": args.data_root,
            "image_size": args.image_size,
            "input_channels": defaults.get("input_channels", 3),
            "num_classes": defaults.get("num_classes", 10),
            "dataset_split_seed": args.seed,
            "dataset_train_fraction": args.dataset_train_fraction,
            "dataset_cache": True,
            "seed": args.seed,
        }
        dataset = GetDataSet(dataset_name, options)
        print(
            "{} ready: train={} test={} shape={} classes={}".format(
                dataset_name,
                dataset.train_data_size,
                dataset.test_data_size,
                dataset.train_data.shape[1:],
                dataset.num_classes,
            )
        )


if __name__ == "__main__":
    main()
