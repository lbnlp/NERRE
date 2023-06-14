"""Cross-validation evaluation of fine-tuned GPT3 model."""

import argparse
import os
import numpy as np


def prepare_single_fold_dataset(dataset_path, train_indices, val_indices, experiment_dir):
    """Creates training and validation jsonl files from dataset."""
    with open(dataset_path, "r") as f:
        lines = f.readlines()
    train_lines = [lines[i] for i in train_indices]
    val_lines = [lines[i] for i in val_indices]
    with open(os.path.join(experiment_dir, "train.jsonl"), "w") as f:
        f.writelines(train_lines)
    with open(os.path.join(experiment_dir, "val.jsonl"), "w") as f:
        f.writelines(val_lines)


def prepare_datasets(dataset_path, n_folds, train_size=0.90, experiment_dir="experiments", seed=42):
    """Creates training and validation jsonl files from dataset with random selection."""
    with open(dataset_path, "r") as f:
        lines = f.readlines()
    n_lines = len(lines)
    n_train = int(n_lines * train_size)

    np.random.seed(seed)
    for i in range(n_folds):
        train_indices = np.random.choice(n_lines, n_train, replace=False)
        val_indices = np.setdiff1d(np.arange(n_lines), train_indices)
        fold_dir = os.path.join(experiment_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        prepare_single_fold_dataset(dataset_path, train_indices, val_indices, fold_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to jsonlines dataset containing annotations.")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of runs for validation.")
    parser.add_argument("--train_size", type=float, default=0.90, help="Size of training set (as fraction of total number). Default is 0.9")
    parser.add_argument("--experiment_dir", type=str, default="experiments_msda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    prepare_datasets(args.dataset_path, args.n_folds, args.train_size, args.experiment_dir)
