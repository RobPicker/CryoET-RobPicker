import math
import os
import random
import importlib
import importlib.util
from copy import copy

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def read_df(fn):
    if 'parquet' in fn:
        df = pd.read_parquet(fn, engine="fastparquet")
    else:
        df = pd.read_csv(fn)
    return df


def get_data(cfg):
    if type(cfg.train_df) == list:
        cfg.train_df = cfg.train_df[cfg.fold]
    print(f"reading {cfg.train_df}")
    df = read_df(cfg.train_df)

    test_df = None
    if getattr(cfg, "test", False):
        test_df = read_df(cfg.test_df)

    meta_df = None
    if getattr(cfg, "meta_df", None):
        if type(cfg.meta_df) == list:
            cfg.meta_df = cfg.meta_df[cfg.fold]
        meta_df = read_df(cfg.meta_df)

    if getattr(cfg, "test_df", None):
        if type(cfg.test_df) == list:
            cfg.test_df = cfg.test_df[cfg.fold]
        test_df = read_df(cfg.test_df)

    if meta_df is None:
        if cfg.fold == -1:
            meta_df = df[df["fold"] == 0]
        else:
            meta_df = df[df["fold"] == cfg.fold]
        train_df = df[df["fold"] != cfg.fold]
    else:
        train_df = df

    val_df = None
    if test_df is not None:
        val_df = test_df

    return train_df, meta_df, val_df


def resolve_config_module(name: str) -> str:
    """Resolve config module path for importlib."""
    if "." in name:
        return name
    return f"robpicker.configs.{name}"


def load_config(config_name_or_path):
    """
    Load config from either a module name or a file path.

    Returns (cfg, config_path) where config_path is the file path of the config.
    """
    if os.path.isfile(config_name_or_path) or os.path.isfile(config_name_or_path + '.py'):
        # Load from file path
        config_path = config_name_or_path if os.path.isfile(config_name_or_path) else config_name_or_path + '.py'
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return copy(config_module.cfg), config_path
    else:
        # Load from module
        module_name = resolve_config_module(config_name_or_path)
        config_module = importlib.import_module(module_name)
        return copy(config_module.cfg), config_module.__file__
