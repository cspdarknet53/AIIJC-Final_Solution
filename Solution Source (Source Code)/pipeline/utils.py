import random
import json
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


JSON_INDENT = 2


def load_json_file(path: str) -> Any:
    """Loading a json file.

    :param path: path to the json file
    :return: json file
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def dump_to_json_file(data: Any, path: str) -> None:
    """Dumping data to a json file.

    :param data: data for dumping
    :param path: path to the saved json file
    """
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=JSON_INDENT)
        f.write('\n')


def set_global_seed(seed: int, is_cudnn_deterministic: bool = True) -> None:
    """Setting seed for reproducible results.

    :param seed: seed number (no matter which one)
    :param is_cudnn_deterministic: is the algorithm determined on сuda or not
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = is_cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def split_data(
    df: pd.DataFrame,
    stratify: Optional[str],
    seed: int,
    test_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Division of the dataframe into a training and test one.

    :param df: dataframe for division
    :param stratify: if not None, data is split in a stratified fashion, using this as the class labels
    :param seed: seed number (no matter which one)
    :param test_size: test size
    :return: training and test dataframes
    """
    df = df.copy()
    if stratify is not None:
        df['count'] = -1
        count_df = df.groupby(stratify).count()
        stratify_names = count_df[count_df['count'] >= 2].index.values
        df = df.loc[df[stratify].isin(stratify_names)].reset_index(drop=True)
        stratify = df[stratify].values
        df = df.drop(columns='count')

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df
