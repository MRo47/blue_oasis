import logging
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def stratified_group_split(
    df: DataFrame,
    label_id: str,
    group_id: str,
    min_samples_per_class: int = 5,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 37
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Splits a dataframe into training, validation, and test sets using a nested
    stratified group k-fold approach to prevent data leakage.

    The process is as follows:
    1. Filter out classes with fewer than `min_samples_per_class`.
    2. Split the data into a (train + val) set and a test set, ensuring groups are not split.
    3. Split the (train + val) set into a final train set and a validation set,
        again ensuring groups are not split.

    Parameters
    ----------
    df : DataFrame
        The dataframe to be split.
    label_id : str
        The name of the column containing the labels for stratification.
    group_id : str
        The name of the column containing the group ids.
    min_samples_per_class : int, optional
        Minimum number of samples required for a class to be included, by default 5.
    test_size : float, optional
        The proportion of the dataset to be used for the test set, by default 0.2.
    val_size : float, optional
        The proportion of the (train+val) set to be used for validation, by default 0.2.
    random_state : int, optional
        The random seed for reproducibility, by default 101.

    Returns
    -------
    Tuple[DataFrame, DataFrame, DataFrame]
        A tuple containing the training, validation, and test dataframes.
    """
    # filter out rare classes
    class_counts = df[label_id].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index

    if len(valid_classes) < class_counts.shape[0]:
        logging.warning(f"Filtered out {class_counts.shape[0] - len(valid_classes)} classes "
                        f"with < {min_samples_per_class} samples.")

    df_filtered = df[df[label_id].isin(valid_classes)].copy()

    X = df_filtered.index
    y = df_filtered[label_id]
    groups = df_filtered[group_id]

    # first split: (train + val) + test
    # Calculate n_splits needed to achieve the desired test_size
    n_splits_test = int(np.ceil(1.0 / test_size))
    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=random_state)

    try:
        train_val_idx, test_idx = next(sgkf_test.split(X, y, groups))
    except ValueError as e:
        logging.error(f"Could not perform initial split. Maybe test_size is too high or "
                        f"not enough groups for stratification. Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    train_val_df = df_filtered.iloc[train_val_idx]
    test_df = df_filtered.iloc[test_idx]

    # create the second split: train + val from the train_val_df
    X_train_val = train_val_df.index
    y_train_val = train_val_df[label_id]
    groups_train_val = train_val_df[group_id]

    # calculate n_splits for the validation set
    n_splits_val = int(np.ceil(1.0 / val_size))
    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=random_state)

    try:
        train_idx, val_idx = next(sgkf_val.split(X_train_val, y_train_val, groups_train_val))
    except ValueError as e:
        # this can happen if a class in train_val_df has only one group
        logging.warning(f"Stratified group split for validation failed: {e}. "
                        "The validation set will be empty, and the samples will remain in training.")
        # in this case, we can just return train_val_df as train_df and an empty val_df
        train_df = train_val_df
        val_df = pd.DataFrame(columns=df.columns)
    else:
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

    logging.info(f"Original dataset size: {len(df)}")
    logging.info(f"Filtered dataset size: {len(df_filtered)}")
    logging.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # sanity check for group overlap (should be zero)
    train_groups = set(train_df[group_id])
    val_groups = set(val_df[group_id])
    test_groups = set(test_df[group_id])

    assert len(train_groups.intersection(val_groups)) == 0, "Data Leakage: Groups overlap between train and val!"
    assert len(train_groups.intersection(test_groups)) == 0, "Data Leakage: Groups overlap between train and test!"
    assert len(val_groups.intersection(test_groups)) == 0, "Data Leakage: Groups overlap between val and test!"
    logging.info("Successfully verified no group overlap between splits.")

    # check for missing classes
    original_classes = set(df_filtered[label_id].unique())
    for split_name, split_df in (('Train', train_df), ('Validation', val_df), ('Test', test_df)):
        missing = original_classes - set(split_df[label_id].unique())
        if missing and not split_df.empty:
            logging.warning(f"{split_name} split is missing {len(missing)} classes.")

    return train_df, val_df, test_df

def main():
  from pathlib import Path

  annotations_csv = Path("~/data/kenya_birds/annotations.csv").expanduser()
  df = pd.read_csv(annotations_csv)

  train_df, val_df, test_df = stratified_group_split(
      df,
      label_id='Species eBird Code',
      group_id='Filename',
      test_size=0.2, # 20% for test
      val_size=0.25,  # 25% of the remaining 80% will be validation (i.e., 20% of original)
      min_samples_per_class=5
  )
  
  print("\n--- Split Results ---")
  print(f"Train DF shape: {train_df.shape}")
  print(f"Validation DF shape: {val_df.shape}")
  print(f"Test DF shape: {test_df.shape}")


if __name__ == "__main__":
   main()
