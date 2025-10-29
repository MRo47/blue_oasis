from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from bird_classifier.logging_config import logger


def filter_classes(df: DataFrame, min_sample_size: int = 1):
    """
    Filters out classes with less than min_sample_size samples from a dataframe.

    Parameters
    ----------
    df : DataFrame
        The dataframe to filter.
    min_sample_size : int
        The minimum number of samples required for a class to be kept.

    Returns
    -------
    DataFrame
        The filtered dataframe.

    Notes
    -----
    If min_sample_size is less than or equal to 1, the dataframe is returned unchanged.
    """
    df = df.copy()
    if min_sample_size <= 1:
        logger.warning("Nothing to filter, min_sample_size <= 1")
        return df
    class_counts = df['Species eBird Code'].value_counts()
    valid_classes = class_counts[class_counts >= min_sample_size].index
    num_classes = df['Species eBird Code'].nunique()
    logger.warning(f"Filtering out {num_classes - len(valid_classes)}/{num_classes}, "
                f"classes with less than {min_sample_size} samples.")
    logger.info(f"New dataset will have {len(valid_classes)} classes")
    return df[df['Species eBird Code'].isin(valid_classes)]

def stratified_group_split(
    df: DataFrame,
    label_id: str,
    group_id: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 37
) -> Tuple[DataFrame, DataFrame, DataFrame]:

    """
    Splits a dataframe into training, validation, and test sets, ensuring no
    group leakage, even with classes that are hard to stratify.

    The logic is as follows:
    1.  Identify "problematic" classes (those with too few unique groups).
    2.  Partition the entire dataset by GROUPS:
        - `df_stratifiable`: Contains data from groups with ONLY well-behaved classes.
        - `df_problematic`: Contains data from groups with AT LEAST ONE problematic class.
        These two dataframes have no groups in common.
    3.  Split `df_stratifiable` using a nested StratifiedGroupKFold.
    4.  Split `df_problematic` using a nested GroupKFold (prioritizing group integrity).
    5.  Concatenate the respective splits to form the final train, val, and test sets.

    Parameters
    ----------
    df : DataFrame
        The dataframe to split.
    label_id : str
        The column containing the class labels.
    group_id : str
        The column containing the group labels.
    test_size : float
        The proportion of the dataframe to use for the test set.
    val_size : float
        The proportion of the dataframe (after removing the test set) to use for the validation set.
    random_state : int
        The random state to use for the split.

    Returns
    -------
    Tuple[DataFrame, DataFrame, DataFrame]
        The train, validation and test dataframes.
    
    Note: The final split sizes are an approximation of the requested test_size and val_size.
    Due to the constraint of keeping groups intact, the actual proportions may vary.
    """
    df = df.copy()
    
    # 1. identify problematic classes (too few unique groups per class)
    # A class is problematic if it doesn't appear in enough unique groups to be split.
    # We need at least n_splits for the test set and n_splits for the val set.
    n_splits_test = int(np.ceil(1.0 / test_size))
    val_size_of_train = val_size / (1 - test_size)
    n_splits_val = int(np.ceil(1.0 / val_size_of_train))
    
    class_group_counts = df.groupby(label_id)[group_id].nunique()
    # A class is problematic if it can't be safely split in either stage.
    problematic_classes = class_group_counts[
        (class_group_counts < n_splits_test) | (class_group_counts < n_splits_val)
    ].index

    # 2. partition df by groups
    if not problematic_classes.empty:
        logger.warning(
            f"Identified {len(problematic_classes)} problematic classes with too few "
            f"unique groups for reliable stratification."
        )
        # Find all groups that contain any of these problematic classes
        problematic_groups = df[df[label_id].isin(problematic_classes)][group_id].unique()
        
        logger.info(f"Assigning {len(problematic_groups)} groups containing these classes to the 'problematic' split path.")
        
        df_problematic = df[df[group_id].isin(problematic_groups)]
        df_stratifiable = df[~df[group_id].isin(problematic_groups)]
    else:
        logger.info("No problematic classes found. Using a standard stratified split.")
        df_stratifiable = df
        df_problematic = pd.DataFrame(columns=df.columns)

    all_splits = {'train': [], 'val': [], 'test': []}

    # 3. split stratifiable data
    if not df_stratifiable.empty:
        sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=random_state)
        X_s = df_stratifiable.index
        y_s = df_stratifiable[label_id]
        groups_s = df_stratifiable[group_id]

        train_val_idx_s, test_idx_s = next(sgkf_test.split(X_s, y_s, groups_s))
        train_val_s = df_stratifiable.iloc[train_val_idx_s]
        all_splits['test'].append(df_stratifiable.iloc[test_idx_s])

        sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=random_state)
        X_tv_s = train_val_s.index
        y_tv_s = train_val_s[label_id]
        groups_tv_s = train_val_s[group_id]

        train_idx_s, val_idx_s = next(sgkf_val.split(X_tv_s, y_tv_s, groups_tv_s))
        all_splits['train'].append(train_val_s.iloc[train_idx_s])
        all_splits['val'].append(train_val_s.iloc[val_idx_s])

    # 4. split problematic data
    if not df_problematic.empty:
        # Use GroupKFold here since stratification is not reliable
        gkf_test = GroupKFold(n_splits=n_splits_test)
        X_p = df_problematic.index
        groups_p = df_problematic[group_id]
        
        train_val_idx_p, test_idx_p = next(gkf_test.split(X_p, groups=groups_p))
        train_val_p = df_problematic.iloc[train_val_idx_p]
        all_splits['test'].append(df_problematic.iloc[test_idx_p])

        if not train_val_p.empty:
            gkf_val = GroupKFold(n_splits=n_splits_val)
            X_tv_p = train_val_p.index
            groups_tv_p = train_val_p[group_id]

            # Need to handle case where there are not enough groups for the split
            if train_val_p[group_id].nunique() < n_splits_val:
                logger.warning("Not enough unique groups in problematic set for validation split. Assigning all to training.")
                all_splits['train'].append(train_val_p)
            else:
                train_idx_p, val_idx_p = next(gkf_val.split(X_tv_p, groups=groups_tv_p))
                all_splits['train'].append(train_val_p.iloc[train_idx_p])
                all_splits['val'].append(train_val_p.iloc[val_idx_p])

    # 5. concatenate final splits
    train_df = pd.concat(all_splits['train'], ignore_index=True)
    val_df = pd.concat(all_splits['val'], ignore_index=True)
    test_df = pd.concat(all_splits['test'], ignore_index=True)
    
    logger.info(f"Final split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 6. verify
    train_groups = set(train_df[group_id].unique())
    val_groups = set(val_df[group_id].unique())
    test_groups = set(test_df[group_id].unique())

    assert len(train_groups.intersection(val_groups)) == 0, "Leakage: train/val groups overlap!"
    assert len(train_groups.intersection(test_groups)) == 0, "Leakage: train/test groups overlap!"
    assert len(val_groups.intersection(test_groups)) == 0, "Leakage: val/test groups overlap!"
    logger.info("Successfully verified no group overlap between final splits.")

    return train_df, val_df, test_df

def main():
    from pathlib import Path

    annotations_csv = Path("~/data/kenya_birds/annotations.csv").expanduser()
    df = pd.read_csv(annotations_csv)

    df = filter_classes(df, min_sample_size=3)

    train_df, val_df, test_df = stratified_group_split(
        df,
        label_id='Species eBird Code',
        group_id='Filename',
        test_size=0.2,
        val_size=0.2,
    )
    
    logger.info("--- Split Results ---")
    logger.info(f"Train DF shape: {train_df.shape}")
    logger.info(f"Validation DF shape: {val_df.shape}")
    logger.info(f"Test DF shape: {test_df.shape}")

if __name__ == "__main__":
    main()