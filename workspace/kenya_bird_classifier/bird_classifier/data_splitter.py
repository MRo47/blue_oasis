import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def stratified_group_split(df: DataFrame, label_id: str, group_id: str,
                           min_valid_classes: int = 20, n_splits: int = 5, 
                           val_ratio: float = 0.2, shuffle: bool = True,
                           random_state: int = 101):
  """
  Split a dataframe into training, validation, and test sets using a stratified group k-fold approach.

  Parameters
  ----------
  df : DataFrame
      The dataframe to be split.
  label_id : str
      The name of the column containing the labels.
  group_id : str
      The name of the column containing the group ids.
  min_valid_classes : int, optional (default=20)
      The minimum number of samples required per class for the class to be considered valid.
  n_splits : int, optional (default=5)
      The number of folds for the stratified group k-fold split.
  val_ratio : float, optional (default=0.2)
      The proportion of the training set to be used for validation.
  shuffle : bool, optional (default=True)
      Whether to shuffle the data before splitting.
  random_state : int, optional (default=101)
      The random seed to be used for shuffling.

  Returns
  -------
  train_df : DataFrame
      The training set.
  val_df : DataFrame
      The validation set.
  test_df : DataFrame
      The test set.
  """
  freq_classes = df[label_id].value_counts()
  valid_species = freq_classes[freq_classes >= min_valid_classes].index
  df = df[df[label_id].isin(valid_species)]

  sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, 
                              random_state=random_state)

  X = df.index
  y = df[label_id]
  groups = df[group_id]

  # first train test split 1/5
  for train_idx, test_idx in sgkf.split(X, y, groups):
      break

  train_df = df.iloc[train_idx].copy()
  test_df = df.iloc[test_idx].copy()

  class_counts = train_df[label_id].value_counts()
  rare_classes = class_counts[class_counts < 2].index.tolist()

  if rare_classes:
        print(f"Found {len(rare_classes)} rare classes (<2 samples). "
              "Keeping them in training set.")

  # Separate rare vs. regular classes
  regular_df = train_df[~train_df[label_id].isin(rare_classes)].copy()
  rare_df = train_df[train_df[label_id].isin(rare_classes)].copy()

  stratifiable = True
  reg_class_counts = regular_df[label_id].value_counts()

  if (reg_class_counts < 2).any() or len(reg_class_counts) < 2:
      stratifiable = False

  # validation split
  if stratifiable:
      try:
          reg_train_df, val_df = train_test_split(
              regular_df,
              test_size=val_ratio,
              stratify=regular_df[label_id],
              random_state=random_state,
          )
      except ValueError as e:
          print(f"Stratified split failed ({e}), falling back to random split.")
          reg_train_df, val_df = train_test_split(
              regular_df,
              test_size=val_ratio,
              random_state=random_state,
          )
  else:
      print("Not enough samples per class for stratified split — using random split.")
      reg_train_df, val_df = train_test_split(
          regular_df,
          test_size=val_ratio,
          random_state=random_state,
      )

  # Merge rare classes back into training
  train_df = pd.concat([reg_train_df, rare_df], ignore_index=True)

  print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
  print(f"Unique groups -> train: {train_df[group_id].nunique()}, test: {test_df[group_id].nunique()}")
  missing_in_test = set(df[label_id].unique()) - set(test_df[label_id].unique())
  if missing_in_test:
      print(f"Warning: Missing {len(missing_in_test)} classes in test split")
  
  return train_df, val_df, test_df

def main():
  from pathlib import Path


  annotations_csv = Path("~/data/kenya_birds/annotations.csv").expanduser()
  df = pd.read_csv(annotations_csv)

  stratified_group_split(df, label_id='Species eBird Code', group_id='Filename')


if __name__ == "__main__":
   main()
