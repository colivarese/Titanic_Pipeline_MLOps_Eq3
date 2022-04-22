import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class MissingIndicator(BaseEstimator, TransformerMixin):

  def __init__(self, columnsList: List[str]):
    self.columnsList = columnsList

  def fit(self, x: pd.DataFrame):
    pass

  def transform(self, X: pd.DataFrame):
    for column in self.columnsList:
      X[f"{column}_nan"] = X[column].isnull().astype(int)
    return X