from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler as MMScaler
import pandas as pd 

class MinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.scaler = MMScaler()

    def fit(self, X:pd.DataFrame) -> None:
        self.scaler.fit(X)

    def transform(self, X:pd.DataFrame) -> None:
        X_scaled = self.scaler.transform(X)
        return X_scaled