from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as oneHot
import pandas as pd 
from typing import List

class OneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str]):
        self.encoder = oneHot(handle_unknown = 'ignore',drop='first')
        self.variables = variables

    def fit(self, X:pd.DataFrame) -> None:
        self.encoder.fit(X[self.variables])

    def transform(self, X:pd.DataFrame) -> None:
        X[self.encoder.get_feature_names_out(self.variables)] = self.encoder.transform(X[self.variables]).toarray()
        X.drop(self.variables, axis=1, inplace=True)
        return X

