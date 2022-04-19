import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class CategoricalImputerEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):   
        self.variables = variables
    
    def fit(self, X: pd.DataFrame):
        return self
    
    def transform(self, X: pd.DataFrame):
        X[self.variables] = X[self.variables].fillna('missing')
        return X

# esto es para poner en el pipeline
#categ_imputer = CategoricalImputerEncoder(variables=cat_vars)
#X_train = categ_imputer.transform(X_train)
#X_test = categ_imputer.transform(X_test)
