import pytest
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tol=0.02, variables: List[str] = None):
        self.tol =tol
        self.variables = variables
    
    def fit(self, X: pd.DataFrame):
        self.valid_labels_dict = {}
        for var in self.variables:
            t = X[var].value_counts() / X.shape[0]
            self.valid_labels_dict[var] = t[t>self.tol].index.tolist()

    def transform(self, X:pd.DataFrame):
        for var in self.variables:
            tmp = [col for col in X[var].unique() if col not in self.valid_labels_dict[var]]
            X[var] = X[var].replace(to_replace=tmp, value=len(tmp) * ['Rare'])
        return X


def GetRareLabelTestData():
    df = pd.read_csv('rare_label_csv_test')
    result = pd.read_csv('rare_labels_csv_result')
    return [(df, result)]

@pytest.mark.parametrize("df, result", GetRareLabelTestData())
def test_extract_only_letter(df,result):
    print(df)
    cat_vars = ['sex', 'cabin', 'embarked', 'title']
    rare_labels = RareLabelCategoricalEncoder(tol=0.02, variables=cat_vars)
    rare_labels.fit(df)
    test_result = rare_labels.transform(df)
    assert test_result.equals(result)




