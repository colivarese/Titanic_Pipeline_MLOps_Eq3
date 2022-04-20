import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

URL = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
# Loading data from specific url
df = pd.read_csv(URL)

# Uncovering missing data
df.replace("?", np.nan, inplace=True)
df["age"] = df["age"].astype("float")
df["fare"] = df["fare"].astype("float")


class GetFirstCabinEncoder(BaseEstimator, TransformerMixin):
    """_summary_

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """

    def __init__(self):
        return None

    def fit(self):
        return self  # pass

    def transform(self, df: pd.DataFrame):
        for row in df.itertuples():
            idx = row.Index
            try:
                df.at[idx, "cabin"] = df.at[idx, "cabin"].split()[0]
            except ValueError:
                df.at[idx, "cabin"] = np.nan
        return df


class GetTitleEncoder(BaseEstimator, TransformerMixin):
    """_summary_

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """

    def __init__(self):
        return None

    def fit(self):
        return None  # pass | self

    def transform(self, df: pd.DataFrame):
        for row in df.itertuples():
            line = df.at[row.Index, "name"]
            if re.search("Mrs", line):
                df.at[row.Index, "title"] = "Mrs"
            elif re.search("Mr", line):
                df.at[row.Index, "title"] = "Mr"
            elif re.search("Miss", line):
                df.at[row.Index, "title"] = "Miss"
            elif re.search("Master", line):
                df.at[row.Index, "title"] = "Master"
            else:
                df.at[row.Index, "title"] = "Other"
        return df


# esto se pondria en el pipeline
# get_first_cabin_encoder = GetFirstCabinEncoder()
# df = get_first_cabin_encoder.transform(df)
# title_encoder = GetTitleEncoder()
# Extract the title from 'name'
# df = title_encoder.transform(df)
