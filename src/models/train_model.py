

from src.features.missing_indicator import MissingIndicator
from src.features.cabin_only_letter import CabinOnlyLetter
from src.features.categorical_imputer_encoder import CategoricalImputerEncoder
from src.features.median_imputation import NumericalImputesEncoder
from src.features.rare_label_categorial import RareLabelCategoricalEncoder
from src.features.one_hot_encoder import OneHotEncoder
from src.features.min_max_scaler import MinMaxScaler



import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

SEED_MODEL = 42
NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']
CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']
TARGET = 'survived'

titanic_pipeline = Pipeline(
    [
        ('missing_indicator', MissingIndicator(NUMERICAL_VARS)),
        ('cabin_only_letter', CabinOnlyLetter('cabin')),
        ('categorical_imputer', CategoricalImputerEncoder(CATEGORICAL_VARS)),
        ('median_imputation', NumericalImputesEncoder(NUMERICAL_VARS)),
        ('rare_labels', RareLabelCategoricalEncoder(tol=0.02,  variables=CATEGORICAL_VARS)),
        ('dummy_vars', OneHotEncoder(CATEGORICAL_VARS)),
        ('scaling', MinMaxScaler()),
        ('log_reg', LogisticRegression(C=0.0005, class_weight='balanced', random_state=SEED_MODEL))
        
    ]
)



data = 'src/models/cleaned_data'
# Loading data from specific url
df = pd.read_csv(data)

X_train, X_test, y_train, y_test = train_test_split( df.drop(TARGET, axis=1), df[TARGET], test_size=0.2, random_state=SEED_MODEL)


titanic_pipeline.fit(X_train, y_train)

#titanic_pipeline.fit(X_train, y_train)

#titanic_pipeline.score(X_test, y_test)

#class_pred = titanic_pipeline.predict(X_test)
#proba_pred = titanic_pipeline.predict_proba(X_test)[:,1]
#print('test roc-auc : {}'.format(roc_auc_score(y_test, proba_pred)))
#print('test accuracy: {}'.format(accuracy_score(y_test, class_pred)))