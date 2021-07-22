from ._base import BaseTransform

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from typing import List

import pandas as pd
import numpy as np

class Imputer(BaseTransform):
    def __init__(self, model:IterativeImputer or SimpleImputer, 
                       columns:List[str] or None = None,
                       params:dict = {}):
        super().__init__({'columns':columns, **params})
        self.columns = columns
        self.current_columns = columns
        self.imputer = model(**params, random_state=42)

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        for column in X:
            if len(X.loc[X[column].isnull()]) == len(X):
                X[column] = [0 for i in range(len(X))]
        self.current_columns = self.columns if self.columns != None else X.columns
        self.imputer.fit(X[self.current_columns])
        return self

    def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = None):
        X_transform = pd.DataFrame(self.imputer.transform(X[self.current_columns]), columns = self.current_columns)
        for column in self.current_columns:
            X[column] = X_transform[column].values
        return X

class ImputerValue(Imputer):
    def __init__(self, columns:List[str] or None = None, **params):
        params = params if params != {} else \
                    {'strategy':'mean', 'missing_values':np.nan} 
        super().__init__(SimpleImputer, columns, params)


class ImputerIterative(Imputer):
    def __init__(self, columns:List[str] or None = None, **params):
        params = params if params != {} else \
            {'max_iter':10, 'initial_strategy':'mean', 'missing_values':np.nan} 
        super().__init__(IterativeImputer, columns, params)