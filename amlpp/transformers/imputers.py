from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from typing import List

import pandas as pd
import numpy as np

from ._base_transform import BaseTransform

##############################################################################
class Imputer(BaseTransform):
    """Base Imputer class
    Parameters
    ----------
    model: IterativeImputer or SimpleImputer
        Model Iterative or Simple
    columns: List [str] or None = None
        The name of the columns in which the filling will be carried out
    params: dict = {}
        Parameters for fillers
    """
    def __init__(self, model:IterativeImputer or SimpleImputer, 
                       columns:List[str] or None = None,
                       params:dict = {}):
        super().__init__({'columns':columns, **params})
        self.columns = columns
        self.current_columns = columns
        self.imputer = model(**params, random_state=42)

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        del_columns = []
        for column in X:
            if len(X.loc[X[column].isnull()]) == len(X):
                del X[column]
                del_columns.append(column)

        self.current_columns = self.columns if not self.columns is None else X.columns
        self.current_columns = [col for col in self.current_columns if not col in del_columns]
        self.imputer.fit(X[self.current_columns])
        return self

    def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = None):
        X_transform = pd.DataFrame(self.imputer.transform(X[self.current_columns]), columns = self.current_columns)
        X[self.current_columns] = X_transform[self.current_columns]
        for column in self.columns if not self.columns is None else X.columns:
            if column in self.current_columns:
                X[column] = X_transform[column].values
            else:
                del X[column]
        return X



class ImputerValue(Imputer):
    """ Imputer based on prediction
    inserted values based on the rest of the data available
    Parameters
    ----------
    columns: List [str] or None = None
        The name of the columns in which the filling will be carried out
    params: dict = {}
        Parameters for fillers
    """
    def __init__(self, columns:List[str] or None = None, **params):
        params = {'strategy':'mean', 'missing_values':np.nan, **params} 
        super().__init__(SimpleImputer, columns, params)


class ImputerIterative(Imputer):
    """ Imputer based on simple values
    inserted values based on the rest of the data available
    Parameters
    ----------
    columns: List [str] or None = None
        The name of the columns in which the filling will be carried out
    params: dict = {}
        Parameters for fillers
    """
    def __init__(self, columns:List[str] or None = None, **params):
        params = {'max_iter':10, 'initial_strategy':'mean', 'missing_values':np.nan, **params} 
        super().__init__(IterativeImputer, columns, params)